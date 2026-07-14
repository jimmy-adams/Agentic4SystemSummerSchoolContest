#!/usr/bin/env python3
"""Memory-bounded BigFormer inference with layer-major weight streaming.

The release BigFormer is larger than GPU memory.  A batch-major executor copies
all transformer weights once per input batch; this executor reverses the loops:
each block is uploaded once and reused for every micro-batch.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper


_PROFILE_ENV = "C3_PROFILE"
_ONE_GIB = 1024 ** 3


def external_model_size(onnx_path: str, model: Any | None = None) -> int:
    """Return the ONNX protobuf plus unique external-data file sizes."""
    path = Path(onnx_path).resolve()
    if model is None:
        model = onnx.load(str(path), load_external_data=False)

    total = path.stat().st_size
    locations = set()
    for tensor in model.graph.initializer:
        info = {entry.key: entry.value for entry in tensor.external_data}
        if info.get("location"):
            locations.add(info["location"])

    for location in locations:
        data_path = (path.parent / location).resolve()
        try:
            data_path.relative_to(path.parent)
        except ValueError as exc:
            raise ValueError(f"unsafe ONNX external-data path: {location}") from exc
        if data_path.exists():
            total += data_path.stat().st_size
    return total


def is_bigformer(onnx_path: str, model: Any | None = None) -> bool:
    """Identify the oversized transformer used by the C3.5 BigFormer case."""
    if "bigformer" in Path(onnx_path).name.lower():
        return True
    return external_model_size(onnx_path, model) > 10 * _ONE_GIB


class BigFormerStreamingExecutor:
    """Execute C3 BigFormer with FP16-resident weights and FP32 compute.

    The teammate v2 preload makes the 19 GB FP32 model fit as roughly 9.5 GB
    of FP16 GPU storage.  This executor additionally makes execution
    layer-major, so each block is expanded to FP32 only once instead of once
    per input batch.
    """

    _BLOCK_KEYS = (
        "qkv", "proj", "ff1", "ff2",
        "qkv_b", "proj_b", "ff1_b", "ff2_b",
        "ln1_w", "ln1_b", "ln2_w", "ln2_b",
    )

    def __init__(self, onnx_path: str, profile: bool = False):
        # NVIDIA_TF32_OVERRIDE=0 would defeat the selective FFN TF32 path.
        os.environ.pop("NVIDIA_TF32_OVERRIDE", None)
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("BigFormer streaming backend requires CUDA")

        self.torch = torch
        self.F = torch.nn.functional
        self.path = Path(onnx_path).resolve()
        self.base_dir = self.path.parent
        self.profile = profile or os.environ.get(_PROFILE_ENV) == "1"
        self.model = onnx.load(str(self.path), load_external_data=False)
        self.initializers = {tensor.name: tensor for tensor in self.model.graph.initializer}
        self.identity_map = {
            node.output[0]: node.input[0]
            for node in self.model.graph.node
            if node.op_type == "Identity" and node.input and node.output
        }
        self.output_names = [value.name for value in self.model.graph.output]
        self.blocks, self.head_weight_name = self._discover_blocks()
        if not self.blocks:
            raise ValueError("BigFormer graph has no discoverable transformer blocks")

        self.device = torch.device("cuda")
        self.shared_gpu: dict[str, Any] = {}
        self.resident_half: dict[str, Any] = {}
        self.stats = {
            "preload_seconds": 0.0,
            "copy_seconds": 0.0,
            "compute_seconds": 0.0,
            "head_seconds": 0.0,
            "weight_bytes": 0,
            "cast_bytes": 0,
            "block_cast_bytes": 0,
        }
        self._preload_half_weights()

    def _resolve(self, name: str) -> str:
        visited = set()
        while name in self.identity_map and name not in visited:
            visited.add(name)
            name = self.identity_map[name]
        return name

    def _initializer(self, name: str | None):
        if not name:
            return None
        return self.initializers.get(self._resolve(name))

    @staticmethod
    def _numpy_dtype(data_type: int) -> np.dtype:
        try:
            return np.dtype(onnx.helper.tensor_dtype_to_np_dtype(data_type))
        except AttributeError:
            return np.dtype(onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[data_type])

    def _array(self, name: str) -> np.ndarray:
        """Return an initializer as an array, mmap'ing external data."""
        actual = self._resolve(name)
        tensor = self.initializers.get(actual)
        if tensor is None:
            raise KeyError(f"missing initializer: {name} (resolved to {actual})")

        info = {entry.key: entry.value for entry in tensor.external_data}
        if info.get("location"):
            data_path = (self.base_dir / info["location"]).resolve()
            try:
                data_path.relative_to(self.base_dir)
            except ValueError as exc:
                raise ValueError(
                    f"unsafe external-data path for {actual}: {info['location']}"
                ) from exc

            dtype = self._numpy_dtype(tensor.data_type)
            shape = tuple(int(dim) for dim in tensor.dims)
            offset = int(info.get("offset", 0))
            expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            length = int(info.get("length", expected))
            file_size = data_path.stat().st_size
            if length < expected or offset < 0 or offset + expected > file_size:
                raise ValueError(f"invalid external tensor bounds for {actual}")
            return np.memmap(
                data_path, dtype=dtype, mode="r", offset=offset,
                shape=shape, order="C",
            )

        return np.asarray(numpy_helper.to_array(tensor, base_dir=str(self.base_dir)))

    def _upload(self, name: str | None, *, cache: bool = False):
        """Return one FP32 GPU weight, using resident FP16 storage if possible."""
        if not name:
            return None
        actual = self._resolve(name)
        if cache and actual in self.shared_gpu:
            return self.shared_gpu[actual]

        if actual in self.resident_half:
            source = self.resident_half[actual]
            gpu = source.float()
            self.stats["cast_bytes"] += int(gpu.numel() * gpu.element_size())
        else:
            array = self._array(actual)
            # External-data mmap arrays are read-only. PyTorch only reads them,
            # and the synchronous H2D copy finishes before the view is released.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                cpu = self.torch.from_numpy(array)
            gpu = cpu.to(device=self.device, non_blocking=False)
            self.stats["weight_bytes"] += int(gpu.numel() * gpu.element_size())
        if cache:
            self.shared_gpu[actual] = gpu
        return gpu

    def _required_weight_names(self) -> set[str]:
        names = set()
        for block in self.blocks:
            for key in self._BLOCK_KEYS:
                name = block[key]
                if key in {"qkv", "proj", "ff1", "ff2"}:
                    names.add(self._resolve(name))
                elif self._initializer(name) is not None:
                    names.add(self._resolve(name))
        for name in (
            "tok_emb.weight", "pos_emb", "head.bias",
            "ln_f.weight", "ln_f.bias",
        ):
            if self._initializer(name) is not None:
                names.add(self._resolve(name))
        if self.head_weight_name:
            names.add(self._resolve(self.head_weight_name))
        return names

    def _preload_sort_key(self, name: str):
        tensor = self.initializers[name]
        info = {entry.key: entry.value for entry in tensor.external_data}
        return (info.get("location", ""), int(info.get("offset", 0)), name)

    def _preload_half_weights(self):
        """Deduplicate and preload all required weights as FP16 GPU tensors."""
        names = self._required_weight_names()
        required = 0
        for name in names:
            tensor = self.initializers[name]
            required += int(np.prod(tensor.dims, dtype=np.int64)) * 2

        free_bytes, _ = self.torch.cuda.mem_get_info()
        # Preserve four GiB for one expanded FP32 block, activations, and logits.
        if required + 4 * _ONE_GIB > free_bytes:
            print(
                f"[bigformer] FP16 residency needs {required/_ONE_GIB:.2f}GiB; "
                "using one-pass CPU weight streaming",
                file=sys.stderr,
            )
            return

        start = time.perf_counter()
        transfer_stream = self.torch.cuda.Stream()
        pending = []
        with self.torch.inference_mode():
            for name in sorted(names, key=self._preload_sort_key):
                array = self._array(name)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    cpu = self.torch.from_numpy(array)
                try:
                    # Convert directly into a reusable-size pinned FP16 staging
                    # tensor. H2D then overlaps the next tensor's mmap read and
                    # CPU conversion, and transfers half as many bytes as FP32.
                    staging = self.torch.empty(
                        cpu.shape, dtype=self.torch.float16,
                        device="cpu", pin_memory=True)
                    staging.copy_(cpu)
                    gpu = self.torch.empty(
                        cpu.shape, dtype=self.torch.float16,
                        device=self.device)
                    event = self.torch.cuda.Event()
                    with self.torch.cuda.stream(transfer_stream):
                        gpu.copy_(staging, non_blocking=True)
                        event.record(transfer_stream)
                    pending.append((event, staging))
                    if len(pending) > 2:
                        old_event, _ = pending.pop(0)
                        old_event.synchronize()
                except RuntimeError:
                    # Some hosts cap page-locked memory. Keep a functional
                    # direct-copy fallback instead of failing initialization.
                    gpu = cpu.to(
                        device=self.device, dtype=self.torch.float16,
                        non_blocking=False)
                self.resident_half[name] = gpu
                self.stats["weight_bytes"] += int(gpu.numel() * gpu.element_size())
            transfer_stream.synchronize()
            pending.clear()
        self.stats["preload_seconds"] = time.perf_counter() - start
        print(
            f"[bigformer] resident FP16 weights: "
            f"{self.stats['weight_bytes']/_ONE_GIB:.2f}GiB in "
            f"{self.stats['preload_seconds']:.2f}s",
            file=sys.stderr,
        )

    def _storage_weight(self, name: str):
        """Return compact resident storage for embedding lookup when available."""
        actual = self._resolve(name)
        if actual in self.resident_half:
            return self.resident_half[actual]
        return self._upload(actual, cache=True)

    def _discover_blocks(self):
        block_weights: dict[int, dict[str, str]] = {}
        head_weight = None
        for node in self.model.graph.node:
            if node.op_type != "MatMul":
                continue
            parts = node.name.strip("/").split("/")
            if len(parts) >= 2 and parts[0].startswith("blocks."):
                try:
                    block_id = int(parts[0].split(".", 1)[1])
                except (IndexError, ValueError):
                    continue
                sublayer = parts[1]
                for input_name in node.input:
                    actual = self._resolve(input_name)
                    if actual in self.initializers:
                        block_weights.setdefault(block_id, {})[sublayer] = actual
                        break
            elif "head" in node.name.lower():
                for input_name in node.input:
                    actual = self._resolve(input_name)
                    if actual in self.initializers:
                        head_weight = actual
                        break

        blocks = []
        for block_id in sorted(block_weights):
            weights = block_weights[block_id]
            required = {"qkv", "proj", "ff1", "ff2"}
            if not required.issubset(weights):
                missing = sorted(required - set(weights))
                raise ValueError(f"BigFormer block {block_id} missing {missing}")
            blocks.append({
                "id": block_id,
                "qkv": weights["qkv"],
                "proj": weights["proj"],
                "ff1": weights["ff1"],
                "ff2": weights["ff2"],
                "qkv_b": f"blocks.{block_id}.qkv.bias",
                "proj_b": f"blocks.{block_id}.proj.bias",
                "ff1_b": f"blocks.{block_id}.ff1.bias",
                "ff2_b": f"blocks.{block_id}.ff2.bias",
                "ln1_w": f"blocks.{block_id}.ln1.weight",
                "ln1_b": f"blocks.{block_id}.ln1.bias",
                "ln2_w": f"blocks.{block_id}.ln2.weight",
                "ln2_b": f"blocks.{block_id}.ln2.bias",
            })
        return blocks, head_weight

    def _optional_name(self, logical_name: str) -> str | None:
        return logical_name if self._initializer(logical_name) is not None else None

    def _load_block(self, block: dict[str, Any]) -> dict[str, Any]:
        cast_before = self.stats["cast_bytes"]
        loaded = {}
        for key in self._BLOCK_KEYS:
            name = block[key]
            if key not in {"qkv", "proj", "ff1", "ff2"}:
                name = self._optional_name(name)
            loaded[key] = self._upload(name)
        self.stats["block_cast_bytes"] += self.stats["cast_bytes"] - cast_before
        return loaded

    def _prefetch_block(self, block: dict[str, Any], stream):
        """Expand the next resident block on a side stream."""
        with self.torch.cuda.stream(stream):
            loaded = self._load_block(block)
            ready = self.torch.cuda.Event()
            ready.record(stream)
        return loaded, ready

    @staticmethod
    def _record_weights_on_stream(weights, stream):
        for tensor in weights.values():
            if tensor is not None:
                tensor.record_stream(stream)

    def _linear(self, x, weight, bias=None):
        shape = x.shape[:-1] + (weight.shape[-1],)
        flat = x.reshape(-1, x.shape[-1])
        if bias is None:
            return (flat @ weight).reshape(shape)
        return self.torch.addmm(bias, flat, weight).reshape(shape)

    def _micro_batch_size(self, requested: int, sample_count: int,
                          sequence_length: int, ff_width: int) -> int:
        # Cap the largest FFN temporary near 1 GiB.  This preserves room for a
        # block's weights, attention temporaries, shared weights, and outputs.
        bytes_per_sample = max(1, sequence_length * ff_width * 4)
        memory_cap = max(1, _ONE_GIB // bytes_per_sample)
        # The contest H200 is exposed as a 16 GiB MIG slice. A 128-sample
        # micro-batch avoids allocator retries while preserving throughput.
        total_memory = self.torch.cuda.get_device_properties(0).total_memory
        if total_memory <= 20 * _ONE_GIB:
            memory_cap = min(memory_cap, 128)
        return max(1, min(requested, sample_count, memory_cap))

    def _sync_if_profiling(self):
        if self.profile:
            self.torch.cuda.synchronize()

    @contextmanager
    def _tf32_guard(self):
        old_value = self.torch.backends.cuda.matmul.allow_tf32
        self.torch.backends.cuda.matmul.allow_tf32 = False
        try:
            yield
        finally:
            self.torch.backends.cuda.matmul.allow_tf32 = old_value

    def run(self, input_tensors: dict[str, np.ndarray], batch_size: int = 2048):
        if len(self.output_names) != 1:
            raise ValueError("BigFormer streaming backend expects one graph output")
        if not input_tensors:
            raise ValueError("no BigFormer inputs supplied")

        input_ids_np = next(iter(input_tensors.values()))
        if input_ids_np.ndim == 1:
            input_ids_np = input_ids_np[:, None]
        if input_ids_np.ndim != 2:
            raise ValueError(f"expected BigFormer token IDs shaped [N,S], got {input_ids_np.shape}")

        torch = self.torch
        F = self.F
        self.stats["copy_seconds"] = 0.0
        self.stats["compute_seconds"] = 0.0
        self.stats["head_seconds"] = 0.0
        self.stats["cast_bytes"] = 0
        self.stats["block_cast_bytes"] = 0
        with self._tf32_guard(), torch.inference_mode():
            input_ids = torch.as_tensor(
                input_ids_np, dtype=torch.long, device=self.device)
            sample_count, sequence_length = input_ids.shape

            # Keep the large tables compact. Only selected rows/positions are
            # converted to FP32, avoiding another persistent full-size copy.
            tok_emb = self._storage_weight("tok_emb.weight")
            pos_emb = self._storage_weight("pos_emb")
            if pos_emb.ndim == 2:
                positions = pos_emb[:sequence_length].unsqueeze(0)
            elif pos_emb.ndim == 3:
                positions = pos_emb[:, :sequence_length]
            else:
                raise ValueError(f"unexpected positional embedding shape: {pos_emb.shape}")
            x = tok_emb[input_ids].float() + positions.float()
            hidden_size = int(x.shape[-1])
            if hidden_size % 128 != 0:
                raise ValueError(f"unsupported BigFormer hidden size: {hidden_size}")
            num_heads = hidden_size // 128
            head_dim = hidden_size // num_heads

            ff_shape = self._initializer(self.blocks[0]["ff1"]).dims
            micro_batch = self._micro_batch_size(
                batch_size, sample_count, sequence_length, int(ff_shape[-1]))
            # Fine-grained TF32 routing. Small batches can use every block; the
            # measured batch-256 default keeps the final four blocks strict FP32.
            # C3_TF32_BLOCKS remains available for repeatable tuning.
            if micro_batch <= 32:
                tf32_block_count = len(self.blocks)
            else:
                default_tf32_blocks = max(0, len(self.blocks) - 4)
                tf32_block_count = max(0, min(
                    len(self.blocks), int(os.environ.get(
                        "C3_TF32_BLOCKS", str(default_tf32_blocks)))))

            low_memory_gpu = (
                torch.cuda.get_device_properties(0).total_memory <= 20 * _ONE_GIB)
            prefetch_enabled = (
                bool(self.resident_half)
                and not self.profile
                and os.environ.get("C3_DISABLE_PREFETCH") != "1"
                and (not low_memory_gpu or os.environ.get("C3_FORCE_PREFETCH") == "1")
            )
            compute_stream = torch.cuda.current_stream()
            cast_stream = torch.cuda.Stream() if prefetch_enabled else None
            prefetched = None
            if prefetch_enabled:
                weights = self._load_block(self.blocks[0])
                if len(self.blocks) > 1:
                    prefetched = self._prefetch_block(self.blocks[1], cast_stream)

            for block_index, block in enumerate(self.blocks):
                use_tf32 = block_index < tf32_block_count
                if prefetch_enabled:
                    if block_index > 0:
                        weights, ready = prefetched
                        compute_stream.wait_event(ready)
                        self._record_weights_on_stream(weights, compute_stream)
                        prefetched = None
                        if block_index + 1 < len(self.blocks):
                            prefetched = self._prefetch_block(
                                self.blocks[block_index + 1], cast_stream)
                else:
                    self._sync_if_profiling()
                    copy_start = time.perf_counter()
                    weights = self._load_block(block)
                    self._sync_if_profiling()
                    if self.profile:
                        self.stats["copy_seconds"] += time.perf_counter() - copy_start

                compute_start = time.perf_counter()
                for start in range(0, sample_count, micro_batch):
                    end = min(start + micro_batch, sample_count)
                    chunk = x[start:end]
                    batch = end - start

                    residual = chunk
                    chunk = F.layer_norm(
                        chunk, [hidden_size], weights["ln1_w"],
                        weights["ln1_b"], 1e-5)
                    qkv = self._linear(chunk, weights["qkv"], weights["qkv_b"])
                    q, k, v = qkv.chunk(3, dim=-1)
                    q = q.view(batch, sequence_length, num_heads, head_dim).transpose(1, 2)
                    k = k.view(batch, sequence_length, num_heads, head_dim).transpose(1, 2)
                    v = v.view(batch, sequence_length, num_heads, head_dim).transpose(1, 2)
                    attention = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
                    attention = F.softmax(attention, dim=-1)
                    attention = attention @ v
                    attention = attention.transpose(1, 2).reshape(
                        batch, sequence_length, hidden_size)
                    chunk = residual + self._linear(
                        attention, weights["proj"], weights["proj_b"])

                    residual = chunk
                    chunk = F.layer_norm(
                        chunk, [hidden_size], weights["ln2_w"],
                        weights["ln2_b"], 1e-5)
                    if use_tf32:
                        torch.backends.cuda.matmul.allow_tf32 = True
                    chunk = self._linear(chunk, weights["ff1"], weights["ff1_b"])
                    chunk = F.gelu(chunk)
                    chunk = self._linear(chunk, weights["ff2"], weights["ff2_b"])
                    if use_tf32:
                        torch.backends.cuda.matmul.allow_tf32 = False
                    x[start:end].copy_(residual + chunk)

                self._sync_if_profiling()
                if self.profile:
                    self.stats["compute_seconds"] += time.perf_counter() - compute_start
                del weights

            final_ln_w = self._upload(self._optional_name("ln_f.weight"), cache=True)
            final_ln_b = self._upload(self._optional_name("ln_f.bias"), cache=True)
            head_w = self._upload(self.head_weight_name, cache=True)
            head_b = self._upload(self._optional_name("head.bias"), cache=True)

            self._sync_if_profiling()
            head_start = time.perf_counter()
            output_chunks = []
            for start in range(0, sample_count, micro_batch):
                end = min(start + micro_batch, sample_count)
                chunk = x[start:end]
                if final_ln_w is not None:
                    chunk = F.layer_norm(
                        chunk, [hidden_size], final_ln_w, final_ln_b, 1e-5)
                chunk = self._linear(chunk, head_w, head_b)
                output_chunks.append(chunk.cpu())
            self._sync_if_profiling()
            if self.profile:
                self.stats["head_seconds"] += time.perf_counter() - head_start
            logits = torch.cat(output_chunks, dim=0).numpy().astype(np.float32, copy=False)

        if self.profile:
            legacy_batches = (sample_count + micro_batch - 1) // micro_batch
            resident_gib = self.stats["weight_bytes"] / _ONE_GIB
            cast_gib = self.stats["cast_bytes"] / _ONE_GIB
            block_cast_gib = self.stats["block_cast_bytes"] / _ONE_GIB
            print(
                f"[bigformer-profile] blocks={len(self.blocks)} micro_batch={micro_batch} "
                f"resident={resident_gib:.2f}GiB preload={self.stats['preload_seconds']:.3f}s; "
                f"fp32-cast={cast_gib:.2f}GiB once "
                f"(batch-major blocks ~= {block_cast_gib * legacy_batches:.2f}GiB); "
                f"cast/copy={self.stats['copy_seconds']:.3f}s "
                f"blocks={self.stats['compute_seconds']:.3f}s "
                f"head={self.stats['head_seconds']:.3f}s",
                file=sys.stderr,
            )
        return {self.output_names[0]: logits}


def infer_bigformer(onnx_path: str, input_tensors: dict[str, np.ndarray],
                    batch_size: int, profile: bool = False):
    """One-shot entry point used by infer.py."""
    start = time.perf_counter()
    executor = BigFormerStreamingExecutor(onnx_path, profile=profile)
    outputs = executor.run(input_tensors, batch_size=batch_size)
    return outputs, time.perf_counter() - start
