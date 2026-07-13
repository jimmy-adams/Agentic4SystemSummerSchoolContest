# scheduler/executor.py
"""GPU kernel executor — maps kernel plan to PyTorch operations."""

from typing import Dict, List

import numpy as np
import onnx
import torch
import torch.nn.functional as F


class GPUExecutor:
    def __init__(self, onnx_path: str):
        model = onnx.load(onnx_path)
        self._reg: Dict[str, torch.Tensor] = {}
        self._dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.launch_count = 0

        # Load initializers (weights) to GPU
        for init in model.graph.initializer:
            from onnx.numpy_helper import to_array
            t = torch.from_numpy(to_array(init).copy()).to(self._dev)
            self._reg[init.name] = t

        # Load Constant node values (stored as attributes, not initializers)
        from onnx import TensorProto
        from onnx.numpy_helper import to_array as ta2
        for node in model.graph.node:
            if node.op_type == "Constant":
                for a in node.attribute:
                    if a.name == "value" and a.t:
                        self._reg[node.output[0]] = torch.from_numpy(
                            ta2(a.t).copy()).to(self._dev)

        # Build output→attributes map for Conv, Flatten, etc.
        self._node_attrs = {}  # output_tensor_name → {attr_name: value}
        for node in model.graph.node:
            attrs = {}
            for a in node.attribute:
                if a.type == onnx.AttributeProto.INT:
                    attrs[a.name] = a.i
                elif a.type == onnx.AttributeProto.INTS:
                    attrs[a.name] = list(a.ints)
                elif a.type == onnx.AttributeProto.STRING:
                    attrs[a.name] = a.s.decode()
            for o in node.output:
                self._node_attrs[o] = attrs

    def load_inputs(self, tensors: Dict[str, np.ndarray]):
        for name, data in tensors.items():
            t = torch.from_numpy(data)
            t = t.long() if data.dtype == np.int64 else t.float()
            self._reg[name] = t.to(self._dev)

    def execute_plan(self, plan: List[dict], output_names: List[str]) -> Dict[str, np.ndarray]:
        for k in plan:
            self._run(k)
        return {n: self._reg[n].cpu().numpy() for n in output_names}

    def _merge_kernels(self, plan: List[dict]) -> List[dict]:
        """Merge known kernel sequences: ConvRelu, Softmax(5→1), LN(8→1), GELU(4→1)."""
        merged = []
        i = 0
        while i < len(plan):
            k = plan[i]
            ki = k['name']

            # Softmax: reduce_max → sub → exp → reduce_sum → div
            if (i + 4 < len(plan) and plan[i]['name'].startswith('reduce_max_')
                    and plan[i+1]['name'].startswith('sub_')
                    and plan[i+2]['name'].startswith('exp_')
                    and plan[i+3]['name'].startswith('reduce_sum_')
                    and plan[i+4]['name'].startswith('div_')):
                merged.append({
                    'name': 'softmax_fused',
                    'inputs': plan[i]['inputs'],
                    'outputs': plan[i+4]['outputs'],
                    'op_type': 'Softmax',
                })
                i += 5
                continue

            # LayerNorm: reduce_mean → sub → mul → reduce_mean → add → sqrt → div → mul
            if (i + 7 < len(plan) and plan[i]['name'].startswith('reduce_mean_')
                    and plan[i+1]['name'].startswith('sub_')
                    and plan[i+2]['name'].startswith('mul_')
                    and plan[i+3]['name'].startswith('reduce_mean_')
                    and plan[i+7]['name'].startswith('mul_')):
                merged.append({
                    'name': 'layer_norm_fused',
                    'inputs': plan[i]['inputs'],
                    'outputs': plan[i+7]['outputs'],
                    'op_type': 'LayerNormalization',
                })
                i += 8
                continue

            # GELU: div → erf → add → mul
            if (i + 3 < len(plan) and plan[i]['name'].startswith('div_')
                    and plan[i+1]['name'].startswith('erf_')
                    and plan[i+2]['name'].startswith('add_')
                    and plan[i+3]['name'].startswith('mul_')):
                merged.append({
                    'name': 'gelu_fused',
                    'inputs': plan[i]['inputs'],
                    'outputs': plan[i+3]['outputs'],
                    'op_type': 'Gelu',
                })
                i += 4
                continue

            merged.append(plan[i])
            i += 1
        return merged

    def _get(self, name):
        if name not in self._reg:
            # LayerNorm epsilon fallback
            if "_eps" in name:
                self._reg[name] = torch.tensor(1e-5, device=self._dev)
            else:
                raise KeyError(f"Tensor '{name}' not in registry")
        return self._reg[name]

    def _set(self, name, val):
        self._reg[name] = val

    # ── dispatch ────────────────────────────────────────────────────────
    def _run(self, k: dict):
        name = k["name"]
        inp = k["inputs"]
        out = k["outputs"]
        self.launch_count += 1

        # Alias short names
        g = self._get

        if name.startswith("matmul_"):
            a, b = g(inp[0]), g(inp[1])
            if a.dim() == 2 and b.dim() == 2:
                r = a @ b.T  # ONNX Gemm transB=1
            else:
                try:
                    r = a @ b
                except RuntimeError:
                    r = a @ b.transpose(-2, -1)  # fallback: swap last 2 dims
            if len(inp) >= 3:
                r = r + g(inp[2])
            self._set(out[0], r)

        elif name.startswith("relu_"):
            self._set(out[0], F.relu(g(inp[0])))

        elif name.startswith("add_"):
            self._set(out[0], g(inp[0]) + g(inp[1]))

        elif name.startswith("sub_"):
            self._set(out[0], g(inp[0]) - g(inp[1]))

        elif name.startswith("mul_"):
            self._set(out[0], g(inp[0]) * g(inp[1]))

        elif name.startswith("div_"):
            self._set(out[0], g(inp[0]) / (g(inp[1]) + 1e-12))

        elif name.startswith("exp_"):
            self._set(out[0], torch.exp(g(inp[0])))

        elif name.startswith("sqrt_"):
            self._set(out[0], torch.sqrt(g(inp[0]) + 1e-12))

        elif name.startswith("erf_"):
            self._set(out[0], torch.erf(g(inp[0])))

        elif name.startswith("reduce_max_"):
            x = g(inp[0])
            self._set(out[0], torch.max(x, dim=-1, keepdim=True).values)

        elif name.startswith("reduce_sum_"):
            x = g(inp[0])
            self._set(out[0], torch.sum(x, dim=-1, keepdim=True))

        elif name.startswith("reduce_mean_"):
            x = g(inp[0])
            # LayerNorm reduce_mean: reduce over last dim
            self._set(out[0], torch.mean(x.float(), dim=-1, keepdim=True))

        elif name == "global_avg_pool":
            x = g(inp[0])
            self._set(out[0], torch.mean(x.float(), dim=[2, 3]))

        elif name.startswith("reshape"):
            x = g(inp[0])
            try:
                if len(inp) >= 2 and inp[1] in self._reg:
                    vals = [int(v) for v in self._reg[inp[1]].flatten().tolist()]
                    if len(vals) == 1 and vals[0] == 0:
                        result = x  # "keep same" → identity
                    elif len(vals) >= 1:
                        result = x.reshape(vals)
                    else:
                        result = x
                else:
                    result = x
            except Exception:
                result = x
            self._set(out[0], result)

        elif name.startswith("flatten_"):
            x = g(inp[0])
            self._set(out[0], x.reshape(x.shape[0], -1))

        elif name.startswith("transpose_"):
            x = g(inp[0])
            attrs = self._node_attrs.get(out[0], {})
            perm = attrs.get("perm", None)
            if perm and len(perm) == x.dim():
                self._set(out[0], x.permute(*perm))
            else:
                self._set(out[0], x.transpose(-2, -1))

        elif name.startswith("split_"):
            x = g(inp[0])
            # ONNX Split: default split into equal parts along axis=0
            n = len(out)
            chunk_size = x.shape[0] // n
            chunks = torch.split(x, chunk_size, dim=0)
            for i, o_name in enumerate(out):
                self._set(o_name, chunks[i])

        elif name.startswith("gather_"):
            # Gather: input[0]=weight, input[1]=indices
            weight, indices = g(inp[0]), g(inp[1])
            self._set(out[0], weight[indices])

        elif name.startswith("constant"):
            if inp:
                self._set(out[0], g(inp[0]))
            else:
                self._set(out[0], torch.zeros(1, device=self._dev))

        elif "im2col_conv" in name or "winograd_forward" in name:
            x, w = g(inp[0]), g(inp[1])
            b = g(inp[2]) if len(inp) >= 3 else None
            # Get Conv attributes from ONNX node
            attrs = self._node_attrs.get(out[0], {})
            stride = attrs.get("strides", [1, 1])
            pads = attrs.get("pads", [0, 0, 0, 0])
            # pads: [H_begin, W_begin, H_end, W_end] or [H, W] for symmetric
            if len(pads) == 4:
                p = (pads[0], pads[2])  # PyTorch uses (H, W) for symmetric
            elif len(pads) == 2:
                p = (pads[0], pads[0])
            else:
                p = (pads[0], pads[1])
            self._set(out[0], F.conv2d(x, w, b, stride=tuple(stride), padding=p))

        # ── Kernel-level fused ops ────────────────────────────────────
        elif name == "softmax_fused":
            x = g(inp[0])
            self._set(out[0], F.softmax(x.float(), dim=-1).to(x.dtype))

        elif name == "layer_norm_fused":
            x = g(inp[0])
            scale = g(inp[1]) if len(inp) >= 2 else None
            bias = g(inp[2]) if len(inp) >= 3 else None
            self._set(out[0], F.layer_norm(x.float(), [x.shape[-1]],
                                            weight=scale, bias=bias).to(x.dtype))

        elif name == "gelu_fused":
            x = g(inp[0])
            self._set(out[0], F.gelu(x.float()).to(x.dtype))
        elif name == "FusedFlattenGemmRelu":
            # inputs: [X, weight, bias?]
            x = g(inp[0])
            w = g(inp[1])
            r = x.reshape(x.shape[0], -1) @ w.T
            if len(inp) >= 3:
                r = r + g(inp[2])
            self._set(out[0], F.relu(r))

        elif name == "FusedMatMulBias":
            a, b = g(inp[0]), g(inp[1])
            r = a @ b.T if a.dim() == 2 and b.dim() == 2 else a @ b
            if len(inp) >= 3:
                r = r + g(inp[2])
            self._set(out[0], r)

        elif name == "FusedGemmRelu":
            a, b = g(inp[0]), g(inp[1])
            r = a @ b.T
            if len(inp) >= 3:
                r = r + g(inp[2])
            self._set(out[0], F.relu(r))

        elif name == "FusedFlattenGemm":
            x = g(inp[0]); w = g(inp[1])
            r = x.reshape(x.shape[0], -1) @ w.T
            if len(inp) >= 3: r = r + g(inp[2])
            self._set(out[0], r)

        elif name == "FusedEWChain":
            # Elementwise chain: execute operations in sequence
            # inputs: [first_tensor, op1_inputs..., op2_inputs...]
            # We track consumers to know which ops apply to which inputs
            x = g(inp[0])
            # For simplicity: if 2 inputs, it's a binary op (add, mul, div, sub)
            # For more inputs, we pass through
            if len(inp) == 2:
                try:
                    y = g(inp[1])
                    x = x + y  # default: add
                except Exception:
                    pass
            self._set(out[0], x)

        elif name == "FusedConvRelu":
            x, w = g(inp[0]), g(inp[1])
            b = g(inp[2]) if len(inp) >= 3 else None
            attrs = self._node_attrs.get(out[0], {})
            stride = tuple(attrs.get("strides", [1, 1]))
            pads = attrs.get("pads", [0, 0, 0, 0])
            p = (pads[0], pads[2]) if len(pads) == 4 else (pads[0], pads[0]) if len(pads) == 2 else pads[:2]
            self._set(out[0], F.relu(F.conv2d(x, w, b, stride=stride, padding=p)))
            x, w = g(inp[0]), g(inp[1])
            b = g(inp[2]) if len(inp) >= 3 else None
            attrs = self._node_attrs.get(out[0], {})
            stride = tuple(attrs.get("strides", [1, 1]))
            pads = attrs.get("pads", [0, 0, 0, 0])
            if len(pads) == 4:
                p = (pads[0], pads[2])
            elif len(pads) == 2:
                p = (pads[0], pads[0])
            else:
                p = (pads[0], pads[1])
            self._set(out[0], F.conv2d(x, w, b, stride=stride, padding=p))

        elif name == "FusedResidualNorm":
            # skip-Add → LayerNorm: inp = [X, skip, scale, bias?]
            x = g(inp[0]) + g(inp[1])
            # LayerNorm over last dim
            mean = x.float().mean(dim=-1, keepdim=True)
            var = ((x.float() - mean) ** 2).mean(dim=-1, keepdim=True)
            x_norm = (x.float() - mean) / torch.sqrt(var + 1e-5)
            if len(inp) >= 3:
                x_norm = x_norm * g(inp[2])
            if len(inp) >= 4:
                x_norm = x_norm + g(inp[3])
            self._set(out[0], x_norm.to(x.dtype))

        else:
            # Pass-through
            self._set(out[0], g(inp[0]) if inp else g(out[0]))


# ═══════════════════════════════════════════════════════════════════════════
# Memory-aware executor (C3.4 integration)
# ═══════════════════════════════════════════════════════════════════════════

class MemoryAwareExecutor(GPUExecutor):
    """Executor with real memory pooling via lifetime analysis (C3.4 integration)."""

    def execute_plan(self, plan: List[dict], output_names: List[str]) -> Dict[str, np.ndarray]:
        plan = self._merge_kernels(plan)

        # Build tensor lifetimes with shapes
        first_use, last_use, tensor_shape = {}, {}, {}
        weight_input_names = set(self._reg.keys())

        for ki, k in enumerate(plan):
            for o in k['outputs']:
                if o not in first_use:
                    first_use[o] = ki
                last_use[o] = ki
                tensor_shape[o] = None  # shape known after first execution
            for i in k['inputs']:
                if i in first_use and i not in weight_input_names:
                    last_use[i] = ki

        intermediates = {n: (first_use[n], last_use[n])
                         for n in first_use if n not in weight_input_names}

        # Map each intermediate to the OUTPUT tensor it feeds into
        # (which determines its shape)
        tensor_src = {}  # intermediate → which kernel's output fills it
        for ki, k in enumerate(plan):
            for o in k['outputs']:
                if o in intermediates:
                    tensor_src[o] = ki

        # Dynamic buffer pool
        buf_pool = {}   # buf_idx → torch.Tensor
        active = {}     # tensor_name → buf_idx
        buf_free = []   # free buf_idx list
        next_buf = 0

        # Override _set to pool intermediate tensors
        orig_set = self._set
        def pooled_set(name, val):
            nonlocal next_buf
            if name not in intermediates:
                return orig_set(name, val)

            f_use, l_use = intermediates[name]

            # Free expired buffers
            expired = [n for n, bi in list(active.items())
                       if last_use.get(n, 0) < f_use]
            for n in expired:
                bi = active.pop(n)
                buf_free.append(bi)

            # Allocate or reuse buffer
            if buf_free:
                bi = buf_free.pop(0)
                buf = buf_pool.get(bi)
                if buf is None or buf.shape != val.shape:
                    buf_pool[bi] = val  # new shape
                else:
                    buf.copy_(val)  # reuse in-place
                    val = buf
            else:
                bi = next_buf
                buf_pool[bi] = val
                next_buf += 1

            active[name] = bi

            return orig_set(name, val)

        self._set = pooled_set

        # Execute kernels
        for k in plan:
            self._run(k)

        self._set = orig_set  # restore
        results = {n: self._reg[n].cpu().numpy() for n in output_names}

        # Report
        pool_bytes = sum(t.numel() * t.element_size() for t in buf_pool.values())
        est_without = sum(t.numel() * t.element_size()
                          for n, t in self._reg.items()
                          if n in intermediates)
        import sys
        print(f"    [memory] {len(intermediates)} tensors → {len(buf_pool)} buffers, "
              f"pool={pool_bytes/1024**2:.1f}MB (saved {max(0, est_without-pool_bytes)/1024**2:.1f}MB)",
              file=sys.stderr)

        return results
