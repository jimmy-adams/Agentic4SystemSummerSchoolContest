#!/usr/bin/env python3
"""C3.5 Persistent Worker — stdin/stdout JSON protocol.

Protocol (see C35_WORKER_PROTOCOL.md):
  1. Startup → import frameworks, init CUDA → output "READY\n" on stdout
  2. Loop: read JSON task from stdin → load model + infer + write output → output {"status":"ok","samples":N}\n
  3. On {"cmd":"exit"} → exit(0)

All logging goes to stderr. stdout is reserved for protocol messages only.
"""
import json
import os
import sys
import time
import traceback

# ── Init: one-time framework imports & CUDA context ────────────────────
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

import numpy as np
import onnx
import onnxruntime as ort

HAS_TORCH = False  # keep simple: ORT only


# ═══════════════════════════════════════════════════════════════════════════
# Core functions (reused from infer.py)
# ═══════════════════════════════════════════════════════════════════════════

def load_inputs(input_dir: str) -> dict:
    """Load input tensors from manifest.json + .npy files."""
    with open(os.path.join(input_dir, "manifest.json")) as f:
        manifest = json.load(f)
    tensors = {}
    for entry in manifest["tensors"]:
        tensors[entry["name"]] = np.load(os.path.join(input_dir, entry["file"]))
    return tensors


def write_outputs(output_dir: str, outputs: dict):
    """Write output tensors in manifest.json + .npy format."""
    os.makedirs(output_dir, exist_ok=True)
    manifest = {"tensors": []}
    for name, data in outputs.items():
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        npy_file = f"{name}.npy"
        np.save(os.path.join(output_dir, npy_file), data)
        manifest["tensors"].append({
            "name": name, "file": npy_file,
            "dtype": "float32", "shape": list(data.shape),
        })
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def get_output_names(onnx_path: str) -> list:
    """Get output tensor names from ONNX model."""
    model = onnx.load(onnx_path)
    return [o.name for o in model.graph.output]


def detect_model_type(onnx_path: str) -> str:
    """Read ONNX graph operator types and classify model."""
    model = onnx.load(onnx_path)
    op_types = {node.op_type for node in model.graph.node}
    if "MatMul" in op_types and "Softmax" in op_types:
        return "transformer"
    if "Conv" in op_types and "Gemm" in op_types:
        return "resnet"
    if op_types.issubset({"Flatten", "Gemm", "Relu", "Reshape"}):
        return "mlp"
    return "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# Inference backends
# ═══════════════════════════════════════════════════════════════════════════

def infer_ort(onnx_path: str, input_tensors: dict, output_names: list,
              batch_size: int, session_cache: dict = None) -> dict:
    """ONNX Runtime GPU inference. session_cache allows reuse across warmup rounds.

    For large models (BigFormer ~19GB weights > 16GB GPU), uses CUDA with
    arena disabled + CPU fallback to enable memory offloading.
    """
    cache_key = onnx_path
    if session_cache is not None and cache_key in session_cache:
        session = session_cache[cache_key]
    else:
        providers = []
        cuda_opts = {"device_id": "0"}
        model_size_hint = os.path.getsize(onnx_path)
        if os.path.exists(onnx_path + ".data"):
            model_size_hint += os.path.getsize(onnx_path + ".data")
        is_large = model_size_hint > 10 * 1024**3

        if is_large:
            # BigFormer ~19GB > 16GB GPU → CPU-only execution
            print(f"[worker] BigFormer ({model_size_hint/1e9:.1f}GB): CPU-only", file=sys.stderr)
            providers = [("CPUExecutionProvider", {})]
        else:
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.append(("CUDAExecutionProvider", cuda_opts))
            providers.append(("CPUExecutionProvider", {}))

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if is_large:
            sess_options.inter_op_num_threads = 8
            sess_options.intra_op_num_threads = 8
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        else:
            sess_options.intra_op_num_threads = 8

        session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
        if session_cache is not None:
            session_cache[cache_key] = session

    input_metas = session.get_inputs()
    input_names = [m.name for m in input_metas]
    onnx_to_np = {}
    for onnx_name in input_names:
        if onnx_name in input_tensors:
            onnx_to_np[onnx_name] = input_tensors[onnx_name]
        else:
            for k, v in input_tensors.items():
                if k.lower() == onnx_name.lower():
                    onnx_to_np[onnx_name] = v
                    break

    N = list(onnx_to_np.values())[0].shape[0]
    B = min(batch_size, N)

    # Warmup within this task
    if "CUDA" in str(session.get_providers()):
        warm_feed = {n: d[:min(8, N)] for n, d in onnx_to_np.items()}
        _ = session.run(output_names, warm_feed)

    all_outputs = {name: [] for name in output_names}
    for start in range(0, N, B):
        end = min(start + B, N)
        feed = {n: onnx_to_np[n][start:end] for n in onnx_to_np}
        results = session.run(output_names, feed)
        for name, arr in zip(output_names, results):
            all_outputs[name].append(arr)

    return {n: np.concatenate(chunks, axis=0) for n, chunks in all_outputs.items()}


# ═══════════════════════════════════════════════════════════════════════════
# Worker main loop
# ═══════════════════════════════════════════════════════════════════════════

def run_worker():
    """Main worker loop: READY → task loop → exit."""
    
    # ── Phase 1: Signal readiness ──────────────────────────────────────
    print("READY", flush=True)
    print("[worker] Framework initialized, CUDA ready", file=sys.stderr)

    session_cache = {}  # reuse ORT sessions across repeated tasks for same model

    # ── Phase 2: Task loop ─────────────────────────────────────────────
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            task = json.loads(line)
        except json.JSONDecodeError:
            print(f"[worker] Invalid JSON: {line}", file=sys.stderr)
            continue

        # Exit command
        if task.get("cmd") == "exit":
            print("[worker] Received exit", file=sys.stderr)
            break

        # Process task
        onnx_path = task["onnx"]
        input_dir = task["input"]
        output_dir = task["output"]
        batch_size = task.get("batch_size", 2048)

        model_type = detect_model_type(onnx_path)
        print(f"[worker] Task: {model_type} bs={batch_size}", file=sys.stderr)

        try:
            t0 = time.perf_counter()
            input_tensors = load_inputs(input_dir)
            output_names = get_output_names(onnx_path)
            N = list(input_tensors.values())[0].shape[0]

            outputs = infer_ort(onnx_path, input_tensors, output_names, batch_size, session_cache)

            write_outputs(output_dir, outputs)
            dt = time.perf_counter() - t0

            result = {"status": "ok", "samples": N}
            print(json.dumps(result), flush=True)
            print(f"[worker] Done: {N} samples in {dt:.2f}s", file=sys.stderr)

        except Exception as e:
            print(f"[worker] Error: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            result = {"status": "error", "error": str(e)}
            print(json.dumps(result), flush=True)

    # ── Phase 3: Clean exit ────────────────────────────────────────────
    print("[worker] Exiting", file=sys.stderr)
    sys.exit(0)


if __name__ == "__main__":
    run_worker()
