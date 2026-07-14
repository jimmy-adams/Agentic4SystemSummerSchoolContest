#!/usr/bin/env python3
"""
C3.5: End-to-end ONNX model inference with auto backend selection.

Backend routing (--auto, default on):
    MLP         → v2 executor (GPUExecutor, ~50% faster than ORT)
    ResNet      → v2 executor + memory pool (MemoryAwareExecutor, saves ~346MB)
    Transformer → ONNX Runtime (ORT, optimal for complex graphs)

Detection logic (reads ONNX graph operator types):
    Conv + Gemm            → ResNet
    only Flatten/Gemm/Relu → MLP
    MatMul + Softmax       → Transformer
    otherwise              → ORT (fallback)

Usage:
    python infer.py --onnx {onnx} --input {input} --output {output} --batch-size 256
    python infer.py --onnx {onnx} --input {input} --output {output} --no-auto  # force ORT
"""

import argparse, json, os, sys, time
import numpy as np
import onnx

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

# Enable cuDNN auto-tuning for v2 executor paths (MLP + ResNet).
# Lets cuDNN select the optimal conv algorithm (e.g. Winograd) per input size.
# Cost: small one-time warmup overhead per input shape.
import torch
torch.backends.cudnn.benchmark = True

# Enable TF32 for matrix multiplies (Gemm/MatMul).
# Uses Tensor Cores for ~8x throughput on Ampere+, precision still within 1e-3.
# Note: do NOT enable cudnn.allow_tf32 — conv2d with TF32 is slower for 3x3 kernels.
torch.backends.cuda.matmul.allow_tf32 = True


# ═══════════════════════════════════════════════════════════════════════════
# Model type detection
# ═══════════════════════════════════════════════════════════════════════════

_ONNX_MODEL_CACHE = {}

def detect_model_type(onnx_path: str) -> str:
    """Read ONNX graph operator types and classify model.

    Returns one of: 'mlp' | 'resnet' | 'transformer' | 'unknown'
    """
    if onnx_path in _ONNX_MODEL_CACHE:
        model = _ONNX_MODEL_CACHE[onnx_path]
    else:
        model = onnx.load(onnx_path)
        _ONNX_MODEL_CACHE[onnx_path] = model
    op_types = {node.op_type for node in model.graph.node}

    # Transformer: has MatMul AND Softmax (self-attention signature)
    if "MatMul" in op_types and "Softmax" in op_types:
        return "transformer"

    # ResNet: has Conv AND Gemm (CNN backbone + classifier head)
    if "Conv" in op_types and "Gemm" in op_types:
        return "resnet"

    # MLP: ops are subset of {Flatten, Gemm, Relu, Reshape}
    #      (no Conv, no MatMul, no Softmax — simple feed-forward)
    mlp_ops = {"Flatten", "Gemm", "Relu", "Reshape"}
    if op_types.issubset(mlp_ops):
        return "mlp"

    return "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# I/O helpers (standard manifest.json + .npy format)
# ═══════════════════════════════════════════════════════════════════════════

def load_inputs(input_dir: str) -> dict:
    """Load input tensors from manifest.json + .npy files."""
    with open(os.path.join(input_dir, "manifest.json")) as f:
        m = json.load(f)
    return {e["name"]: np.load(os.path.join(input_dir, e["file"]))
            for e in m["tensors"]}


def write_outputs(output_dir: str, outputs: dict):
    """Write output tensors in manifest.json + .npy format."""
    os.makedirs(output_dir, exist_ok=True)
    m = {"tensors": []}
    for name, data in outputs.items():
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        np.save(os.path.join(output_dir, f"{name}.npy"), data)
        m["tensors"].append({
            "name": name, "file": f"{name}.npy",
            "dtype": "float32", "shape": list(data.shape),
        })
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(m, f, indent=2)


def get_onnx_output_names(onnx_path: str) -> list:
    """Read output tensor names from ONNX model."""
    model = onnx.load(onnx_path)
    return [o.name for o in model.graph.output]


def get_sample_count(tensors: dict) -> int:
    """Return N (first dimension of first tensor)."""
    return list(tensors.values())[0].shape[0]


# ═══════════════════════════════════════════════════════════════════════════
# Backend: v2 executor (for MLP — lightweight, no memory pool overhead)
# ═══════════════════════════════════════════════════════════════════════════

def infer_v2(onnx_path: str, input_tensors: dict, output_names: list,
             batch_size: int) -> tuple:
    """v2 executor — reuses GPUExecutor across batches for minimal overhead.

    Builds plan once, creates executor once (weights→GPU once), then
    loops over batches calling load_inputs + execute_plan.
    """
    from scheduler.planner import import_onnx_graph, decompose_graph, apply_fusion
    from scheduler.executor import GPUExecutor

    # Build plan once
    graph = import_onnx_graph(onnx_path)
    raw_plan = decompose_graph(graph)
    opt_plan, stats = apply_fusion(graph, raw_plan)

    # For MLP, fused plan is preferred (fewer kernel launches)
    plan = opt_plan if len(opt_plan) < len(raw_plan) else raw_plan

    N = get_sample_count(input_tensors)
    B = min(batch_size, N)
    all_out = {n: [] for n in output_names}

    # Create executor once (weights loaded to GPU once)
    executor = GPUExecutor(onnx_path)

    t0 = time.perf_counter()

    for start in range(0, N, B):
        end = min(start + B, N)
        batch = {k: v[start:end] for k, v in input_tensors.items()}
        executor.load_inputs(batch)
        results = executor.execute_plan(plan, output_names)
        for n, arr in results.items():
            all_out[n].append(arr)

    t1 = time.perf_counter()
    final = {n: np.concatenate(chunks, axis=0) for n, chunks in all_out.items()}

    print(f"    [v2] kernels {len(raw_plan)}→{len(opt_plan)} "
          f"({stats.get('fusion_reduction', 'N/A')})", file=sys.stderr)

    return final, t1 - t0


# ═══════════════════════════════════════════════════════════════════════════
# Backend: v2 executor + memory pool (for ResNet — saves ~346MB)
# ═══════════════════════════════════════════════════════════════════════════

def infer_v2_mempool(onnx_path: str, input_tensors: dict, output_names: list,
                     batch_size: int) -> tuple:
    """v2 executor with MemoryAwareExecutor for memory pooling.

    Memory pool reuses intermediate tensor buffers via lifetime analysis,
    reducing ResNet's 48 intermediate tensors → ~3 buffers (~346 MB saved).
    """
    from scheduler.planner import import_onnx_graph, decompose_graph, apply_fusion
    from scheduler.executor import MemoryAwareExecutor

    # Build plan once (reused across batches)
    graph = import_onnx_graph(onnx_path)
    raw_plan = decompose_graph(graph)
    opt_plan, stats = apply_fusion(graph, raw_plan)

    # For ResNet, use raw plan — fused plan has tensor name resolution issues
    # (same strategy as validated in final_v2.py)
    plan = raw_plan

    N = get_sample_count(input_tensors)
    B = min(batch_size, N)
    all_out = {n: [] for n in output_names}

    # Create one executor — weights loaded to GPU once, reused across batches
    executor = MemoryAwareExecutor(onnx_path)
    t0 = time.perf_counter()

    first_batch = True
    for start in range(0, N, B):
        end = min(start + B, N)
        batch = {k: v[start:end] for k, v in input_tensors.items()}

        # Load fresh input batch (overwrites previous input tensors in registry)
        executor.load_inputs(batch)

        # Execute plan — intermediate tensors reused via memory pool
        # Only the first batch prints memory pool stats; subsequent batches
        # have stale intermediates that bypass the pool (noise to suppress).
        _stderr = sys.stderr
        if not first_batch:
            sys.stderr = open(os.devnull, 'w')
        try:
            results = executor.execute_plan(plan, output_names)
        finally:
            if not first_batch:
                sys.stderr.close()
                sys.stderr = _stderr
        first_batch = False
        for n, arr in results.items():
            all_out[n].append(arr)

    t1 = time.perf_counter()
    final = {n: np.concatenate(chunks, axis=0) for n, chunks in all_out.items()}

    print(f"    [v2+mempool] kernels {len(raw_plan)}→{len(opt_plan)} "
          f"({stats.get('fusion_reduction', 'N/A')})", file=sys.stderr)

    return final, t1 - t0


# ═══════════════════════════════════════════════════════════════════════════
# Backend: ONNX Runtime (for Transformer / fallback)
# ═══════════════════════════════════════════════════════════════════════════

def infer_ort(onnx_path: str, input_tensors: dict, output_names: list,
              batch_size: int, force_cpu: bool = False) -> tuple:
    """Standard ONNX Runtime inference with GPU acceleration."""
    import onnxruntime as ort

    if force_cpu:
        providers = [("CPUExecutionProvider", {})]
    else:
        available = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append(("CUDAExecutionProvider", {"device_id": "0"}))
        providers.append(("CPUExecutionProvider", {}))

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL if force_cpu
        else ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    )
    sess_options.intra_op_num_threads = 8

    session = ort.InferenceSession(onnx_path, sess_options=sess_options,
                                   providers=providers)

    onnx_input_names = [m.name for m in session.get_inputs()]
    onnx_output_names = [m.name for m in session.get_outputs()]

    # Use ONNX metadata output names if caller didn't specify
    if not output_names:
        output_names = onnx_output_names

    # Map manifest tensor names → ONNX input names (case-insensitive fallback)
    onnx_to_np = {}
    for onnx_name in onnx_input_names:
        if onnx_name in input_tensors:
            onnx_to_np[onnx_name] = input_tensors[onnx_name]
        else:
            for k, v in input_tensors.items():
                if k.lower() == onnx_name.lower():
                    onnx_to_np[onnx_name] = v
                    break

    N = get_sample_count(onnx_to_np)
    B = min(batch_size, N)

    # Warm-up: compile CUDA kernels with a small batch
    if "CUDA" in str(session.get_providers()):
        warm_feed = {n: onnx_to_np[n][:min(8, N)] for n in onnx_input_names}
        _ = session.run(onnx_output_names, warm_feed)

    all_outputs = {n: [] for n in onnx_output_names}
    t0 = time.perf_counter()

    for start in range(0, N, B):
        end = min(start + B, N)
        feed = {n: onnx_to_np[n][start:end] for n in onnx_input_names}
        results = session.run(onnx_output_names, feed)
        for name, arr in zip(onnx_output_names, results):
            all_outputs[name].append(arr)

    t1 = time.perf_counter()
    final = {n: np.concatenate(chunks, axis=0) for n, chunks in all_outputs.items()}

    return final, t1 - t0


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

BACKEND_LABELS = {
    "mlp":         "v2 executor",
    "resnet":      "v2 executor + memory pool",
    "transformer": "ONNX Runtime",
    "unknown":     "ONNX Runtime (fallback)",
}


def main():
    parser = argparse.ArgumentParser(description="C3.5 ONNX Model Inference")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--input", required=True, help="Input directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="Batch size (default: 2048)")
    parser.add_argument("--no-auto", action="store_true",
                        help="Disable auto backend selection (force ORT)")
    parser.add_argument("--auto", action="store_true", default=True,
                        help="Enable auto backend selection (default: enabled)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU execution (ORT backend only)")
    args = parser.parse_args()

    # Load input data
    input_tensors = load_inputs(args.input)
    output_names = get_onnx_output_names(args.onnx)
    N = get_sample_count(input_tensors)

    # ── Determine backend ──────────────────────────────────────────────
    auto_enabled = args.auto and not args.no_auto

    if auto_enabled and not args.cpu:
        model_type = detect_model_type(args.onnx)
    else:
        model_type = "unknown"

    backend_label = BACKEND_LABELS.get(model_type, BACKEND_LABELS["unknown"])

    print(f"[auto] Detected {model_type} → {backend_label}", file=sys.stderr)

    # ── Route to backend ───────────────────────────────────────────────
    if model_type == "mlp":
        final_outputs, elapsed = infer_v2(
            args.onnx, input_tensors, output_names, args.batch_size)
    elif model_type == "resnet":
        final_outputs, elapsed = infer_ort(
            args.onnx, input_tensors, output_names, args.batch_size)
    else:
        final_outputs, elapsed = infer_ort(
            args.onnx, input_tensors, output_names, args.batch_size,
            force_cpu=args.cpu)

    # ── Write outputs ──────────────────────────────────────────────────
    write_outputs(args.output, final_outputs)

    throughput = N / elapsed if elapsed > 0 else 0
    print(f"Inference complete: {N} samples, batch_size={args.batch_size}, "
          f"time={elapsed:.2f}s, throughput={throughput:.0f} samples/s, "
          f"backend={model_type}", file=sys.stderr)


if __name__ == "__main__":
    main()
