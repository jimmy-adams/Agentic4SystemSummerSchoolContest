#!/usr/bin/env python3
"""
Auto backend benchmark: accuracy + performance across all 3 models.

Tests detection correctness, inference accuracy (1e-3 rtol), and speed vs ORT.
"""

import sys, os, time, json, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

import onnx, onnxruntime as ort
from scheduler.planner import import_onnx_graph, decompose_graph, apply_fusion
from scheduler.executor import GPUExecutor, MemoryAwareExecutor

# ── Paths ────────────────────────────────────────────────────────────────
MODEL_BASE = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models'
DATA_BASE  = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35'
MODELS = ['mlp_v1', 'resnet_v1', 'transformer_v1']


def detect_model_type(onnx_path: str) -> str:
    model = onnx.load(onnx_path)
    op_types = {node.op_type for node in model.graph.node}
    if "MatMul" in op_types and "Softmax" in op_types:
        return "transformer"
    if "Conv" in op_types and "Gemm" in op_types:
        return "resnet"
    if op_types.issubset({"Flatten", "Gemm", "Relu", "Reshape"}):
        return "mlp"
    return "unknown"


def load_data(name):
    idir = f'{DATA_BASE}/{name}/input'
    with open(f'{idir}/manifest.json') as f:
        m = json.load(f)
    data = {e['name']: np.load(f'{idir}/{e["file"]}') for e in m['tensors']}
    golden = np.load(f'{DATA_BASE}/{name}/golden/logits.npy')
    labels = None
    lp = f'{DATA_BASE}/{name}/labels.npy'
    if os.path.exists(lp):
        labels = np.load(lp)
    return data, golden, labels


def suppress_stderr():
    """Context manager to suppress stderr (for noisy memory pool logs)."""
    return open(os.devnull, 'w')


# ═══════════════════════════════════════════════════════════════════════════
print(f"{'Model':20s} {'Detect':>10s} {'Backend':>22s} "
      f"{'Auto':>8s} {'ORT':>8s} {'Δ':>7s} {'max_diff':>10s} {'Prec':>6s} {'Acc':>8s}")
print("-" * 110)

for name in MODELS:
    mp = f'{MODEL_BASE}/{name}.onnx'
    data, golden, labels = load_data(name)
    first = list(data.keys())[0]
    N = data[first].shape[0]
    B = 64 if name == 'resnet_v1' else 256

    model = onnx.load(mp)
    onames = [o.name for o in model.graph.output]

    detected = detect_model_type(mp)

    # ── ORT baseline (run first to avoid GPU warm-up bias) ────────────
    sess_ort = ort.InferenceSession(mp, providers=[
        ('CUDAExecutionProvider', {'device_id': '0'}), 'CPUExecutionProvider'])
    inames_ort = [i.name for i in sess_ort.get_inputs()]
    onames_ort = [o.name for o in sess_ort.get_outputs()]

    # ORT warmup
    _ = sess_ort.run(onames_ort,
                     {n: data[n][:min(8,N)] for n in inames_ort if n in data})

    t_ort0 = time.perf_counter()
    all_ort = []
    for start in range(0, N, B):
        end = min(start + B, N)
        feed = {n: data[n][start:end] for n in inames_ort if n in data}
        all_ort.append(sess_ort.run(onames_ort, feed)[0])
    t_ort1 = time.perf_counter()
    elapsed_ort = t_ort1 - t_ort0

    # ── Auto backend inference ────────────────────────────────────────
    t_auto0 = time.perf_counter()

    if detected == 'mlp':
        backend_name = "v2 executor"
        graph = import_onnx_graph(mp)
        raw_plan = decompose_graph(graph)
        opt_plan, stats = apply_fusion(graph, raw_plan)
        plan = opt_plan if len(opt_plan) < len(raw_plan) else raw_plan

        executor = GPUExecutor(mp)
        all_out = []
        for start in range(0, N, B):
            end = min(start + B, N)
            batch = {k: v[start:end] for k, v in data.items()}
            executor.load_inputs(batch)
            results = executor.execute_plan(plan, onames)
            all_out.append(results[onames[0]])
        out = np.concatenate(all_out, axis=0)

    elif detected == 'resnet':
        backend_name = "v2 + mempool"
        graph = import_onnx_graph(mp)
        raw_plan = decompose_graph(graph)
        opt_plan, stats = apply_fusion(graph, raw_plan)
        plan = raw_plan

        # Create executor once (weights loaded to GPU once)
        executor = MemoryAwareExecutor(mp)

        all_out = []
        first_batch = True
        for start in range(0, N, B):
            end = min(start + B, N)
            batch = {k: v[start:end] for k, v in data.items()}
            executor.load_inputs(batch)
            # Suppress memory pool log after first batch
            if first_batch:
                results = executor.execute_plan(plan, onames)
                first_batch = False
            else:
                with open(os.devnull, 'w') as devnull:
                    old_stderr = sys.stderr
                    sys.stderr = devnull
                    results = executor.execute_plan(plan, onames)
                    sys.stderr = old_stderr
            all_out.append(results[onames[0]])
        out = np.concatenate(all_out, axis=0)

    else:  # transformer → ORT (same as baseline)
        backend_name = "ORT"
        # ORT is the auto backend for transformer — reuse baseline results
        out = np.concatenate(all_ort, axis=0)

    t_auto1 = time.perf_counter()
    if detected != 'transformer':
        elapsed_auto = t_auto1 - t_auto0
    else:
        elapsed_auto = elapsed_ort

    # ── Accuracy check ────────────────────────────────────────────────
    if detected == 'transformer':
        # Transformer: sequence output (3D), precision check only (no argmax acc)
        ok = np.allclose(out, golden, rtol=1e-3, atol=1e-3)
        md = np.max(np.abs(out - golden))
        acc_str = "N/A"
    else:
        ok = np.allclose(out, golden, rtol=1e-3, atol=1e-3)
        md = np.max(np.abs(out - golden))
        if labels is not None and labels.ndim == 1:
            acc = (out.argmax(1) == labels).mean()
            acc_str = f"{acc*100:.2f}%"
        else:
            acc_str = "N/A"

    # ── Report ────────────────────────────────────────────────────────
    delta = (1 - elapsed_auto / elapsed_ort) * 100 if elapsed_ort > 0 else 0
    print(f"{name:20s} {detected:>10s} {backend_name:>22s} "
          f"{elapsed_auto:7.2f}s {elapsed_ort:7.2f}s {delta:+6.1f}% "
          f"{md:10.2e} {'OK' if ok else 'FAIL':>6s} {acc_str:>8s}")

print("-" * 110)
print("Done.")
