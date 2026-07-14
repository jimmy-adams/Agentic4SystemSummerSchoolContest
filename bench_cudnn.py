#!/usr/bin/env python3
"""
v2 executor optimization: cuDNN auto-tuning (benchmark mode) + TF32.

Tests whether enabling cuDNN optimizations closes the gap with ORT
while still passing the 1e-3 precision gate.
"""

import os, time, json, sys, numpy as np

os.environ['NVIDIA_TF32_OVERRIDE'] = '0'

import torch
import onnx
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scheduler.planner import import_onnx_graph, decompose_graph, apply_fusion
from scheduler.executor import GPUExecutor

BASE_M = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models'
BASE_D = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35'


def load_data(name):
    d = f'{BASE_D}/{name}'
    with open(f'{d}/input/manifest.json') as f:
        m = json.load(f)
    data = {e['name']: np.load(f'{d}/input/{e["file"]}') for e in m['tensors']}
    return data, np.load(f'{d}/golden/logits.npy')


def run_v2(mp, data, N, B, use_benchmark=False, use_tf32=False):
    """Run v2 executor with optional cuDNN optimizations."""
    # Configure cuDNN
    torch.backends.cudnn.benchmark = use_benchmark
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32

    graph = import_onnx_graph(mp)
    raw_plan = decompose_graph(graph)
    opt_plan, stats = apply_fusion(graph, raw_plan)
    plan = raw_plan  # ResNet: use raw (as validated)

    model = onnx.load(mp)
    onames = [o.name for o in model.graph.output]

    executor = GPUExecutor(mp)
    t0 = time.perf_counter()
    for start in range(0, N, B):
        end = min(start + B, N)
        batch = {k: v[start:end] for k, v in data.items()}
        executor.load_inputs(batch)
        executor.execute_plan(plan, onames)
    t = time.perf_counter() - t0

    # Restore defaults
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return t


def run_ort(mp, data, N, B):
    sess = ort.InferenceSession(mp, providers=[
        ('CUDAExecutionProvider', {'device_id': '0'}), 'CPUExecutionProvider'])
    inames = [i.name for i in sess.get_inputs()]
    onames = [o.name for o in sess.get_outputs()]
    _ = sess.run(onames, {n: data[n][:min(8,N)] for n in inames if n in data})
    t0 = time.perf_counter()
    for start in range(0, N, B):
        end = min(start + B, N)
        sess.run(onames, {n: data[n][start:end] for n in inames if n in data})
    return time.perf_counter() - t0


def precision_check(mp, data, golden):
    """Full precision check for v2 executor."""
    graph = import_onnx_graph(mp)
    raw_plan = decompose_graph(graph)
    model = onnx.load(mp)
    onames = [o.name for o in model.graph.output]

    executor = GPUExecutor(mp)
    first = list(data.keys())[0]
    N = data[first].shape[0]
    B = min(256, N)
    all_out = []
    for start in range(0, N, B):
        end = min(start+B, N)
        batch = {k: v[start:end] for k, v in data.items()}
        executor.load_inputs(batch)
        results = executor.execute_plan(raw_plan, onames)
        all_out.append(results[onames[0]])
    out = np.concatenate(all_out, axis=0)
    return np.max(np.abs(out - golden)), np.allclose(out, golden, rtol=1e-3, atol=1e-3)


# ═══════════════════════════════════════════════════════════════════════════
print("v2 Executor: cuDNN Optimization Test")
print("=" * 80)

for name in ['resnet_v1', 'mlp_v1']:
    data, golden = load_data(name)
    N = list(data.values())[0].shape[0]
    B = 64 if name == 'resnet_v1' else 256
    mp = f'{BASE_M}/{name}.onnx'

    print(f"\n--- {name} (N={N}, B={B}) ---")
    print(f"{'Config':40s} {'Time':>8s} {'Δ ORT':>8s} {'max_diff':>10s} {'Gate':>6s}")

    # ORT baseline
    t_ort = run_ort(mp, data, N, B)

    # Configurations
    configs = [
        ('v2 (default)', False, False),
        ('v2 + cudnn.benchmark', True, False),
        ('v2 + tf32', False, True),
        ('v2 + benchmark + tf32', True, True),
    ]

    for label, benchmark, tf32 in configs:
        try:
            t_v2 = run_v2(mp, data, N, B, use_benchmark=benchmark, use_tf32=tf32)
            md, ok = precision_check(mp, data, golden)
            delta = f"{(t_ort/t_v2-1)*100:+6.1f}%"
            print(f"  {label:38s} {t_v2:7.2f}s {delta:>8s} {md:10.2e} "
                  f"{'OK' if ok else 'FAIL':>6s}")
        except Exception as e:
            print(f"  {label:38s} ERROR: {str(e)[:50]}")

print("\nDone.")
