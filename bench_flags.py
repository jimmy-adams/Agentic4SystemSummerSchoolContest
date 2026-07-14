#!/usr/bin/env python3
"""Test all safe CUDA/cuDNN/ORT flags for cumulative speedup."""

import os, time, json, sys, numpy as np

os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

import torch
import onnxruntime as ort
from scheduler.planner import import_onnx_graph, decompose_graph, apply_fusion
from scheduler.executor import MemoryAwareExecutor

BASE_M = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models'
BASE_D = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35'


def load_data(name):
    d = f'{BASE_D}/{name}'
    with open(f'{d}/input/manifest.json') as f:
        m = json.load(f)
    data = {e['name']: np.load(f'{d}/input/{e["file"]}') for e in m['tensors']}
    return data, np.load(f'{d}/golden/logits.npy')


def run_v2(mp, data, N, B):
    graph = import_onnx_graph(mp)
    raw_plan = decompose_graph(graph)
    opt_plan, stats = apply_fusion(graph, raw_plan)
    plan = raw_plan
    model = __import__('onnx').load(mp)  # avoid top-level import
    onames = [o.name for o in model.graph.output]
    executor = MemoryAwareExecutor(mp)
    t0 = time.perf_counter()
    for start in range(0, N, B):
        end = min(start + B, N)
        batch = {k: v[start:end] for k, v in data.items()}
        executor.load_inputs(batch)
        with open(os.devnull, 'w') as f:
            old = sys.stderr
            sys.stderr = f
            executor.execute_plan(plan, onames)
            sys.stderr = old
    return time.perf_counter() - t0


def run_ort(mp, data, N, B):
    opts = ort.SessionOptions()
    sess = ort.InferenceSession(mp, sess_options=opts, providers=[
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
    graph = import_onnx_graph(mp)
    raw_plan = decompose_graph(graph)
    model = __import__('onnx').load(mp)
    onames = [o.name for o in model.graph.output]
    executor = MemoryAwareExecutor(mp)
    first = list(data.keys())[0]
    N = data[first].shape[0]
    B = min(256, N)
    all_out = []
    for start in range(0, N, B):
        end = min(start+B, N)
        batch = {k: v[start:end] for k, v in data.items()}
        executor.load_inputs(batch)
        with open(os.devnull, 'w') as f:
            old = sys.stderr
            sys.stderr = f
            results = executor.execute_plan(raw_plan, onames)
            sys.stderr = old
        all_out.append(results[onames[0]])
    out = np.concatenate(all_out, axis=0)
    return np.max(np.abs(out - golden)), np.allclose(out, golden, rtol=1e-3, atol=1e-3)


# ═══════════════════════════════════════════════════════════════════════════
name = 'resnet_v1'
data, golden = load_data(name)
N = list(data.values())[0].shape[0]
B = 64
mp = f'{BASE_M}/{name}.onnx'

print(f"ResNet v2 + mempool: testing cumulative flags")
print(f"{'Flags':55s} {'Time':>8s} {'Δ%':>8s} {'max_diff':>10s} {'Gate':>6s}")
print("-" * 95)

# Test each flag individually and cumulatively
tests = [
    ('baseline (cudnn.benchmark only)', lambda: (
        setattr(torch.backends.cudnn, 'benchmark', True), None)[1]),
    ('+ allow_tf32 (matmul)', lambda: (
        setattr(torch.backends.cudnn, 'benchmark', True),
        setattr(torch.backends.cuda.matmul, 'allow_tf32', True), None)[-1]),
    ('+ allow_tf32 (cudnn)', lambda: (
        setattr(torch.backends.cudnn, 'benchmark', True),
        setattr(torch.backends.cuda.matmul, 'allow_tf32', True),
        setattr(torch.backends.cudnn, 'allow_tf32', True), None)[-1]),
    ('+ inference_mode()', lambda: (
        setattr(torch.backends.cudnn, 'benchmark', True),
        setattr(torch.backends.cuda.matmul, 'allow_tf32', True),
        setattr(torch.backends.cudnn, 'allow_tf32', True),
        torch.inference_mode().__enter__(), None)[-1]),
]

baseline = None
for label, setup_fn in tests:
    # Reset
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    setup_fn()

    t = run_v2(mp, data, N, B)
    md, ok = precision_check(mp, data, golden)

    if baseline is None:
        baseline = t
        delta = "-"
    else:
        delta = f"{(baseline/t-1)*100:+6.1f}%"

    print(f"  {label:53s} {t:7.2f}s {delta:>8s} {md:10.2e} {'OK' if ok else 'FAIL':>6s}")

# Also test ORT with optimal settings
print(f"\n  --- ORT ---")
t_ort = run_ort(mp, data, N, B)
print(f"  {'CUDA EP (baseline)':53s} {t_ort:7.2f}s")
print(f"  {'v2 best vs ORT':53s} {(t_ort/baseline-1)*100:+6.1f}%")

# Reset
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

print("\nDone.")
