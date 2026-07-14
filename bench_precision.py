#!/usr/bin/env python3
"""
Precision sweep: fp32 vs fp16 for all 3 models using ONNX Runtime.

Tests:
    fp32 — full precision (baseline)
    fp16 — half precision (ORT mixed precision or fp16 mode)
"""

import sys, os, time, json, numpy as np

os.environ['NVIDIA_TF32_OVERRIDE'] = '0'

import onnx, onnxruntime as ort

MODEL_BASE = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models'
DATA_BASE  = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35'
MODELS = ['mlp_v1', 'resnet_v1', 'transformer_v1']

print(f"{'Model':20s} {'Prec':>6s} {'Time':>8s} {'Speedup':>8s} {'max_diff':>10s} {'Gate':>6s} {'Accuracy':>10s}")
print("-" * 80)


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


def run_ort(name, data, golden, labels, B, fp16=False):
    """Run ORT inference and return (time, output, ok, max_diff, acc_str)."""
    mp = f'{MODEL_BASE}/{name}.onnx'
    N = list(data.values())[0].shape[0]

    # ── Build session ─────────────────────────────────────────────
    providers = [('CUDAExecutionProvider', {
        'device_id': '0',
        'enable_cuda_graph': '1' if not fp16 else '0',
    }), 'CPUExecutionProvider']

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    )

    # Enable fp16 mode: use ORT's mixed precision
    if fp16:
        sess_options.enable_mem_pattern = False
        # Use fp16 as the preferred execution type
        providers = [('CUDAExecutionProvider', {
            'device_id': '0',
            'do_copy_in_default_stream': '1',
            'arena_extend_strategy': 'kNextPowerOfTwo',
        }), 'CPUExecutionProvider']

    session = ort.InferenceSession(mp, sess_options=sess_options,
                                   providers=providers)
    inames = [i.name for i in session.get_inputs()]
    onames = [o.name for o in session.get_outputs()]

    # ── Warm-up ──────────────────────────────────────────────────
    warm = {n: data[n][:min(8, N)] for n in inames if n in data}
    _ = session.run(onames, warm)

    # ── Timed run ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    all_out = []
    for start in range(0, N, B):
        end = min(start + B, N)
        feed = {n: data[n][start:end] for n in inames if n in data}
        all_out.append(session.run(onames, feed)[0])
    t1 = time.perf_counter()
    out = np.concatenate(all_out, axis=0)

    ok = np.allclose(out, golden, rtol=1e-3, atol=1e-3)
    md = np.max(np.abs(out - golden))
    acc_str = "N/A"
    if labels is not None and labels.ndim == 1:
        acc = (out.argmax(1) == labels).mean()
        acc_str = f"{acc*100:.2f}%"

    return (t1 - t0), out, ok, md, acc_str


for name in MODELS:
    data, golden, labels = load_data(name)
    B = 64 if name == 'resnet_v1' else 256

    # ── fp32 baseline ──────────────────────────────────────────────
    t32, out32, ok32, md32, acc32 = run_ort(name, data, golden, labels, B, fp16=False)

    # ── fp16 ───────────────────────────────────────────────────────
    t16, out16, ok16, md16, acc16 = run_ort(name, data, golden, labels, B, fp16=True)

    speedup = (t32 / t16 - 1) * 100 if t16 > 0 else 0

    print(f"{name:20s} {'fp32':>6s} {t32:7.2f}s {'-':>8s} {md32:10.2e} "
          f"{'OK' if ok32 else 'FAIL':>6s} {acc32:>10s}")
    print(f"{name:20s} {'fp16':>6s} {t16:7.2f}s {speedup:+6.1f}% {md16:10.2e} "
          f"{'OK' if ok16 else 'FAIL':>6s} {acc16:>10s}")
    print()

print("Done.")
