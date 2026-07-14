#!/usr/bin/env python3
"""
C3.5 optimization exploration: ORT tuning, CUDA Graph, batch size, streams.

Tests:
  1. Graph optimization level (BASIC vs EXTENDED vs ALL)
  2. CUDA Graph enabled vs disabled
  3. Execution mode (parallel vs sequential)
  4. Batch size sweep
  5. Intra-op thread count
"""

import os, time, json, sys, numpy as np

os.environ['NVIDIA_TF32_OVERRIDE'] = '0'

import onnxruntime as ort

BASE_M = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models'
BASE_D = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35'


def load_data(name):
    d = f'{BASE_D}/{name}'
    with open(f'{d}/input/manifest.json') as f:
        m = json.load(f)
    data = {e['name']: np.load(f'{d}/input/{e["file"]}') for e in m['tensors']}
    return data, np.load(f'{d}/golden/logits.npy')


def infer_ort(mp, data, N, B, providers, sess_opts=None):
    sess = ort.InferenceSession(mp, sess_options=sess_opts, providers=providers)
    inames = [i.name for i in sess.get_inputs()]
    onames = [o.name for o in sess.get_outputs()]
    _ = sess.run(onames, {n: data[n][:min(8,N)] for n in inames if n in data})
    t0 = time.perf_counter()
    for start in range(0, N, B):
        end = min(start + B, N)
        sess.run(onames, {n: data[n][start:end] for n in inames if n in data})
    return time.perf_counter() - t0


def check(out, golden):
    return np.max(np.abs(out - golden)), np.allclose(out, golden, rtol=1e-3, atol=1e-3)


# ═══════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("C3.5 Optimization Exploration: ORT Tuning")
print("=" * 80)

for name in ['resnet_v1', 'mlp_v1', 'transformer_v1']:
    data, golden = load_data(name)
    N = list(data.values())[0].shape[0]
    mp = f'{BASE_M}/{name}.onnx'

    results = []

    # Baseline
    base_providers = [('CUDAExecutionProvider', {'device_id': '0'}), 'CPUExecutionProvider']

    # ── Test 1: Graph optimization level ─────────────────────────────
    print(f"\n--- {name} ---")
    print(f"{'Config':45s} {'Time':>8s} {'Δ%':>7s} {'max_diff':>10s} {'Gate':>6s}")

    for opt_name, opt_level in [
        ('BASIC', ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
        ('EXTENDED', ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED),
        ('ALL (default)', ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    ]:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = opt_level
        B = 64 if name == 'resnet_v1' else 256
        t = infer_ort(mp, data, N, B, base_providers, opts)
        # Check precision (just one batch for speed)
        sess = ort.InferenceSession(mp, sess_options=opts, providers=base_providers)
        inames = [i.name for i in sess.get_inputs()]
        onames = [o.name for o in sess.get_outputs()]
        feed = {n: data[n][:min(64,N)] for n in inames if n in data}
        out = sess.run(onames, feed)[0]
        md, ok = check(out, golden[:min(64,len(golden))])
        delta = ""
        if len(results) > 0:
            base_t = results[0][1]
            delta = f"{(base_t/t-1)*100:+6.1f}%"
        results.append((opt_name, t, md, ok))
        print(f"  {opt_name:43s} {t:7.2f}s {delta:>7s} {md:10.2e} {'OK' if ok else 'FAIL':>6s}")

    # ── Test 2: CUDA execution mode ──────────────────────────────────
    opts = ort.SessionOptions()
    for exec_mode in [ort.ExecutionMode.ORT_SEQUENTIAL, ort.ExecutionMode.ORT_PARALLEL]:
        opts.execution_mode = exec_mode
        mode_name = 'SEQUENTIAL' if exec_mode == ort.ExecutionMode.ORT_SEQUENTIAL else 'PARALLEL'
        t = infer_ort(mp, data, N, B, base_providers, opts)
        base_t = results[0][1]
        print(f"  exec_mode={mode_name:35s} {t:7.2f}s {(base_t/t-1)*100:+6.1f}%")

    # ── Test 3: CUDA Graph ──────────────────────────────────────────
    for cuda_graph in [True, False]:
        if cuda_graph:
            providers = [('CUDAExecutionProvider', {
                'device_id': '0',
                'enable_cuda_graph': '1',
            }), 'CPUExecutionProvider']
        else:
            providers = base_providers
        try:
            t = infer_ort(mp, data, N, B, providers)
            base_t = results[0][1]
            print(f"  cuda_graph={'on':>3s}                                 {t:7.2f}s {(base_t/t-1)*100:+6.1f}%")
        except Exception as e:
            print(f"  cuda_graph={'on':>3s}                                 ERROR: {str(e)[:60]}")

    # ── Test 4: Intra-op threads ────────────────────────────────────
    for nthreads in [4, 8, 16]:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = nthreads
        t = infer_ort(mp, data, N, B, base_providers, opts)
        base_t = results[0][1]
        print(f"  intra_op_threads={nthreads:2d}                          {t:7.2f}s {(base_t/t-1)*100:+6.1f}%")

    # ── Test 5: Batch size sweep ────────────────────────────────────
    opts = ort.SessionOptions()
    for bs in [128, 256, 512, 1024, 2048]:
        try:
            t = infer_ort(mp, data, N, bs, base_providers, opts)
            base_t = results[0][1]
            print(f"  batch_size={bs:<4d}                                 {t:7.2f}s {(base_t/t-1)*100:+6.1f}%")
        except Exception as e:
            print(f"  batch_size={bs:<4d}                                 OOM or ERROR")

print("\nDone.")
