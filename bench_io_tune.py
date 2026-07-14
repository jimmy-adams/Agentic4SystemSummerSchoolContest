#!/usr/bin/env python3
"""
C3.5: I/O and memory optimization exploration.

Tests:
  1. Regular ORT (baseline)
  2. ORT with I/O binding (pre-allocated GPU buffers)
  3. Pinned memory + async transfer
  4. Memory-mapped file loading
  5. Threaded I/O prefetch
"""

import os, time, json, threading, numpy as np

os.environ['NVIDIA_TF32_OVERRIDE'] = '0'

import onnxruntime as ort

BASE_M = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models'
BASE_D = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35'

def load_data(name):
    d = f'{BASE_D}/{name}'
    with open(f'{d}/input/manifest.json') as f:
        m = json.load(f)
    data = {e['name']: np.load(f'{d}/input/{e["file"]}') for e in m['tensors']}
    return data


def infer_ort_baseline(mp, data, N, B):
    """Standard ORT inference (current baseline)."""
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


def infer_ort_iobinding(mp, data, N, B):
    """ORT with I/O binding — pre-allocated GPU buffers, zero-copy."""
    import torch
    sess = ort.InferenceSession(mp, providers=[
        ('CUDAExecutionProvider', {'device_id': '0'}), 'CPUExecutionProvider'])
    inames = [i.name for i in sess.get_inputs()]
    onames = [o.name for o in sess.get_outputs()]

    io_binding = sess.io_binding()

    # Pre-allocate output buffer on GPU
    first_in = data[inames[0]]
    sample_in = first_in[:B]
    sample_out = sess.run(onames, {n: data[n][:B] for n in inames if n in data})[0]

    # Warmup
    _ = sess.run(onames, {n: data[n][:min(8,N)] for n in inames if n in data})

    t0 = time.perf_counter()
    for start in range(0, N, B):
        end = min(start + B, N)
        io_binding.clear_binding_inputs()
        io_binding.clear_binding_outputs()

        for n in inames:
            if n in data:
                io_binding.bind_cpu_input(n, data[n][start:end])

        # Bind output to pre-allocated GPU buffer
        out_buf = torch.empty(sample_out.shape, dtype=torch.float32, device='cuda')
        io_binding.bind_output(onames[0], out_buf.device.type)

        sess.run_with_iobinding(io_binding)
    return time.perf_counter() - t0


def infer_ort_threaded(mp, data, N, B):
    """Threaded I/O: prefetch next batch while GPU computes."""
    import queue

    sess = ort.InferenceSession(mp, providers=[
        ('CUDAExecutionProvider', {'device_id': '0'}), 'CPUExecutionProvider'])
    inames = [i.name for i in sess.get_inputs()]
    onames = [o.name for o in sess.get_outputs()]
    _ = sess.run(onames, {n: data[n][:min(8,N)] for n in inames if n in data})

    # Pre-fetch queue
    q = queue.Queue(maxsize=2)

    def prefetch():
        for start in range(0, N, B):
            end = min(start + B, N)
            q.put({n: data[n][start:end].copy() for n in inames if n in data})
        q.put(None)  # sentinel

    t = threading.Thread(target=prefetch, daemon=True)
    t.start()

    t0 = time.perf_counter()
    while True:
        feed = q.get()
        if feed is None:
            break
        sess.run(onames, feed)
    t.join()
    return time.perf_counter() - t0


# ═══════════════════════════════════════════════════════════════════════════
print(f"{'Model':20s} {'Method':25s} {'Time':>8s} {'Δ%':>8s}")
print("-" * 65)

for name in ['resnet_v1', 'transformer_v1', 'mlp_v1']:
    data = load_data(name)
    N = list(data.values())[0].shape[0]
    B = 64 if name == 'resnet_v1' else 256
    mp = f'{BASE_M}/{name}.onnx'

    print(f"\n--- {name} (N={N}, B={B}) ---")

    t_baseline = infer_ort_baseline(mp, data, N, B)
    print(f"  {'baseline':25s} {t_baseline:7.2f}s {'-':>8s}")

    try:
        t_io = infer_ort_iobinding(mp, data, N, B)
        print(f"  {'IO binding':25s} {t_io:7.2f}s {(t_baseline/t_io-1)*100:+7.1f}%")
    except Exception as e:
        print(f"  {'IO binding':25s} ERROR: {str(e)[:50]}")

    try:
        t_thread = infer_ort_threaded(mp, data, N, B)
        print(f"  {'threaded prefetch':25s} {t_thread:7.2f}s {(t_baseline/t_thread-1)*100:+7.1f}%")
    except Exception as e:
        print(f"  {'threaded prefetch':25s} ERROR: {str(e)[:50]}")

print("\nDone.")
