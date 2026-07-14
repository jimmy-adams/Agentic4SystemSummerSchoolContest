#!/usr/bin/env python3
"""C3.5 precision impact: fp32 vs fp16 speed/accuracy tradeoff."""

import os, time, json, sys, numpy as np

os.environ['NVIDIA_TF32_OVERRIDE'] = '0'

import onnx
import onnxruntime as ort
from onnxruntime.transformers import float16  # ORT's fp16 converter

BASE_M = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models'
BASE_D = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35'
MODELS = ['mlp_v1', 'resnet_v1', 'transformer_v1']

GATE_RTOL = 1e-3
GATE_ACC  = 0.85


def load_data(name):
    d = f'{BASE_D}/{name}'
    with open(f'{d}/input/manifest.json') as f:
        m = json.load(f)
    data = {e['name']: np.load(f'{d}/input/{e["file"]}') for e in m['tensors']}
    golden = np.load(f'{d}/golden/logits.npy')
    labels = None
    lp = f'{d}/labels.npy'
    if os.path.exists(lp):
        labels = np.load(lp)
    return data, golden, labels


def infer_ort(model_path, data, N, B):
    """Standard fp32 ORT inference. Returns (time, output)."""
    sess = ort.InferenceSession(model_path, providers=[
        ('CUDAExecutionProvider', {'device_id': '0'}), 'CPUExecutionProvider'])
    inames = [i.name for i in sess.get_inputs()]
    onames = [o.name for o in sess.get_outputs()]

    # Warm-up
    _ = sess.run(onames, {n: data[n][:min(8,N)] for n in inames if n in data})

    t0 = time.perf_counter()
    all_out = []
    for start in range(0, N, B):
        end = min(start + B, N)
        all_out.append(sess.run(onames,
            {n: data[n][start:end] for n in inames if n in data})[0])
    return time.perf_counter() - t0, np.concatenate(all_out, axis=0)


def check(out, golden, labels):
    ok  = np.allclose(out, golden, rtol=GATE_RTOL, atol=GATE_RTOL)
    md  = np.max(np.abs(out - golden))
    acc = None
    if labels is not None and labels.ndim == 1:
        acc = (out.argmax(1) == labels).mean()
    return ok, md, acc


# ═══════════════════════════════════════════════════════════════════════════
print(f"C3.5 Precision Impact: fp32 vs fp16")
print(f"Gate: rtol={GATE_RTOL}, acc≥{GATE_RTOL}")
print()
print(f"{'Model':20s} {'Prec':>6s} {'Time':>8s} {'Speedup':>8s} "
      f"{'max_diff':>10s} {'Gate':>6s} {'Acc':>10s}")
print("-" * 78)

for name in MODELS:
    data, golden, labels = load_data(name)
    N = list(data.values())[0].shape[0]
    B = 64 if name == 'resnet_v1' else 256
    mp = f'{BASE_M}/{name}.onnx'

    # ── fp32 ────────────────────────────────────────────────────────
    t32, out32 = infer_ort(mp, data, N, B)
    ok32, md32, acc32 = check(out32, golden, labels)

    # ── fp16: convert model to float16 ──────────────────────────────
    model = onnx.load(mp)
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    mp16 = f'/tmp/{name}_fp16.onnx'
    onnx.save(model_fp16, mp16)

    # Some models (e.g., with int64 inputs) can't be fully FP16
    # ORT handles mixed precision internally
    try:
        t16, out16 = infer_ort(mp16, data, N, B)
        ok16, md16, acc16 = check(out16, golden, labels)
        speedup = (t32 / t16 - 1) * 100 if t16 > 0 else 0
    except Exception as e:
        t16, ok16, md16, acc16, speedup = 0, False, 0, None, 0
        print(f"  fp16 ERROR: {e}")

    # ── Report ──────────────────────────────────────────────────────
    def acc_str(acc):
        return f"{acc*100:.2f}%" if acc is not None else "N/A"

    print(f"{name:20s} {'fp32':>6s} {t32:7.2f}s {'-':>8s} "
          f"{md32:10.2e} {'OK' if ok32 else 'FAIL':>6s} {acc_str(acc32):>10s}")

    if t16 > 0:
        print(f"{name:20s} {'fp16':>6s} {t16:7.2f}s {speedup:+6.1f}% "
              f"{md16:10.2e} {'OK' if ok16 else 'FAIL':>6s} {acc_str(acc16):>10s}")

    os.remove(mp16)  # cleanup
    print()

print("Done.")
