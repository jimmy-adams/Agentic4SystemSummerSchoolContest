#!/usr/bin/env python3
"""
C3.5 Mixed Precision: use ORT's float16 converter with op_block_list.
Sensitive ops blocked from fp16 conversion → stay fp32.
Non-sensitive ops converted to fp16 → fast compute.
"""

import os, time, json, numpy as np

os.environ['NVIDIA_TF32_OVERRIDE'] = '0'

import onnx
import onnxruntime as ort
from onnxruntime.transformers import float16

BASE_M = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models'
BASE_D = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35'
MODELS = ['mlp_v1', 'resnet_v1', 'transformer_v1']

# Sensitive ops — must stay fp32 (from C3.2 spec)
SENSITIVE_OPS = ['Softmax', 'LayerNormalization', 'ReduceMean', 'ReduceMax',
                 'ReduceSum', 'ReduceMin', 'ReduceProd', 'BatchNormalization',
                 'LRN', 'InstanceNormalization']


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
    sess = ort.InferenceSession(model_path, providers=[
        ('CUDAExecutionProvider', {'device_id': '0'}), 'CPUExecutionProvider'])
    inames = [i.name for i in sess.get_inputs()]
    onames = [o.name for o in sess.get_outputs()]
    _ = sess.run(onames, {n: data[n][:min(8,N)] for n in inames if n in data})
    t0 = time.perf_counter()
    all_out = []
    for start in range(0, N, B):
        end = min(start + B, N)
        all_out.append(sess.run(onames,
            {n: data[n][start:end] for n in inames if n in data})[0])
    return time.perf_counter() - t0, np.concatenate(all_out, axis=0)


def check(out, golden, labels):
    ok  = np.allclose(out, golden, rtol=1e-3, atol=1e-3)
    md  = np.max(np.abs(out - golden))
    acc = None
    if labels is not None and labels.ndim == 1:
        acc = (out.argmax(1) == labels).mean()
    return ok, md, acc


# ═══════════════════════════════════════════════════════════════════════════
print(f"C3.5 Mixed Precision: sensitive ops stay fp32, rest→fp16")
print(f"Blocked ops: {SENSITIVE_OPS}")
print()
print(f"{'Model':20s} {'Precision':>22s} {'Time':>8s} {'Δ':>8s} "
      f"{'max_diff':>10s} {'Gate':>6s} {'Acc':>10s}")
print("-" * 94)

for name in MODELS:
    data, golden, labels = load_data(name)
    N = list(data.values())[0].shape[0]
    B = 64 if name == 'resnet_v1' else 256
    mp = f'{BASE_M}/{name}.onnx'
    model = onnx.load(mp)

    # Count ops
    sens_in_model = [n.op_type for n in model.graph.node
                     if n.op_type in SENSITIVE_OPS]
    total_ops = len(model.graph.node)

    # ── fp32 ────────────────────────────────────────────────────────
    t32, out32 = infer_ort(mp, data, N, B)
    ok32, md32, acc32 = check(out32, golden, labels)

    # ── fp16 (all ops) ──────────────────────────────────────────────
    m16 = float16.convert_float_to_float16(model, keep_io_types=True)
    mp16 = f'/tmp/{name}_f16.onnx'
    onnx.save(m16, mp16)
    t16, out16 = infer_ort(mp16, data, N, B)
    ok16, md16, acc16 = check(out16, golden, labels)

    # ── Mixed: block sensitive ops ──────────────────────────────────
    blocked = [op for op in SENSITIVE_OPS if op in
               {n.op_type for n in model.graph.node}]
    mmix = float16.convert_float_to_float16(
        onnx.load(mp),  # fresh load
        keep_io_types=True,
        op_block_list=blocked,
    )
    mpmix = f'/tmp/{name}_mix.onnx'
    onnx.save(mmix, mpmix)
    tmix, outmix = infer_ort(mpmix, data, N, B)
    okmix, mdmix, accmix = check(outmix, golden, labels)

    # ── Report ──────────────────────────────────────────────────────
    def a(acc):
        return f"{acc*100:.2f}%" if acc is not None else "N/A"

    print(f"{name:20s} {'fp32':>22s} {t32:7.2f}s {'-':>8s} "
          f"{md32:10.2e} {'OK' if ok32 else 'FAIL':>6s} {a(acc32):>10s}")
    print(f"{name:20s} {'fp16 (all ops)':>22s} {t16:7.2f}s "
          f"{'+' if t32>t16 else ''}{abs(t32/t16-1)*100:5.0f}% "
          f"{md16:10.2e} {'OK' if ok16 else 'FAIL':>6s} {a(acc16):>10s}")
    print(f"{name:20s} {'mixed (block='+','.join(blocked)+')':>22s} {tmix:7.2f}s "
          f"{'+' if t32>tmix else ''}{abs(t32/tmix-1)*100:5.0f}% "
          f"{mdmix:10.2e} {'OK' if okmix else 'FAIL':>6s} {a(accmix):>10s}")
    print(f"  → {len(blocked)}/{total_ops} ops stay fp32, "
          f"{(total_ops-len(blocked))/total_ops*100:.0f}% in fp16")
    print()

    for f in [mp16, mpmix]:
        if os.path.exists(f):
            os.remove(f)

print("Done.")
