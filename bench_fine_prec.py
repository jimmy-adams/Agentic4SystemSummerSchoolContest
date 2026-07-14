#!/usr/bin/env python3
"""
Fine-grained mixed precision: test blocking different op combinations
to find the minimum set that passes the 1e-3 gate.

Key hypothesis: MatMul accumulation across layers is the main precision loss.
"""

import os, time, json, numpy as np

os.environ['NVIDIA_TF32_OVERRIDE'] = '0'

import onnx
import onnxruntime as ort
from onnxruntime.transformers import float16

BASE_M = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models'
BASE_D = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35'


def load_data(name):
    d = f'{BASE_D}/{name}'
    with open(f'{d}/input/manifest.json') as f:
        m = json.load(f)
    data = {e['name']: np.load(f'{d}/input/{e["file"]}') for e in m['tensors']}
    golden = np.load(f'{d}/golden/logits.npy')
    return data, golden


def infer_ort(model_path, data, N, B):
    # Note: default ORT optimization (ORT_ENABLE_ALL) gives best transformer precision
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


def test_config(name, mp, data, golden, N, B, blocked, label):
    """Test a specific blocking configuration. Returns (time, max_diff, ok)."""
    model = onnx.load(mp)
    ops_present = {n.op_type for n in model.graph.node}
    actual_blocked = [op for op in blocked if op in ops_present]

    try:
        m16 = float16.convert_float_to_float16(
            model, keep_io_types=True, op_block_list=actual_blocked)
        tmp = f'/tmp/{name}_test.onnx'
        onnx.save(m16, tmp)
        t, out = infer_ort(tmp, data, N, B)
        os.remove(tmp)
        md = np.max(np.abs(out - golden))
        ok = np.allclose(out, golden, rtol=1e-3, atol=1e-3)
        return t, md, ok
    except Exception as e:
        err_msg = str(e)[:80]
        return 0, 0, None  # signal error


# ═══════════════════════════════════════════════════════════════════════════
print("Fine-grained Mixed Precision: finding minimal blocking set")
print()

for name in ['transformer_v1', 'resnet_v1', 'mlp_v1']:
    data, golden = load_data(name)
    N = list(data.values())[0].shape[0]
    B = 64 if name == 'resnet_v1' else 256
    mp = f'{BASE_M}/{name}.onnx'

    model = onnx.load(mp)
    ops = {n.op_type for n in model.graph.node}
    total = len(model.graph.node)

    print(f"=== {name} ({total} ops, types: {sorted(ops)}) ===")

    # Baseline
    t32, md32, ok32 = test_config(name, mp, data, golden, N, B, [], 'fp32')
    print(f"  {'fp32':30s}  {t32:7.2f}s  max_diff={md32:.2e}  {'OK' if ok32 else 'FAIL'}")

    # Strategy: test cumulative blocking
    configs = []

    if 'Softmax' in ops or 'LayerNormalization' in ops:
        configs.append((['Softmax', 'LayerNormalization'], 'block Softmax+LN'))

    if 'MatMul' in ops:
        configs.append((['Softmax', 'LayerNormalization', 'MatMul'],
                        'block +MatMul'))
        configs.append((['MatMul'], 'block MatMul only'))

    if 'Gemm' in ops:
        configs.append((['Softmax', 'LayerNormalization', 'Gemm'],
                        'block +Gemm'))

    if 'Conv' in ops:
        configs.append((['Conv'], 'block Conv only'))
        configs.append((['Conv', 'Gemm'], 'block Conv+Gemm'))

    # Also try: block all compute-heavy ops
    heavy = [o for o in ['MatMul', 'Gemm', 'Conv'] if o in ops]
    sensitive = [o for o in ['Softmax', 'LayerNormalization',
                              'ReduceMean', 'ReduceMax',
                              'ReduceSum', 'BatchNormalization'] if o in ops]
    if heavy or sensitive:
        configs.append((sensitive + heavy, f'block all heavy+sensitive ({len(sensitive+heavy)} ops)'))

    for blocked, label in configs:
        t, md, ok = test_config(name, mp, data, golden, N, B, blocked, label)
        if ok is None:
            print(f"  {label:30s}  ERROR (dup cast names)")
        else:
            speedup = (t32 / t - 1) * 100 if t > 0 else 0
            gate = '✅' if ok else ('⚠️' if md < 5e-3 else '❌')
            print(f"  {label:30s}  {t:7.2f}s ({speedup:+5.0f}%)  "
                  f"max_diff={md:.2e}  {gate}")

    print()

print("Done.")
