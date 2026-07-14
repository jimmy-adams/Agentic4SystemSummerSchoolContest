#!/usr/bin/env python3
"""C3.5 benchmark at batch_size=256 (contest spec default)."""
import sys, time, numpy as np, json, os

# MUST be before ORT import
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'

sys.path.insert(0, '/home/mig20/c3_solution')

from scheduler.planner import import_onnx_graph, decompose_graph, apply_fusion
import onnx, onnxruntime as ort

MODEL_BASE = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models'
DATA_BASE = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35'

print(f"{'Model':20s} {'kernels':>12s} {'time':>8s} {'precision':>12s}")
print("-" * 58)

total = 0
for name in ['mlp_v1', 'resnet_v1', 'transformer_v1']:
    mp = f'{MODEL_BASE}/{name}.onnx'
    idir = f'{DATA_BASE}/{name}/input'
    gp = f'{DATA_BASE}/{name}/golden/logits.npy'

    with open(f'{idir}/manifest.json') as f: m = json.load(f)
    data = {e['name']: np.load(f'{idir}/{e["file"]}') for e in m['tensors']}
    first = list(data.keys())[0]
    N = data[first].shape[0]
    B = min(256, N)

    # Scheduler stats
    graph = import_onnx_graph(mp)
    raw = decompose_graph(graph)
    opt, _ = apply_fusion(graph, raw)
    kern_str = f"{len(raw)}->{len(opt)}" if len(opt) < len(raw) else f"{len(raw)}"

    # ORT inference
    sess = ort.InferenceSession(mp, providers=[
        ('CUDAExecutionProvider', {'device_id': '0'}), 'CPUExecutionProvider'])
    inames = [i.name for i in sess.get_inputs()]
    onames = [o.name for o in sess.get_outputs()]

    # Warmup
    feed = {n: data[n][:B] for n in inames if n in data}
    _ = sess.run(onames, feed)

    t0 = time.perf_counter()
    for start in range(0, N, B):
        end = min(start + B, N)
        feed = {n: data[n][start:end] for n in inames if n in data}
        _ = sess.run(onames, feed)
    t1 = time.perf_counter()
    et = t1 - t0
    total += et

    # Verify
    out_full = []
    for start in range(0, N, B):
        end = min(start + B, N)
        feed = {n: data[n][start:end] for n in inames if n in data}
        out_full.append(sess.run(onames, feed)[0])
    out_all = np.concatenate(out_full, axis=0)
    gold = np.load(gp)
    ok = np.allclose(out_all, gold, rtol=1e-3, atol=1e-3)
    md = np.max(np.abs(out_all - gold))

    print(f"{name:20s} {kern_str:>12s} {et:7.2f}s {'OK' if ok else f'FAIL({md:.1e})':>12s}")

print("-" * 58)
print(f"{'TOTAL':20s} {'':>12s} {total:7.2f}s")
print(f"\n* batch_size=256 (contest spec default)")
