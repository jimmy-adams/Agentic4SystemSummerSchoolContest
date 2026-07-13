#!/usr/bin/env python3
"""v2 executor (raw) vs ORT — per-batch comparison."""
import sys, time, numpy as np, json, os, torch
sys.path.insert(0, '/home/mig20/c3_solution')
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'

from scheduler.planner import import_onnx_graph, decompose_graph
from scheduler.executor import GPUExecutor
import onnx, onnxruntime as ort

MODEL_BASE = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models'
DATA_BASE = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35'

for name in ['mlp_v1', 'resnet_v1']:
    mp = f'{MODEL_BASE}/{name}.onnx'
    idir = f'{DATA_BASE}/{name}/input'
    gp = f'{DATA_BASE}/{name}/golden/logits.npy'

    with open(f'{idir}/manifest.json') as f: m = json.load(f)
    data = {e['name']: np.load(f'{idir}/{e["file"]}') for e in m['tensors']}
    model = onnx.load(mp)
    onames = [o.name for o in model.graph.output]

    B = 64 if name == 'resnet_v1' else 256
    batch = {k: v[:B] for k, v in data.items()}

    # V2
    graph = import_onnx_graph(mp)
    plan = decompose_graph(graph)
    exe = GPUExecutor(mp)
    exe.load_inputs(batch)
    _ = exe.execute_plan(plan, onames)  # warmup
    t0 = time.perf_counter()
    out_v2 = exe.execute_plan(plan, onames)[onames[0]]
    t1 = time.perf_counter()

    gold = np.load(gp)[:B]
    ok = np.allclose(out_v2, gold, rtol=1e-3, atol=1e-3)

    # ORT
    sess = ort.InferenceSession(mp, providers=[('CUDAExecutionProvider',{'device_id':'0'}),'CPUExecutionProvider'])
    inames = [i.name for i in sess.get_inputs()]
    feed = {n: batch[n] for n in inames if n in batch}
    _ = sess.run(onames, feed)  # warmup
    t2 = time.perf_counter()
    _ = sess.run(onames, feed)
    t3 = time.perf_counter()

    tv2, tort = t1-t0, t3-t2
    print(f"{name:20s} {len(plan):3d}kern v2={tv2*1000:5.1f}ms ort={tort*1000:5.1f}ms "
          f"{(1-tv2/tort)*100:+5.0f}% {'OK' if ok else 'FAIL'}")
