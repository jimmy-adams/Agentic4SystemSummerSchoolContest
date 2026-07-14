#!/usr/bin/env python3
"""Compare v2 raw vs v2 fused execution for MLP + ResNet."""
import sys, time, numpy as np, json, os
sys.path.insert(0, '/home/mig20/c3_solution')
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'

from scheduler.planner import import_onnx_graph, decompose_graph, apply_fusion
from scheduler.executor import GPUExecutor
import onnx

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
    gold_batch = np.load(gp)[:B]

    graph = import_onnx_graph(mp)
    raw_plan = decompose_graph(graph)
    opt_plan, stats = apply_fusion(graph, raw_plan)

    # Raw
    exe1 = GPUExecutor(mp); exe1.load_inputs(batch)
    _ = exe1.execute_plan(raw_plan, onames)
    t0 = time.perf_counter()
    out1 = exe1.execute_plan(raw_plan, onames)[onames[0]]
    t1 = time.perf_counter()

    ok1 = np.allclose(out1, gold_batch, rtol=1e-3, atol=1e-3)

    # Fused
    exe2 = GPUExecutor(mp); exe2.load_inputs(batch)
    _ = exe2.execute_plan(opt_plan, onames)
    t2 = time.perf_counter()
    out2 = exe2.execute_plan(opt_plan, onames)[onames[0]]
    t3 = time.perf_counter()

    ok2 = np.allclose(out2, gold_batch, rtol=1e-3, atol=1e-3)

    tr, tf = t1-t0, t3-t2
    speedup = (1-tf/tr)*100
    print(f"{name:20s} raw={len(raw_plan)}k {tr*1000:.1f}ms  "
          f"fused={len(opt_plan)}k {tf*1000:.1f}ms  "
          f"{speedup:+5.0f}%  "
          f"raw={'OK' if ok1 else 'FAIL'} fused={'OK' if ok2 else 'FAIL'}")
