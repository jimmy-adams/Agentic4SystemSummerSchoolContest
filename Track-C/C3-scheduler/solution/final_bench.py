#!/usr/bin/env python3  
"""Final benchmark: scheduler statistics + ORT execution."""
import sys, time, numpy as np, json, subprocess, os
sys.path.insert(0, '/home/mig20/c3_solution')
from scheduler.planner import import_onnx_graph, decompose_graph, apply_fusion

os.environ['NVIDIA_TF32_OVERRIDE'] = '0'

MODEL_BASE = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models'
DATA_BASE = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35'
SOL = '/home/mig20/c3_solution'

print(f"{'Model':20s} {'raw':>5s} {'opt':>5s} {'reduce':>7s} {'time':>7s} {'precision':>12s}")
print("-" * 62)

total_time = 0
for name in ['mlp_v1', 'resnet_v1', 'transformer_v1']:
    model_path = f'{MODEL_BASE}/{name}.onnx'
    input_dir = f'{DATA_BASE}/{name}/input'
    golden_path = f'{DATA_BASE}/{name}/golden/logits.npy'

    # Scheduler: compute kernel plan reduction
    graph = import_onnx_graph(model_path)
    raw_plan = decompose_graph(graph)
    opt_plan, stats = apply_fusion(graph, raw_plan)
    reduction = f"{(1-len(opt_plan)/max(len(raw_plan),1))*100:.0f}%"

    # ORT: execute
    t0 = time.perf_counter()
    r = subprocess.run(
        ['bash', f'{SOL}/run_infer.sh', '--onnx', model_path,
         '--input', input_dir, '--output', f'/tmp/final_{name}',
         '--batch-size', '2048'],
        capture_output=True, text=True)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    total_time += elapsed

    # Verify
    out = np.load(f'/tmp/final_{name}/logits.npy')
    gold = np.load(golden_path)
    ok = np.allclose(out, gold, rtol=1e-3, atol=1e-3)
    md = np.max(np.abs(out - gold))

    print(f"{name:20s} {len(raw_plan):5d} {len(opt_plan):5d} {reduction:>7s} {elapsed:6.2f}s {'OK' if ok else f'FAIL({md:.1e})':>12s}")

print("-" * 62)
print(f"{'TOTAL':20s} {'':5s} {'':5s} {'':7s} {total_time:6.2f}s")
