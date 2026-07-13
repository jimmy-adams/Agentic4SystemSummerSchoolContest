#!/usr/bin/env python3  
"""Final benchmark: scheduler stats + ORT timing."""
import sys, time, numpy as np, json, subprocess, os

MODEL_BASE = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models'
DATA_BASE = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35'

# Scheduler stats (offline, not timed)
sys.path.insert(0, '/home/mig20/c3_solution')
from scheduler.planner import import_onnx_graph, decompose_graph, apply_fusion

print(f"{'Model':20s} {'raw':>5s} {'opt':>5s} {'reduce':>7s} {'ort_time':>9s} {'precision':>12s}")
print("-" * 62)

total = 0
for name in ['mlp_v1', 'resnet_v1', 'transformer_v1']:
    mp = f'{MODEL_BASE}/{name}.onnx'
    inp = f'{DATA_BASE}/{name}/input'
    gold = f'{DATA_BASE}/{name}/golden/logits.npy'

    # Scheduler stats
    g = import_onnx_graph(mp)
    raw = decompose_graph(g)
    opt, _ = apply_fusion(g, raw)
    red = f"{(1-len(opt)/max(len(raw),1))*100:.0f}%"

    # ORT timing (separate process, nvidia-smi clean)
    t0 = time.perf_counter()
    r = subprocess.run(
        ['bash', '/home/mig20/c3_solution/run_infer.sh', '--onnx', mp,
         '--input', inp, '--output', f'/tmp/final_{name}', '--batch-size', '2048'],
        capture_output=True, text=True, timeout=180,
        env={**os.environ, 'NVIDIA_TF32_OVERRIDE': '0',
             'LD_LIBRARY_PATH': '/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:'
                               '/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:'
                               '/usr/local/cuda/targets/x86_64-linux/lib:' 
                               + os.environ.get('LD_LIBRARY_PATH', '')})
    t1 = time.perf_counter()
    et = t1 - t0
    total += et

    out = np.load(f'/tmp/final_{name}/logits.npy')
    gd = np.load(gold)
    ok = np.allclose(out, gd, rtol=1e-3, atol=1e-3)
    md = np.max(np.abs(out - gd))

    print(f"{name:20s} {len(raw):5d} {len(opt):5d} {red:>7s} {et:8.2f}s {'OK' if ok else f'FAIL({md:.1e})':>12s}")

print("-" * 62)
print(f"{'TOTAL':20s} {'':5s} {'':5s} {'':7s} {total:8.2f}s")
