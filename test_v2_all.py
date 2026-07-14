#!/usr/bin/env python3
"""v2 raw pipeline test (no fusion) on all 3 models."""
import sys, time, numpy as np, json, subprocess
sys.path.insert(0, '/home/mig20/c3_solution')
from scheduler.planner import import_onnx_graph, decompose_graph
from scheduler.executor import GPUExecutor
import onnx

MODEL_BASE = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors'
DATA_BASE = f'{MODEL_BASE}/testdata/c35'

for name in ['mlp_v1', 'resnet_v1', 'transformer_v1']:
    model_path = f'{MODEL_BASE}/models/{name}.onnx'
    input_dir = f'{DATA_BASE}/{name}/input'
    golden_path = f'{DATA_BASE}/{name}/golden/logits.npy'

    # Load inputs
    with open(f'{input_dir}/manifest.json') as f: m = json.load(f)
    data = {e['name']: np.load(f'{input_dir}/{e["file"]}') for e in m['tensors']}

    # Parse + decompose (NO fusion)
    graph = import_onnx_graph(model_path)
    plan = decompose_graph(graph)

    # Execute (first 8 samples for quick test)
    executor = GPUExecutor(model_path)
    batch = {k: v[:8] for k, v in data.items()}
    executor.load_inputs(batch)
    model = onnx.load(model_path)
    output_names = [o.name for o in model.graph.output]

    try:
        t0 = time.perf_counter()
        results = executor.execute_plan(plan, output_names)
        t1 = time.perf_counter()

        out = results[output_names[0]]
        gold = np.load(golden_path)[:8]
        ok = np.allclose(out, gold, rtol=1e-3, atol=1e-3)
        print(f'{name:20s}: {len(plan)} kernels, {t1-t0:.4f}s, '
              f'precision={"OK" if ok else f"FAIL({np.max(np.abs(out-gold)):.2e})"}')
    except Exception as e:
        print(f'{name:20s}: {len(plan)} kernels, ERROR: {e}')
