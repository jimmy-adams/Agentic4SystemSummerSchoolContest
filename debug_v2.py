#!/usr/bin/env python3
"""Interactive debug for v2 pipeline — run directly on server:
   python3 debug_v2.py
"""
import sys, numpy as np, json
sys.path.insert(0, '/home/mig20/c3_solution')

from scheduler.planner import import_onnx_graph, decompose_graph
from scheduler.executor import GPUExecutor
import torch, onnx

MODEL_BASE = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models'
DATA_BASE  = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35'

def debug_model(name, fix_transformer=False):
    print(f"\n{'='*60}")
    print(f" Debug: {name}")
    print(f"{'='*60}")

    model_path = f'{MODEL_BASE}/{name}.onnx'
    input_dir  = f'{DATA_BASE}/{name}/input'

    # Load inputs (batch=4 for speed)
    with open(f'{input_dir}/manifest.json') as f:
        m = json.load(f)
    data = {e['name']: np.load(f'{input_dir}/{e["file"]}') for e in m['tensors']}

    # Parse + decompose
    graph = import_onnx_graph(model_path)
    plan = decompose_graph(graph)

    # Show plan highlights
    ops = {}
    for k in plan:
        base = k['name'].split('_')[0]
        ops[base] = ops.get(base, 0) + 1
    print(f"Raw plan: {len(plan)} kernels, ops: {ops}")

    # Show last 5 kernels
    print("Last 5 kernels:")
    for k in plan[-5:]:
        print(f"  {k['name']}: in={k['inputs'][:2]} -> out={k['outputs'][:1]}")

    # Execute with shape logging
    executor = GPUExecutor(model_path)
    batch = {k: v[:4] for k, v in data.items()}
    executor.load_inputs(batch)

    model = onnx.load(model_path)
    output_names = [o.name for o in model.graph.output]

    print(f"\nStep-by-step execution:")
    errors = []
    for i, k in enumerate(plan):
        try:
            # Log input shapes before execution
            in_shapes = {}
            for inp_name in k['inputs']:
                if inp_name in executor._reg:
                    in_shapes[inp_name] = list(executor._reg[inp_name].shape)
                else:
                    in_shapes[inp_name] = 'MISSING'

            executor._run(k)

            # Log output shapes
            out_shapes = {}
            for out_name in k['outputs']:
                if out_name in executor._reg:
                    out_shapes[out_name] = list(executor._reg[out_name].shape)

            if i >= len(plan) - 5:  # Last 5 kernels
                print(f"  [{i}] {k['name']}: in={in_shapes} -> out={out_shapes}")

        except Exception as e:
            errors.append((i, k['name'], str(e), in_shapes))
            if i >= len(plan) - 10:
                print(f"  [{i}] {k['name']}: ERROR — {e}")
                print(f"       inputs: {in_shapes}")
            break

    if errors:
        i, kn, err, shapes = errors[0]
        print(f"\n💥 First error at kernel [{i}] {kn}:")
        print(f"   {err}")
        print(f"   input shapes: {shapes}")
    else:
        # Verify precision
        results = {n: executor._reg[n].cpu().numpy() for n in output_names}
        gold_path = f'{DATA_BASE}/{name}/golden/logits.npy'
        out = results[output_names[0]]
        gold = np.load(gold_path)[:4]
        ok = np.allclose(out, gold, rtol=1e-3, atol=1e-3)
        print(f"\n✅ All {len(plan)} kernels passed! precision={'OK' if ok else f'FAIL({np.max(np.abs(out-gold)):.2e})'}")

    return len(errors) == 0

if __name__ == '__main__':
    for m in ['mlp_v1', 'resnet_v1', 'transformer_v1']:
        ok = debug_model(m)
        if not ok:
            print(f"\n⏸  Stopped at {m} — fix issue before continuing")
            print(f"   (comment out the failing model to test the next one)")
