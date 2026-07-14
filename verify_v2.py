import os, time, json, sys, numpy as np
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

from scheduler.planner import import_onnx_graph, decompose_graph, apply_fusion
from scheduler.executor import MemoryAwareExecutor

mp = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models/resnet_v1.onnx'
d = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35/resnet_v1'

with open(d + '/input/manifest.json') as f:
    m = json.load(f)
data = {e['name']: np.load(d + '/input/' + e['file']) for e in m['tensors']}
N = data[list(data.keys())[0]].shape[0]
B = 64

graph = import_onnx_graph(mp)
raw = decompose_graph(graph)
opt, s = apply_fusion(graph, raw)
onames = [o.name for o in __import__('onnx').load(mp).graph.output]

# Run 1: cold
executor = MemoryAwareExecutor(mp)
t0 = time.perf_counter()
for start in range(0, N, B):
    end = min(start + B, N)
    batch = {k: v[start:end] for k, v in data.items()}
    executor.load_inputs(batch)
    with open(os.devnull, 'w') as f:
        old = sys.stderr
        sys.stderr = f
        executor.execute_plan(raw, onames)
        sys.stderr = old
print(f'Run 1 (cold): {time.perf_counter()-t0:.2f}s')

# Run 2: warm
executor2 = MemoryAwareExecutor(mp)
t0 = time.perf_counter()
for start in range(0, N, B):
    end = min(start + B, N)
    batch = {k: v[start:end] for k, v in data.items()}
    executor2.load_inputs(batch)
    with open(os.devnull, 'w') as f:
        old = sys.stderr
        sys.stderr = f
        executor2.execute_plan(raw, onames)
        sys.stderr = old
print(f'Run 2 (warm): {time.perf_counter()-t0:.2f}s')

# Run 3: no TF32, only benchmark
torch.backends.cuda.matmul.allow_tf32 = False
executor3 = MemoryAwareExecutor(mp)
t0 = time.perf_counter()
for start in range(0, N, B):
    end = min(start + B, N)
    batch = {k: v[start:end] for k, v in data.items()}
    executor3.load_inputs(batch)
    with open(os.devnull, 'w') as f:
        old = sys.stderr
        sys.stderr = f
        executor3.execute_plan(raw, onames)
        sys.stderr = old
print(f'Run 3 (no tf32): {time.perf_counter()-t0:.2f}s')
