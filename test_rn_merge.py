#!/usr/bin/env python3
"""Test ResNet kernel-level merge (no graph fusion)."""
import sys, time, numpy as np, json, onnx
sys.path.insert(0, '/home/mig20/c3_solution')
from scheduler.planner import import_onnx_graph, decompose_graph
from scheduler.executor import GPUExecutor

mp='/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models/resnet_v1.onnx'
idir='/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35/resnet_v1/input'
gp='/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35/resnet_v1/golden/logits.npy'

with open(idir+'/manifest.json') as f: m=json.load(f)
data={e['name']:np.load(idir+'/'+e['file']) for e in m['tensors']}
B=64; batch={k:v[:B] for k,v in data.items()}
model=onnx.load(mp); onames=[o.name for o in model.graph.output]

graph=import_onnx_graph(mp)
raw=decompose_graph(graph)
print(f"Raw: {len(raw)} kernels")

exe=GPUExecutor(mp); exe.load_inputs(batch)
_=exe.execute_plan(raw, onames)  # warmup

t0=time.perf_counter()
out=exe.execute_plan(raw, onames)[onames[0]]
t1=time.perf_counter()

gold=np.load(gp)[:B]
ok=np.allclose(out, gold, 1e-3, 1e-3)
md=np.max(np.abs(out-gold))
print(f"Merged: {exe.launch_count} kernels, {t1-t0:.3f}s, OK={ok} max={md:.1e}")
