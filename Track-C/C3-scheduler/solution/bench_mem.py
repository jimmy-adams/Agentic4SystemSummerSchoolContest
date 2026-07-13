#!/usr/bin/env python3
"""Compare regular vs memory-aware executor on MLP."""
import sys, time, numpy as np, json, torch
sys.path.insert(0, '/home/mig20/c3_solution')

from scheduler.planner import import_onnx_graph, decompose_graph, apply_fusion
from scheduler.executor import GPUExecutor, MemoryAwareExecutor
import onnx

mp = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models/mlp_v1.onnx'
idir = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35/mlp_v1/input'
gp = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35/mlp_v1/golden/logits.npy'

with open(f'{idir}/manifest.json') as f: m = json.load(f)
data = {e['name']: np.load(f'{idir}/{e["file"]}') for e in m['tensors']}
B = 256
batch = {k: v[:B] for k, v in data.items()}
gold = np.load(gp)[:B]

model = onnx.load(mp)
onames = [o.name for o in model.graph.output]
graph = import_onnx_graph(mp)
raw = decompose_graph(graph)
opt, _ = apply_fusion(graph, raw)
plan = opt if len(opt) < len(raw) else raw

# ===== Regular executor =====
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()
exe1 = GPUExecutor(mp); exe1.load_inputs(batch)
t0 = time.perf_counter()
out1 = exe1.execute_plan(plan, onames)[onames[0]]
t1 = time.perf_counter()
mem1 = torch.cuda.max_memory_allocated() / 1024**2
ok1 = np.allclose(out1, gold, rtol=1e-3, atol=1e-3)

# ===== Memory-aware executor =====
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()
exe2 = MemoryAwareExecutor(mp); exe2.load_inputs(batch)
t0 = time.perf_counter()
out2 = exe2.execute_plan(plan, onames)[onames[0]]
t1 = time.perf_counter()
mem2 = torch.cuda.max_memory_allocated() / 1024**2
ok2 = np.allclose(out2, gold, rtol=1e-3, atol=1e-3)

print(f"Regular:     {t1-t0:.3f}s  peak={mem1:.1f}MB  {'OK' if ok1 else 'FAIL'}")
print(f"Mem-aware:   {t1-t0:.3f}s  peak={mem2:.1f}MB  {'OK' if ok2 else 'FAIL'}")
