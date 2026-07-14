# Quick test: raw pipeline execution (no fusion) on MLP
import sys, time, numpy as np, json
sys.path.insert(0, '/home/mig20/c3_solution')

from scheduler.planner import import_onnx_graph, decompose_graph
from scheduler.executor import GPUExecutor
import onnx

MODEL = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models/mlp_v1.onnx'
INPUT = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35/mlp_v1/input'
GOLDEN = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35/mlp_v1/golden/logits.npy'

# Load inputs
with open(INPUT+'/manifest.json') as f: m = json.load(f)
data = {e['name']: np.load(INPUT+'/'+e['file']) for e in m['tensors']}

# Parse + decompose
model = onnx.load(MODEL)
graph = import_onnx_graph(MODEL)
plan = decompose_graph(graph)
print(len(plan), 'kernels:')
for k in plan:
    print(f"  {k['name']}: {k['inputs'][:2]} -> {k['outputs'][:1]}")

# Execute first 8 samples
executor = GPUExecutor(MODEL)
batch_data = {k: v[:8] for k, v in data.items()}
executor.load_inputs(batch_data)
output_names = [o.name for o in model.graph.output]

t0 = time.perf_counter()
results = executor.execute_plan(plan, output_names)
t1 = time.perf_counter()

out = results[output_names[0]]
gold = np.load(GOLDEN)[:8]
ok = np.allclose(out, gold, rtol=1e-3, atol=1e-3)
print(f"Precision: {'OK' if ok else 'FAIL'} max_diff={np.max(np.abs(out-gold)):.2e}")
print(f"Time: {t1-t0:.4f}s ({len(plan)} kernels, 8 samples)")
