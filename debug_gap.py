import sys; sys.path.insert(0,'/home/mig20/c3_solution')
from scheduler.planner import import_onnx_graph, decompose_graph
from scheduler.executor import GPUExecutor
import numpy as np, json

g = import_onnx_graph('/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models/resnet_v1.onnx')
plan = decompose_graph(g)

executor = GPUExecutor('/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models/resnet_v1.onnx')

with open('/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35/resnet_v1/input/manifest.json') as f:
    m = json.load(f)

BASE = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35/resnet_v1/input'
data = {e['name']: np.load(f'{BASE}/{e["file"]}') for e in m['tensors']}
executor.load_inputs({k:v[:4] for k,v in data.items()})

# Find the last relu before GAP and GAP kernels
for i, k in enumerate(plan):
    if 'reduce_mean_2d' in k['name']:
        print(f'Running kernels {i-1} to {i}...')
        executor._run(plan[i-1])  # relu before GAP
        executor._run(plan[i])    # GAP
        break
