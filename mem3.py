import sys, numpy as np, json
sys.path.insert(0, '/home/mig20/c3_solution')
from scheduler.planner import import_onnx_graph, decompose_graph
from scheduler.executor import MemoryAwareExecutor

for name in ['mlp_v1', 'resnet_v1', 'transformer_v1']:
    mp = f'/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models/{name}.onnx'
    idir = f'/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35/{name}/input'
    with open(f'{idir}/manifest.json') as f: m=json.load(f)
    data={e['name']:np.load(f'{idir}/{e["file"]}') for e in m['tensors']}
    B=64 if 'resnet' in name else 256
    batch={k:v[:B] for k,v in data.items()}
    graph=import_onnx_graph(mp)
    raw=decompose_graph(graph)
    plan=raw  # raw plan — skip fusion to avoid graph state issues
    exe=MemoryAwareExecutor(mp); exe.load_inputs(batch)
    _=exe.execute_plan(plan, ['logits'])
