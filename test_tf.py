import sys; sys.path.insert(0,'/home/mig20/c3_solution')
from scheduler.planner import import_onnx_graph, decompose_graph
from scheduler.executor import GPUExecutor
import numpy as np, json, onnx

MODEL = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models/transformer_v1.onnx'
INPUT = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35/transformer_v1/input'
GOLDEN = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35/transformer_v1/golden/logits.npy'

g = import_onnx_graph(MODEL)
plan = decompose_graph(g)
exe = GPUExecutor(MODEL)

with open(INPUT+'/manifest.json') as f: m=json.load(f)
data={e['name']:np.load(INPUT+'/'+e['file']) for e in m['tensors']}
exe.load_inputs({k:v[:4] for k,v in data.items()})

err = False
for i,k in enumerate(plan):
    try:
        exe._run(k)
        if i >= len(plan)-3 or i <= 15:
            print(f"[{i}] {k['name']}: OK")
    except Exception as e:
        print(f"[{i}] {k['name']}: {e}")
        err = True
        break

if not err:
    model = onnx.load(MODEL)
    out = exe._reg[model.graph.output[0].name].cpu().numpy()
    gold = np.load(GOLDEN)[:4]
    ok = np.allclose(out, gold, rtol=1e-3, atol=1e-3)
    print(f"Precision: {'OK' if ok else 'FAIL'}")
