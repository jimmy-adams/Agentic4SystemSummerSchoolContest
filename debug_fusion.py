import sys
sys.path.insert(0, '/home/mig20/c3_solution')
from scheduler.planner import import_onnx_graph, decompose_graph, apply_fusion_and_rebuild

g = import_onnx_graph('/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models/mlp_v1.onnx')

print("=== Raw nodes ===")
for n in g.nodes:
    print(f"  {n['name']}: {n['op_type']} in={n['inputs']} out={n['outputs']}")

print("\n=== Fusion result ===")
plan, stats = apply_fusion_and_rebuild(g)
print(f"Fused plan: {len(plan)} kernels")
for k in plan:
    print(f"  {k['name']}: in={k['inputs']} -> out={k['outputs']}")
