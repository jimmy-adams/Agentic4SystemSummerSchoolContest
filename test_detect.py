import sys, os
sys.path.insert(0, '/home/mig20/c3_solution')
from infer import detect_model_type

BASE = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models'
for name in ['mlp_v1', 'resnet_v1', 'transformer_v1']:
    mp = os.path.join(BASE, f'{name}.onnx')
    dt = detect_model_type(mp)
    print(f'{name:20s} -> {dt}')
