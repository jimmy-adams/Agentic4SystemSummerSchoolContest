import numpy as np, json, os

BASE = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35'
OUT = {'mlp_v1': '/tmp/test_out_mlp', 'resnet_v1': '/tmp/test_out_rn2', 'transformer_v1': '/tmp/test_out_tf'}

for name, out_dir in OUT.items():
    golden = np.load(f'{BASE}/{name}/golden/logits.npy')
    with open(f'{out_dir}/manifest.json') as f:
        m = json.load(f)
    out_name = m['tensors'][0]['name']
    out = np.load(f'{out_dir}/{out_name}.npy')
    ok = np.allclose(out, golden, rtol=1e-3, atol=1e-3)
    md = np.max(np.abs(out - golden))
    print(f'{name:20s} max_diff={md:.2e}  prec={"OK" if ok else "FAIL"}')
