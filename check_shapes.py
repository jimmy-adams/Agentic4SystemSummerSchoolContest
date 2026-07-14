import numpy as np, os

BASE = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35'
for name in ['mlp_v1', 'resnet_v1', 'transformer_v1']:
    print(f'\n=== {name} ===')
    for sub in ['input', 'golden']:
        d = f'{BASE}/{name}/{sub}'
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith('.npy'):
                    a = np.load(f'{d}/{f}')
                    print(f'  {sub}/{f}: shape={a.shape} dtype={a.dtype}')
    lp = f'{BASE}/{name}/labels.npy'
    if os.path.exists(lp):
        a = np.load(lp)
        print(f'  labels.npy: shape={a.shape} dtype={a.dtype}')
