#!/usr/bin/env python3
"""Test onnx2torch for ResNet with TF32 fully disabled."""
import time, numpy as np, json, os, torch
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)  # pure math, no flash attention

from onnx2torch import convert
import onnx

mp = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models/resnet_v1.onnx'
idir = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35/resnet_v1/input'
gp = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35/resnet_v1/golden/logits.npy'

with open(f'{idir}/manifest.json') as f: m = json.load(f)
data = {e['name']: np.load(f'{idir}/{e["file"]}') for e in m['tensors']}

model = onnx.load(mp)
init_names = {i.name for i in model.graph.initializer}
inames = [i.name for i in model.graph.input if i.name not in init_names]

tm = convert(model).eval().cuda()

# Test full 10k at batch=256
N = data[inames[0]].shape[0]
B = 256

# Warmup batch
batch = {n: torch.from_numpy(data[n][:B]).float().cuda() for n in inames}
with torch.no_grad():
    _ = tm(*[batch[n] for n in inames])

t0 = time.perf_counter()
all_out = []
for s in range(0, N, B):
    e = min(s+B, N)
    feed = [torch.from_numpy(data[n][s:e]).float().cuda() for n in inames]
    with torch.no_grad():
        out = tm(*feed)
    all_out.append(out.cpu().numpy())
t1 = time.perf_counter()

out_all = np.concatenate(all_out, axis=0)
gold = np.load(gp)
ok = np.allclose(out_all, gold, rtol=1e-3, atol=1e-3)
md = np.max(np.abs(out_all - gold))

print(f"onnx2torch ResNet batch=256: {t1-t0:.2f}s precision={'OK' if ok else f'FAIL({md:.1e})'}")
