#!/usr/bin/env python3
"""Test onnx2torch on Transformer with TF32 fully disabled."""
import sys, time, numpy as np, json, os, torch
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from onnx2torch import convert
import onnx

mp = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models/transformer_v1.onnx'
idir = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35/transformer_v1/input'
gp = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35/transformer_v1/golden/logits.npy'

with open(f'{idir}/manifest.json') as f: m = json.load(f)
data = {e['name']: np.load(f'{idir}/{e["file"]}') for e in m['tensors']}

model = onnx.load(mp)
init_names = {i.name for i in model.graph.initializer}
inames = [i.name for i in model.graph.input if i.name not in init_names]
onames = [o.name for o in model.graph.output]

torch_model = convert(model).eval().cuda()

# Test batch=4
N = 4
feed = []
for n in inames:
    t = torch.from_numpy(data[n][:N])
    t = t.long() if data[n].dtype == np.int64 else t.float()
    feed.append(t.cuda())

# Warmup
with torch.no_grad():
    _ = torch_model(*feed)

# Time
t0 = time.perf_counter()
with torch.no_grad():
    out = torch_model(*feed)
t1 = time.perf_counter()

gold = np.load(gp)[:N]
ok = np.allclose(out.cpu().numpy(), gold, rtol=1e-3, atol=1e-3)
md = np.max(np.abs(out.cpu().numpy() - gold))

print(f"onnx2torch Transformer: {t1-t0:.3f}s for {N} samples, "
      f"precision={'OK' if ok else f'FAIL({md:.1e})'}")
