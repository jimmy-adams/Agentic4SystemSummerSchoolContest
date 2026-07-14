"""BigFormer via onnx2torch — correct weight handling."""
import numpy as np
import torch
import time, os, sys, json

ONNX_PATH = "/workspace/C3/testcases/models/bigformer_v1.onnx"
INPUT_DIR = "/workspace/C3/testcases/testdata/c35/bigformer_v1/input"
GOLDEN_DIR = "/workspace/C3/testcases/testdata/c35/bigformer_v1/golden"
BATCH_SIZE = 16

print("=== Converting ONNX to PyTorch via onnx2torch ===", file=sys.stderr)
t0 = time.perf_counter()

import onnx
# Load model first (handles external data), convert from memory
onnx_model = onnx.load(ONNX_PATH)
from onnx2torch import convert
model = convert(onnx_model)
dt = time.perf_counter() - t0
print(f"  Conversion: {dt:.1f}s", file=sys.stderr)
print(f"  Model type: {type(model).__name__}", file=sys.stderr)

# Move to GPU
print("=== Moving to GPU ===", file=sys.stderr)
try:
    model = model.cuda().float()
    print("  Full GPU load OK", file=sys.stderr)
except RuntimeError as e:
    print(f"  GPU OOM: {e}", file=sys.stderr)
    print("  Keeping on CPU, will stream per batch", file=sys.stderr)
    model = model.float()

# Load input
print("=== Loading input ===", file=sys.stderr)
with open(os.path.join(INPUT_DIR, "manifest.json")) as f:
    manifest = json.load(f)
input_data = np.load(os.path.join(INPUT_DIR, manifest["tensors"][0]["file"]))
input_ids = torch.from_numpy(input_data).long()
N = input_ids.shape[0]

# Run inference
print(f"=== Inference ({N} samples) ===", file=sys.stderr)
all_logits = []
device = next(model.parameters()).device  # get device model is on

for start in range(0, N, BATCH_SIZE):
    end = min(start + BATCH_SIZE, N)
    batch = input_ids[start:end].to(device)
    
    with torch.no_grad():
        out = model(batch)
    
    all_logits.append(out.cpu().numpy())
    
    if start == 0:
        dt = time.perf_counter() - t0
        print(f"  First batch: {dt:.1f}s", file=sys.stderr)

logits = np.concatenate(all_logits, axis=0).astype(np.float32)
dt = time.perf_counter() - t0

if logits.ndim == 4:
    logits = logits.reshape(-1, logits.shape[-2], logits.shape[-1])

golden = np.load(os.path.join(GOLDEN_DIR, "logits.npy"))
ok = np.allclose(logits, golden, rtol=1e-3, atol=1e-3)
md = np.max(np.abs(logits - golden))
print(f"\nTime: {dt:.1f}s | Precision: {'PASS' if ok else 'FAIL'} | MAX_DIFF: {md:.2e}", file=sys.stderr)
print(f"Shape: {logits.shape}", file=sys.stderr)
