"""Optimized BigFormer: cache weights as .pt for fast reload."""
import onnx, time, sys, torch, numpy as np, os
from onnx.numpy_helper import to_array

ONNX_PATH = "/workspace/C3/testcases/models/bigformer_v1.onnx"
CACHE_PATH = "/tmp/bigformer_weights.pt"

print("=== Weight loading (with cache) ===", file=sys.stderr)
t0 = time.perf_counter()

if os.path.exists(CACHE_PATH):
    # Fast path: load from torch cache
    init_data = torch.load(CACHE_PATH, weights_only=True)
    dt = time.perf_counter() - t0
    print(f"  Loaded from cache: {dt:.1f}s", file=sys.stderr)
else:
    # Slow path: parse ONNX, build cache
    model = onnx.load(ONNX_PATH)
    init_data = {}
    for init in model.graph.initializer:
        arr = to_array(init)
        # Convert to FP32 tensor (pinned not needed, we'll pin on first use)
        init_data[init.name] = torch.from_numpy(arr.copy()).pin_memory()
    
    dt = time.perf_counter() - t0
    print(f"  Loaded from ONNX: {dt:.1f}s", file=sys.stderr)
    
    # Save cache for next time
    t0 = time.perf_counter()
    torch.save(init_data, CACHE_PATH)
    print(f"  Cache saved: {time.perf_counter()-t0:.1f}s", file=sys.stderr)

total_gb = sum(t.numel() * t.element_size() for t in init_data.values()) / 1e9
print(f"  Total: {total_gb:.1f} GB", file=sys.stderr)

# Quick sanity: list first few keys
for i, k in enumerate(init_data):
    if i < 5:
        print(f"  {k}: {list(init_data[k].shape)}")
