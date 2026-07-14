"""Profile BigFormer weight loading bottlenecks."""
import onnx, time, sys, torch, numpy as np
from onnx.numpy_helper import to_array

PATH = "/workspace/C3/testcases/models/bigformer_v1.onnx"

print("=== Profiling weight loading ===", file=sys.stderr)

# 1. ONNX protobuf parsing
t0 = time.perf_counter()
model = onnx.load(PATH)
t1 = time.perf_counter()
print(f"  onnx.load: {t1-t0:.1f}s", file=sys.stderr)

# 2. to_array (reads from .data file)
t0 = time.perf_counter()
total = 0
for init in model.graph.initializer:
    arr = to_array(init)
    total += arr.nbytes
t1 = time.perf_counter()
print(f"  to_array (read {total/1e9:.1f}GB): {t1-t0:.1f}s", file=sys.stderr)

# 3. torch.from_numpy + .copy()
t0 = time.perf_counter()
for init in model.graph.initializer:
    arr = to_array(init)
    t = torch.from_numpy(arr.copy())
t1 = time.perf_counter()
print(f"  + torch copy: {t1-t0:.1f}s", file=sys.stderr)

# 4. torch.from_numpy + .pin_memory()
t0 = time.perf_counter()
for init in model.graph.initializer:
    arr = to_array(init)
    t = torch.from_numpy(arr).pin_memory()
t1 = time.perf_counter()
print(f"  + pin_memory: {t1-t0:.1f}s", file=sys.stderr)

# 5. Direct torch loading (no numpy intermediate)
t0 = time.perf_counter()
for init in model.graph.initializer:
    arr = to_array(init)
    t = torch.from_numpy(arr).pin_memory()
t1 = time.perf_counter()
print(f"  from_numpy + pin: {t1-t0:.1f}s (duplicate of above)", file=sys.stderr)

# 6. torch.save / torch.load roundtrip
print(f"\n=== Torch save/load test ===", file=sys.stderr)
test_dict = {}
for i, init in enumerate(model.graph.initializer[:10]):  # first 10 for test
    arr = to_array(init)
    test_dict[init.name] = torch.from_numpy(arr)

t0 = time.perf_counter()
torch.save(test_dict, "/tmp/bf_test.pt")
t1 = time.perf_counter()
print(f"  torch.save (10 tensors): {t1-t0:.1f}s", file=sys.stderr)

t0 = time.perf_counter()
loaded = torch.load("/tmp/bf_test.pt", weights_only=True)
t1 = time.perf_counter()
print(f"  torch.load: {t1-t0:.1f}s", file=sys.stderr)
