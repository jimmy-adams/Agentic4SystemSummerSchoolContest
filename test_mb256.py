import os, sys, time, numpy as np, json
os.environ["C3_TF32_BLOCKS"] = "23"  # 23 blocks TF32
sys.path.insert(0, "/home/mig20/c3_solution")
from bigformer_streaming import BigFormerStreamingExecutor

# micro_batch=64 for TF32 safety
orig = BigFormerStreamingExecutor._micro_batch_size
def new_mb(self, req, sc, sl, fw):
    return min(req, sc, 64)
BigFormerStreamingExecutor._micro_batch_size = new_mb

executor = BigFormerStreamingExecutor("/workspace/C3/testcases/models/bigformer_v1.onnx")
input_dir = "/workspace/C3/testcases/testdata/c35/bigformer_v1/input"
with open(os.path.join(input_dir, "manifest.json")) as f: m=json.load(f)
inputs = {e["name"]: np.load(os.path.join(input_dir, e["file"])) for e in m["tensors"]}
t0=time.perf_counter(); out=executor.run(inputs, 256); dt=time.perf_counter()-t0
gold=np.load("/workspace/C3/testcases/testdata/c35/bigformer_v1/golden/logits.npy")
ok=np.allclose(out["logits"],gold,rtol=1e-3,atol=1e-3)
print(f"TF32=23 micro=64: {dt:.1f}s PASS={ok} MAX_DIFF={np.max(np.abs(out['logits']-gold)):.2e}")
