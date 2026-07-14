"""BigFormer: Preload ALL weights to GPU as FP16 (9.7GB fits in 16GB).

Convert to FP32 on-the-fly during compute. Eliminates per-block transfer entirely.
"""
import onnx, numpy as np, torch, torch.nn.functional as F
import time, os, sys, json
from collections import defaultdict

ONNX_PATH = "/workspace/C3/testcases/models/bigformer_v1.onnx"
INPUT_DIR = "/workspace/C3/testcases/testdata/c35/bigformer_v1/input"
GOLDEN_DIR = "/workspace/C3/testcases/testdata/c35/bigformer_v1/golden"
BATCH_SIZE, device = 16, torch.device("cuda")

print("=== GPU Preload (FP16 weights, all on GPU) ===")
t0 = time.perf_counter()

model = onnx.load(ONNX_PATH)
identity_map = {n.output[0]: n.input[0] for n in model.graph.node if n.op_type == "Identity"}
def resolve(name):
    visited = set()
    while name in identity_map and name not in visited:
        visited.add(name); name = identity_map[name]
    return name

init_data, init_set = {}, set()
for init in model.graph.initializer:
    arr = onnx.numpy_helper.to_array(init)
    t = torch.from_numpy(arr.copy()).half()
    init_data[init.name] = t
    init_set.add(init.name)

def gp(name):
    r = resolve(name)
    return init_data.get(r) if r in init_set else None

# Block mapping
block_weights = {}
for node in model.graph.node:
    if node.op_type != "MatMul": continue
    parts = node.name.split("/")
    if len(parts) < 3 or not parts[1].startswith("blocks."):
        if "head" in node.name:
            for inp in node.input:
                r = resolve(inp)
                if r in init_set: block_weights["head"] = r
        continue
    bid, sub = int(parts[1].split(".")[1]), parts[2]
    if bid not in block_weights: block_weights[bid] = {}
    for inp in node.input:
        r = resolve(inp)
        if r in init_set: block_weights[bid][sub] = r; break

num_blocks = 24

# Pre-load ALL weights to GPU as FP16 (9.7GB total)
print(f"  Loading all weights to GPU...")
blocks_gpu = []
for bid in range(num_blocks):
    bw = block_weights[bid]
    def g(n):
        t = gp(f"blocks.{bid}.{n}")
        return t.to(device) if t is not None else None
    
    w = {
        "qkv": init_data[bw["qkv"]].to(device),
        "proj": init_data[bw["proj"]].to(device),
        "ff1": init_data[bw["ff1"]].to(device),
        "ff2": init_data[bw["ff2"]].to(device),
        "qkv_b": g("qkv.bias"), "proj_b": g("proj.bias"),
        "ff1_b": g("ff1.bias"), "ff2_b": g("ff2.bias"),
        "ln1_w": g("ln1.weight"), "ln1_b": g("ln1.bias"),
        "ln2_w": g("ln2.weight"), "ln2_b": g("ln2.bias"),
    }
    blocks_gpu.append(w)

print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

# Shared
tok_emb = gp("tok_emb.weight").float().to(device) if gp("tok_emb.weight") is not None else None
pos_emb = gp("pos_emb").float().to(device) if gp("pos_emb") is not None else None
head_w = init_data[block_weights["head"]].to(device) if "head" in block_weights else None
head_b = gp("head.bias").float().to(device) if gp("head.bias") is not None else None
ln_f_w = gp("ln_f.weight").float().to(device) if gp("ln_f.weight") is not None else None
ln_f_b = gp("ln_f.bias").float().to(device) if gp("ln_f.bias") is not None else None

print(f"  Total GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")
print(f"  Load time: {time.perf_counter()-t0:.1f}s")

with open(os.path.join(INPUT_DIR, "manifest.json")) as f: manifest = json.load(f)
input_ids = torch.from_numpy(np.load(os.path.join(INPUT_DIR, manifest["tensors"][0]["file"]))).long()
N, D = input_ids.shape[0], tok_emb.shape[1]

print(f"=== Inference (no per-block transfer) ===")
all_logits = []

for start in range(0, N, BATCH_SIZE):
    end = min(start + BATCH_SIZE, N)
    batch_ids = input_ids[start:end].to(device)
    B, S = batch_ids.shape[0], batch_ids.shape[1]
    x = tok_emb[batch_ids] + pos_emb[:S, :].unsqueeze(0)
    
    for bid in range(num_blocks):
        w = blocks_gpu[bid]
        
        # Attention — convert FP16→FP32 inline
        residual = x.float()
        if w["ln1_w"] is not None:
            x = F.layer_norm(x.float(), [D], weight=w["ln1_w"].float(), 
                           bias=w["ln1_b"].float() if w["ln1_b"] is not None else None, eps=1e-5)
        
        qkv = x @ w["qkv"].float()
        if w["qkv_b"] is not None: qkv = qkv + w["qkv_b"].float()
        q,k,v = qkv.chunk(3, dim=-1)
        q = q.view(B,S,32,128).permute(0,2,1,3); k = k.view(B,S,32,128).permute(0,2,1,3); v = v.view(B,S,32,128).permute(0,2,1,3)
        attn_out = (F.softmax((q @ k.transpose(-2,-1)) * (128**-0.5), dim=-1) @ v)
        attn_out = attn_out.permute(0,2,1,3).reshape(B,S,4096)
        attn_out = attn_out @ w["proj"].float()
        if w["proj_b"] is not None: attn_out = attn_out + w["proj_b"].float()
        x = residual + attn_out
        
        # FFN
        residual = x.float()
        if w["ln2_w"] is not None:
            x = F.layer_norm(x.float(), [D], weight=w["ln2_w"].float(),
                           bias=w["ln2_b"].float() if w["ln2_b"] is not None else None, eps=1e-5)
        
        torch.backends.cuda.matmul.allow_tf32 = True
        x = x @ w["ff1"].float()
        if w["ff1_b"] is not None: x = x + w["ff1_b"].float()
        x = F.gelu(x)
        x = x @ w["ff2"].float()
        if w["ff2_b"] is not None: x = x + w["ff2_b"].float()
        torch.backends.cuda.matmul.allow_tf32 = False
        x = residual + x
    
    if ln_f_w is not None:
        x = F.layer_norm(x.float(), [D], weight=ln_f_w, bias=ln_f_b, eps=1e-5)
    if head_w is not None:
        x = x.float() @ head_w.float()
        if head_b is not None: x = x + head_b
    all_logits.append(x.cpu().numpy())
    if start == 0: print(f"  First batch: {time.perf_counter()-t0:.1f}s")

logits = np.concatenate(all_logits, axis=0).astype(np.float32)
if logits.ndim == 4: logits = logits.reshape(-1, logits.shape[-2], logits.shape[-1])
dt = time.perf_counter() - t0
golden = np.load(os.path.join(GOLDEN_DIR, "logits.npy"))
ok = np.allclose(logits, golden, rtol=1e-3, atol=1e-3)
print(f"\nTime: {dt:.1f}s | Precision: {'PASS' if ok else 'FAIL'} | MAX_DIFF: {np.max(np.abs(logits-golden)):.2e}")
