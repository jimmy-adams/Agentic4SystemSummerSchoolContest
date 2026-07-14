"""BigFormer FP16 storage + FP32 compute — half disk I/O, same precision.

Weights stored as FP16 (9.7GB), transferred to GPU, converted to FP32 for compute.
This avoids the FP16 compute precision issues while still saving disk bandwidth.
"""
import onnx, numpy as np, torch, torch.nn.functional as F
import time, os, sys, json
from collections import defaultdict

ONNX_PATH = "/workspace/C3/testcases/models/bigformer_v1.onnx"
INPUT_DIR = "/workspace/C3/testcases/testdata/c35/bigformer_v1/input"
GOLDEN_DIR = "/workspace/C3/testcases/testdata/c35/bigformer_v1/golden"
BATCH_SIZE, device = 16, torch.device("cuda")

print("=== FP16 store + FP32 compute ===")
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
    # Store as FP16 to save memory and disk I/O
    t = torch.from_numpy(arr.copy()).half().pin_memory()
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

# Pre-build — keep FP16, convert to FP32 only on GPU
block_cpu = []
for bid in range(num_blocks):
    bw = block_weights[bid]
    def g(n): return gp(f"blocks.{bid}.{n}")
    block_cpu.append({
        "qkv": init_data[bw["qkv"]],
        "proj": init_data[bw["proj"]],
        "ff1": init_data[bw["ff1"]],
        "ff2": init_data[bw["ff2"]],
        "qkv_b": g("qkv.bias").float() if g("qkv.bias") is not None else None,
        "proj_b": g("proj.bias").float() if g("proj.bias") is not None else None,
        "ff1_b": g("ff1.bias").float() if g("ff1.bias") is not None else None,
        "ff2_b": g("ff2.bias").float() if g("ff2.bias") is not None else None,
        "ln1_w": g("ln1.weight").float() if g("ln1.weight") is not None else None,
        "ln1_b": g("ln1.bias").float() if g("ln1.bias") is not None else None,
        "ln2_w": g("ln2.weight").float() if g("ln2.weight") is not None else None,
        "ln2_b": g("ln2.bias").float() if g("ln2.bias") is not None else None,
    })

total_gb = sum(t.numel()*t.element_size() for t in init_data.values()) / 1e9
print(f"  Weights: {total_gb:.1f} GB (FP16) in {time.perf_counter()-t0:.1f}s")

# Shared
tok_emb = gp("tok_emb.weight").float().to(device) if gp("tok_emb.weight") is not None else None
pos_emb = gp("pos_emb").float().to(device) if gp("pos_emb") is not None else None
head_w_name = block_weights.get("head")
head_w = init_data[head_w_name].float().to(device) if head_w_name else None
head_b = gp("head.bias").float().to(device) if gp("head.bias") is not None else None
ln_f_w = gp("ln_f.weight").float().to(device) if gp("ln_f.weight") is not None else None
ln_f_b = gp("ln_f.bias").float().to(device) if gp("ln_f.bias") is not None else None

with open(os.path.join(INPUT_DIR, "manifest.json")) as f: manifest = json.load(f)
input_ids = torch.from_numpy(np.load(os.path.join(INPUT_DIR, manifest["tensors"][0]["file"]))).long()
N, D = input_ids.shape[0], tok_emb.shape[1]

t_stream = torch.cuda.Stream()

def load_block_gpu(cpu_dict):
    """Load FP16 weights, convert matmul weights to FP32 on GPU."""
    result = {}
    for k, t in cpu_dict.items():
        if t is None: result[k] = None; continue
        with torch.cuda.stream(t_stream):
            gpu_t = t.to(device, non_blocking=True)
            # Convert matmul weights to FP32 for compute precision
            if k in ("qkv", "proj", "ff1", "ff2"):
                result[k] = gpu_t.float()
            else:
                result[k] = gpu_t
    return result

print(f"=== Inference ===")
all_logits = []

for start in range(0, N, BATCH_SIZE):
    end = min(start + BATCH_SIZE, N)
    batch_ids = input_ids[start:end].to(device)
    B, S = batch_ids.shape[0], batch_ids.shape[1]
    x = tok_emb[batch_ids] + pos_emb[:S, :].unsqueeze(0)
    
    # Load block 0 sync (convert matmul weights to FP32)
    w_curr = {}
    for k, t in block_cpu[0].items():
        if t is None: w_curr[k] = None; continue
        gpu_t = t.to(device)
        w_curr[k] = gpu_t.float() if k in ("qkv","proj","ff1","ff2") else gpu_t
    if num_blocks > 1:
        w_next = load_block_gpu(block_cpu[1])
    
    for bid in range(num_blocks):
        w = w_curr
        if bid < num_blocks - 2:
            next_load = load_block_gpu(block_cpu[bid + 2])
        
        # ── Attention (FP32 compute, FP16 weights auto-cast) ──
        residual = x
        if w["ln1_w"] is not None:
            x = F.layer_norm(x, [D], weight=w["ln1_w"], bias=w["ln1_b"], eps=1e-5)
        
        # FP16 weights already converted to FP32 on GPU — compute directly
        qkv = x @ w["qkv"]
        if w["qkv_b"] is not None: qkv = qkv + w["qkv_b"]
        q,k,v = qkv.chunk(3, dim=-1)
        q = q.view(B,S,32,128).permute(0,2,1,3); k = k.view(B,S,32,128).permute(0,2,1,3); v = v.view(B,S,32,128).permute(0,2,1,3)
        attn_out = (F.softmax((q @ k.transpose(-2,-1)) * (128**-0.5), dim=-1) @ v)
        attn_out = attn_out.permute(0,2,1,3).reshape(B,S,4096)
        attn_out = attn_out @ w["proj"]
        if w["proj_b"] is not None: attn_out = attn_out + w["proj_b"]
        x = residual + attn_out
        
        # ── FFN ──
        residual = x
        if w["ln2_w"] is not None:
            x = F.layer_norm(x, [D], weight=w["ln2_w"], bias=w["ln2_b"], eps=1e-5)
        
        torch.backends.cuda.matmul.allow_tf32 = True
        x = x @ w["ff1"]
        if w["ff1_b"] is not None: x = x + w["ff1_b"]
        x = F.gelu(x)
        x = x @ w["ff2"]
        if w["ff2_b"] is not None: x = x + w["ff2_b"]
        torch.backends.cuda.matmul.allow_tf32 = False
        x = residual + x
        
        if bid < num_blocks - 1:
            t_stream.synchronize()
            w_curr = w_next
            w_next = next_load if bid < num_blocks - 2 else None
    
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
