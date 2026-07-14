"""BigFormer GPU streaming with pipeline: preload next block async."""
import onnx
from onnx.numpy_helper import to_array
import numpy as np
import torch
import torch.nn.functional as F
import time, os, sys, json, gc
from collections import defaultdict

# Enable TF32 tensor cores for ~2x matmul throughput
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True

ONNX_PATH = "/workspace/C3/testcases/models/bigformer_v1.onnx"
INPUT_DIR = "/workspace/C3/testcases/testdata/c35/bigformer_v1/input"
GOLDEN_DIR = "/workspace/C3/testcases/testdata/c35/bigformer_v1/golden"
CACHE_PATH = "/tmp/bigformer_weights.pt"
BATCH_SIZE = 16
device = torch.device("cuda")

print("=== Parse & load ===", file=sys.stderr)
t0 = time.perf_counter()

# ── Fast path: torch cache ──
if os.path.exists(CACHE_PATH):
    init_data = torch.load(CACHE_PATH, weights_only=True)
    # Re-pin memory (torch.save doesn't preserve pinning)
    for k in list(init_data.keys()):
        init_data[k] = init_data[k].pin_memory()
    print(f"  Cache load: {time.perf_counter()-t0:.1f}s", file=sys.stderr)
else:
    # ── Slow path: ONNX parse ──
    model = onnx.load(ONNX_PATH)
    init_data = {}
    for init in model.graph.initializer:
        arr = to_array(init)
        init_data[init.name] = torch.from_numpy(arr.copy()).pin_memory()
    dt = time.perf_counter() - t0
    print(f"  ONNX load: {dt:.1f}s", file=sys.stderr)
    
    # Save cache for next time
    torch.save(init_data, CACHE_PATH)
    print(f"  Cache saved", file=sys.stderr)

# ── Build Identity resolution (needs model.graph) ──
if 'model' not in dir():
    model = onnx.load(ONNX_PATH)

# Identity resolution
identity_map = {}
for node in model.graph.node:
    if node.op_type == "Identity":
        identity_map[node.output[0]] = node.input[0]

def resolve(name):
    visited = set()
    while name in identity_map and name not in visited:
        visited.add(name)
        name = identity_map[name]
    return name

init_data = {}
init_set = set()
for init in model.graph.initializer:
    arr = to_array(init)
    # Pin memory for async transfer
    t = torch.from_numpy(arr.copy()).pin_memory()
    init_data[init.name] = t
    init_set.add(init.name)

def get_param(name):
    r = resolve(name)
    return init_data.get(r) if r in init_set else None

print(f"  {sum(t.numel()*t.element_size() for t in init_data.values())/1e9:.1f} GB pinned", file=sys.stderr)

# Block weight mapping
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
    bid = int(parts[1].split(".")[1])
    sub = parts[2]
    if bid not in block_weights: block_weights[bid] = {}
    for inp in node.input:
        r = resolve(inp)
        if r in init_set:
            block_weights[bid][sub] = r
            break

num_blocks = 24

# Shared params (keep on GPU)
tok_emb = get_param("tok_emb.weight")
pos_emb = get_param("pos_emb")
head_w = init_data[block_weights["head"]].float() if "head" in block_weights else None
head_b = get_param("head.bias")

if tok_emb is not None: tok_emb = tok_emb.float().to(device)
if pos_emb is not None: pos_emb = pos_emb.float().to(device)
if head_w is not None: head_w = head_w.to(device)
if head_b is not None: head_b = head_b.float().to(device)

# Input
with open(os.path.join(INPUT_DIR, "manifest.json")) as f:
    manifest = json.load(f)
input_data = np.load(os.path.join(INPUT_DIR, manifest["tensors"][0]["file"]))
input_ids = torch.from_numpy(input_data).long()
N = input_ids.shape[0]
D = tok_emb.shape[1]

def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / 1.41421356237))

print(f"=== Inference (pipelined) ===", file=sys.stderr)

# Pre-build per-block weight dicts (CPU pinned references only)
block_w_cpu = []
for bid in range(num_blocks):
    bw = block_weights[bid]
    block_w_cpu.append({
        "qkv": init_data[bw["qkv"]].float(),
        "proj": init_data[bw["proj"]].float(),
        "ff1": init_data[bw["ff1"]].float(),
        "ff2": init_data[bw["ff2"]].float(),
        "ln1_w": get_param(f"blocks.{bid}.ln1.weight"),
        "ln1_b": get_param(f"blocks.{bid}.ln1.bias"),
        "ln2_w": get_param(f"blocks.{bid}.ln2.weight"),
        "ln2_b": get_param(f"blocks.{bid}.ln2.bias"),
    })

all_logits = []

for start in range(0, N, BATCH_SIZE):
    end = min(start + BATCH_SIZE, N)
    batch_ids = input_ids[start:end].to(device)
    B = batch_ids.shape[0]
    S = batch_ids.shape[1]
    
    # Embeddings
    x = tok_emb[batch_ids]
    x = x + pos_emb[:S, :].unsqueeze(0)
    
    # ── Start pre-loading block 0 weights ──
    w_gpu = {}
    for key in ["qkv", "proj", "ff1", "ff2"]:
        w_gpu[key] = block_w_cpu[0][key].to(device, non_blocking=True)
    for key in ["ln1_w", "ln1_b", "ln2_w", "ln2_b"]:
        t = block_w_cpu[0][key]
        w_gpu[key] = t.to(device, non_blocking=True) if t is not None else None
    torch.cuda.synchronize()  # wait for first block
    
    for bid in range(num_blocks):
        w = w_gpu
        
        # ── Start async pre-load for next block ──
        if bid < num_blocks - 1:
            w_gpu = {}
            for key in ["qkv", "proj", "ff1", "ff2"]:
                w_gpu[key] = block_w_cpu[bid+1][key].to(device, non_blocking=True)
            for key in ["ln1_w", "ln1_b", "ln2_w", "ln2_b"]:
                t = block_w_cpu[bid+1][key]
                w_gpu[key] = t.to(device, non_blocking=True) if t is not None else None
        
        # ── Compute current block ──
        # LN1
        residual = x
        if w["ln1_w"] is not None:
            x = F.layer_norm(x.float(), [D], weight=w["ln1_w"].float(), 
                           bias=w["ln1_b"].float() if w["ln1_b"] is not None else None, eps=1e-5).to(x.dtype)
        
        # QKV + bias
        qkv = x @ w["qkv"]
        qkv_bias = get_param(f"blocks.{bid}.qkv.bias")
        if qkv_bias is not None: qkv = qkv + qkv_bias.float().to(device)
        
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, S, 32, 128).permute(0, 2, 1, 3)
        k = k.view(B, S, 32, 128).permute(0, 2, 1, 3)
        v = v.view(B, S, 32, 128).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * (128 ** -0.5)
        attn = F.softmax(attn.float(), dim=-1).to(x.dtype)
        attn_out = attn @ v
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, 4096)
        
        # Proj + bias
        attn_out = attn_out @ w["proj"]
        proj_bias = get_param(f"blocks.{bid}.proj.bias")
        if proj_bias is not None: attn_out = attn_out + proj_bias.float().to(device)
        x = residual + attn_out
        
        del w["qkv"], w["proj"]
        
        # LN2 + FFN
        residual = x
        if w["ln2_w"] is not None:
            x = F.layer_norm(x.float(), [D], weight=w["ln2_w"].float(),
                           bias=w["ln2_b"].float() if w["ln2_b"] is not None else None, eps=1e-5).to(x.dtype)
        
        x = x @ w["ff1"]
        ff1_bias = get_param(f"blocks.{bid}.ff1.bias")
        if ff1_bias is not None: x = x + ff1_bias.float().to(device)
        x = gelu(x)
        x = x @ w["ff2"]
        ff2_bias = get_param(f"blocks.{bid}.ff2.bias")
        if ff2_bias is not None: x = x + ff2_bias.float().to(device)
        x = residual + x
        
        del w["ff1"], w["ff2"]
        torch.cuda.empty_cache()
    
    # Final LN + head
    ln_f_w = get_param("ln_f.weight")
    ln_f_b = get_param("ln_f.bias")
    if ln_f_w is not None:
        ln_f_w_g = ln_f_w.float().to(device)
        ln_f_b_g = ln_f_b.float().to(device) if ln_f_b is not None else None
        x = F.layer_norm(x.float(), [D], weight=ln_f_w_g, bias=ln_f_b_g, eps=1e-5).to(x.dtype)
    
    if head_w is not None:
        x = x.float() @ head_w
        if head_b is not None:
            x = x + head_b
    
    all_logits.append(x.cpu().numpy())
    
    if start == 0:
        dt = time.perf_counter() - t0
        print(f"  First batch: {dt:.1f}s shape={all_logits[-1].shape}", file=sys.stderr)

# Results
logits = np.concatenate(all_logits, axis=0).astype(np.float32)
if logits.ndim == 4:
    logits = logits.reshape(-1, logits.shape[-2], logits.shape[-1])
dt = time.perf_counter() - t0
golden = np.load(os.path.join(GOLDEN_DIR, "logits.npy"))
ok = np.allclose(logits, golden, rtol=1e-3, atol=1e-3)
md = np.max(np.abs(logits - golden))
print(f"\nTime: {dt:.1f}s | Precision: {'PASS' if ok else 'FAIL'} | MAX_DIFF: {md:.2e}", file=sys.stderr)
