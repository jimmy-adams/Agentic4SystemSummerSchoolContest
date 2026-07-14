"""BigFormer GPU streaming — fix Identity weight aliasing."""
import onnx
from onnx.numpy_helper import to_array
import numpy as np
import torch
import torch.nn.functional as F
import time, os, sys, json, gc
from collections import defaultdict

ONNX_PATH = "/workspace/C3/testcases/models/bigformer_v1.onnx"
INPUT_DIR = "/workspace/C3/testcases/testdata/c35/bigformer_v1/input"
GOLDEN_DIR = "/workspace/C3/testcases/testdata/c35/bigformer_v1/golden"
BATCH_SIZE = 16
device = torch.device("cuda")

print("=== Loading ONNX ===", file=sys.stderr)
t0 = time.perf_counter()
model = onnx.load(ONNX_PATH)

# ── Build Identity alias resolution ──
identity_map = {}
for node in model.graph.node:
    if node.op_type == "Identity":
        identity_map[node.output[0]] = node.input[0]

def resolve_name(name):
    """Follow Identity chain to find actual initializer name."""
    visited = set()
    while name in identity_map and name not in visited:
        visited.add(name)
        name = identity_map[name]
    return name

# ── Load initializers ──
init_data = {}
for init in model.graph.initializer:
    arr = to_array(init)
    init_data[init.name] = torch.from_numpy(arr.copy())

print(f"  Loaded {len(init_data)} initializers, {sum(t.numel()*t.element_size() for t in init_data.values())/1e9:.1f} GB", file=sys.stderr)

# ── Build resolved param lookup ──
def get_param(name):
    """Get parameter tensor, resolving Identity aliases."""
    resolved = resolve_name(name)
    if resolved in init_data:
        return init_data[resolved]
    return None

# ── Map block weights ──
init_names = set(init_data.keys())
block_weights = {}
for node in model.graph.node:
    if node.op_type != "MatMul":
        continue
    parts = node.name.split("/")
    if len(parts) < 3 or not parts[1].startswith("blocks."):
        # Head
        if "head" in node.name:
            for inp in node.input:
                resolved = resolve_name(inp)
                if resolved in init_data:
                    block_weights["head"] = resolved
        continue
    
    bid = int(parts[1].split(".")[1])
    sub = parts[2] if len(parts) > 2 else "unknown"
    
    if bid not in block_weights:
        block_weights[bid] = {}
    
    for inp in node.input:
        resolved = resolve_name(inp)
        if resolved in init_data:
            block_weights[bid][sub] = resolved
            break

num_blocks = max(b for b in block_weights if isinstance(b, int)) + 1
print(f"Blocks: {num_blocks}", file=sys.stderr)
for bid in range(min(2, num_blocks)):
    print(f"  Block {bid}: {block_weights[bid]}", file=sys.stderr)
print(f"  Head: {block_weights.get('head')}", file=sys.stderr)

# ── Shared/embedding params ──
tok_emb = get_param("tok_emb.weight")
pos_emb = get_param("pos_emb")
head_w = init_data[block_weights["head"]].float() if "head" in block_weights else None
head_b = get_param("head.bias")

# Move tiny shared params to GPU
if tok_emb is not None:
    tok_emb = tok_emb.float().to(device)
if pos_emb is not None:
    pos_emb = pos_emb.float().to(device)
if head_b is not None:
    head_b = head_b.float().to(device)

gc.collect(); torch.cuda.empty_cache()
print(f"GPU after shared: {torch.cuda.memory_allocated()/1e6:.1f}MB", file=sys.stderr)

# ── Load input ──
with open(os.path.join(INPUT_DIR, "manifest.json")) as f:
    manifest = json.load(f)
input_data = np.load(os.path.join(INPUT_DIR, manifest["tensors"][0]["file"]))
input_ids = torch.from_numpy(input_data).long()
N = input_ids.shape[0]
D = tok_emb.shape[1]
print(f"Input: {list(input_ids.shape)} D={D}", file=sys.stderr)

# ── Helper functions ──
def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / 1.41421356237))

def layer_norm(x, weight, bias=None, eps=1e-5):
    mean = x.float().mean(dim=-1, keepdim=True)
    var = ((x.float() - mean) ** 2).mean(dim=-1, keepdim=True)
    out = (x.float() - mean) / torch.sqrt(var + eps)
    out = out * weight.float()
    if bias is not None:
        out = out + bias.float()
    return out.to(x.dtype)

# ── Inference ──
print("=== Inference ===", file=sys.stderr)
all_logits = []

for start in range(0, N, BATCH_SIZE):
    end = min(start + BATCH_SIZE, N)
    batch_ids = input_ids[start:end].to(device)
    B = batch_ids.shape[0]
    S = batch_ids.shape[1]
    
    x = tok_emb[batch_ids]  # [B, S, D]
    x = x + pos_emb[:S, :].unsqueeze(0)
    
    for bid in range(num_blocks):
        bw = block_weights[bid]
        
        # Load this block's weights to GPU
        w_qkv = init_data[bw["qkv"]].float().to(device)
        w_proj = init_data[bw["proj"]].float().to(device)
        w_ff1 = init_data[bw["ff1"]].float().to(device)
        w_ff2 = init_data[bw["ff2"]].float().to(device)
        
        # LN1 params (resolved through Identity)
        ln1_w = get_param(f"blocks.{bid}.ln1.weight")
        ln1_b = get_param(f"blocks.{bid}.ln1.bias")
        if ln1_w is not None: ln1_w = ln1_w.float().to(device)
        if ln1_b is not None: ln1_b = ln1_b.float().to(device)
        
        # Self-attention
        residual = x
        if ln1_w is not None:
            x = F.layer_norm(x.float(), [x.shape[-1]], weight=ln1_w.float(), bias=ln1_b.float() if ln1_b is not None else None, eps=1e-5).to(x.dtype)
        
        qkv = x @ w_qkv  # [B, S, 12288]
        # Add QKV bias
        qkv_bias = get_param(f"blocks.{bid}.qkv.bias")
        if qkv_bias is not None:
            qkv = qkv + qkv_bias.float().to(device)
        q, k, v = qkv.chunk(3, dim=-1)  # each [B,S,4096]
        
        # Multi-head: 32 heads, head_dim=128
        # [B,S,4096] -> [B,S,32,128] -> [B,32,S,128]
        q = q.view(B, S, 32, 128).permute(0, 2, 1, 3)
        k = k.view(B, S, 32, 128).permute(0, 2, 1, 3)
        v = v.view(B, S, 32, 128).permute(0, 2, 1, 3)
        
        # Scaled dot-product: scale = 1/sqrt(head_dim=128)
        attn = (q @ k.transpose(-2, -1)) * (128 ** -0.5)
        attn = F.softmax(attn.float(), dim=-1).to(x.dtype)
        attn_out = attn @ v  # [B,32,S,128]
        
        # Merge heads: [B,32,S,128] -> [B,S,4096]
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, 4096)
        attn_out = attn_out @ w_proj
        x = residual + attn_out
        
        del w_qkv, w_proj; torch.cuda.empty_cache()
        
        # LN2 + FFN
        ln2_w = get_param(f"blocks.{bid}.ln2.weight")
        ln2_b = get_param(f"blocks.{bid}.ln2.bias")
        if ln2_w is not None: ln2_w = ln2_w.float().to(device)
        if ln2_b is not None: ln2_b = ln2_b.float().to(device)
        
        residual = x
        if ln2_w is not None:
            x = F.layer_norm(x.float(), [x.shape[-1]], weight=ln2_w.float(), bias=ln2_b.float() if ln2_b is not None else None, eps=1e-5).to(x.dtype)
        
        x = x @ w_ff1
        ff1_bias = get_param(f"blocks.{bid}.ff1.bias")
        if ff1_bias is not None:
            x = x + ff1_bias.float().to(device)
        x = gelu(x)
        x = x @ w_ff2
        ff2_bias = get_param(f"blocks.{bid}.ff2.bias")
        if ff2_bias is not None:
            x = x + ff2_bias.float().to(device)
        x = residual + x
        
        del w_ff1, w_ff2, ln1_w, ln1_b, ln2_w, ln2_b
        torch.cuda.empty_cache()
    
    # Final LN
    ln_f_w = get_param("ln_f.weight")
    ln_f_b = get_param("ln_f.bias")
    if ln_f_w is not None:
        ln_f_w = ln_f_w.float().to(device)
        if ln_f_b is not None: ln_f_b = ln_f_b.float().to(device)
        x = F.layer_norm(x.float(), [x.shape[-1]], weight=ln_f_w.float(), bias=ln_f_b.float() if ln_f_b is not None else None, eps=1e-5).to(x.dtype)
    
    # Head
    if head_w is not None:
        hw = head_w.to(device)
        if head_b is not None:
            x = x.float() @ hw + head_b
        else:
            x = x.float() @ hw
        del hw
    
    all_logits.append(x.cpu().numpy())
    
    if start == 0:
        dt = time.perf_counter() - t0
        print(f"  First batch: {dt:.1f}s shape={all_logits[-1].shape}", file=sys.stderr)

# ── Results ──
logits = np.concatenate(all_logits, axis=0).astype(np.float32)
if logits.ndim == 4:
    logits = logits.reshape(-1, logits.shape[-2], logits.shape[-1])
dt = time.perf_counter() - t0
golden = np.load(os.path.join(GOLDEN_DIR, "logits.npy"))
ok = np.allclose(logits, golden, rtol=1e-3, atol=1e-3)
md = np.max(np.abs(logits - golden))
print(f"\nTime: {dt:.1f}s | Precision: {'PASS' if ok else 'FAIL'} | MAX_DIFF: {md:.2e}", file=sys.stderr)
