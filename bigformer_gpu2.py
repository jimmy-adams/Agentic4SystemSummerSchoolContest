"""BigFormer GPU streaming: per-block weight loading.

Each of 24 blocks has ~804MB weights (4 matrices). 
Load one block to GPU at a time, activations stay on GPU.
"""
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
print("=== Loading ONNX graph ===", file=sys.stderr)
t0 = time.perf_counter()
model = onnx.load(ONNX_PATH)

# ── Parse graph to find MatMul→weight mapping per block ──
# Strategy: iterate nodes, track block boundaries by /blocks.N/ prefix

init_map = {i.name: i for i in model.graph.initializer}
node_names = [n.name for n in model.graph.node]

# Build block→weight mapping
block_weights = defaultdict(lambda: {"qkv": None, "proj": None, "ff1": None, "ff2": None})
# Also map initializer name→(block_id, role)
weight_location = {}

for i, node in enumerate(model.graph.node):
    if node.op_type != "MatMul":
        continue
    name = node.name
    # Parse block ID from node name like /blocks.0/qkv/MatMul
    parts = name.split("/")
    if len(parts) < 3 or not parts[1].startswith("blocks."):
        continue
    block_id = int(parts[1].split(".")[1])
    sub = parts[2]  # "qkv", "proj", "ff1", "ff2", or just "MatMul"
    
    # Map sub to our categories
    if sub == "qkv":
        role = "qkv"
    elif sub == "proj":
        role = "proj"
    elif sub == "ff1":
        role = "ff1"
    elif sub == "ff2":
        role = "ff2"
    elif sub == "MatMul":
        role = "qkv"  # some nodes are just /blocks.N/MatMul
    elif sub == "MatMul_1":
        role = "proj"
    else:
        role = sub
    
    # Find the weight initializer input
    for inp in node.input:
        if inp in init_map:
            weight_location[inp] = (block_id, role)
            block_weights[block_id][role] = inp
            break

num_blocks = max(block_weights.keys()) + 1
print(f"Blocks: {num_blocks}", file=sys.stderr)
for bid in range(min(3, num_blocks)):
    bw = block_weights[bid]
    print(f"  Block {bid}: qkv={bw['qkv']} proj={bw['proj']} ff1={bw['ff1']} ff2={bw['ff2']}", file=sys.stderr)

# ── Load weight DATA from external file ──
# Instead of loading all at once, we'll stream per block
print("\n=== Pre-loading weight metadata ===", file=sys.stderr)

# We need to load the actual tensor data. Use to_array which reads from .onnx.data
weight_data_cpu = {}
total_size = 0
for name, (bid, role) in weight_location.items():
    init = init_map[name]
    arr = to_array(init)  # reads from external data
    weight_data_cpu[name] = torch.from_numpy(arr.copy())
    total_size += arr.nbytes

# Also load non-block MatMul weights (head, etc.)
head_w_name = None
for i, node in enumerate(model.graph.node):
    if node.op_type != "MatMul":
        continue
    name = node.name
    parts = name.split("/")
    if len(parts) >= 3 and parts[1].startswith("blocks."):
        continue  # already handled
    # Non-block MatMul (e.g. /head/MatMul)
    for inp in node.input:
        if inp in init_map:
            arr = to_array(init_map[inp])
            weight_data_cpu[inp] = torch.from_numpy(arr.copy())
            total_size += arr.nbytes
            if "head" in name.lower() or parts[-1] == "MatMul" and len(parts) == 2:
                head_w_name = inp
            print(f"  Extra weight: {inp} shape={list(arr.shape)} size={arr.nbytes/1e6:.1f}MB", file=sys.stderr)

print(f"  Total weights on CPU: {total_size/1e9:.1f} GB", file=sys.stderr)
print(f"  Head weight: {head_w_name}", file=sys.stderr)

# Also load non-MatMul initializers (token embeddings, biases, LN params)
other_weights = {}
for init in model.graph.initializer:
    if init.name not in weight_location:
        arr = to_array(init)
        other_weights[init.name] = torch.from_numpy(arr.copy())

print(f"  Other weights (embeddings, biases): {sum(w.numel()*w.element_size() for w in other_weights.values())/1e6:.1f} MB", file=sys.stderr)

# ── Shared weights to GPU ──
tok_emb = other_weights.get("tok_emb.weight", None)
pos_emb = other_weights.get("pos_emb", None)
if tok_emb is not None:
    tok_emb = tok_emb.float().to(device)
if pos_emb is not None:
    pos_emb = pos_emb.float().to(device)

# Head weight (last MatMul, /head/MatMul)
head_w = weight_data_cpu.get(head_w_name) if head_w_name else None
if head_w is not None:
    head_w = head_w.float()
    print(f"  Head: {head_w.shape}", file=sys.stderr)
head_b = other_weights.get("head.bias")
if head_b is not None:
    head_b = head_b.float()

gc.collect()
torch.cuda.empty_cache()
print(f"  GPU memory after shared: {torch.cuda.memory_allocated()/1e6:.1f} MB", file=sys.stderr)

# ── Load input ──
print("\n=== Loading input ===", file=sys.stderr)
with open(os.path.join(INPUT_DIR, "manifest.json")) as f:
    manifest = json.load(f)
input_data = np.load(os.path.join(INPUT_DIR, manifest["tensors"][0]["file"]))
input_ids = torch.from_numpy(input_data).long()
N = input_ids.shape[0]
print(f"  Input: {input_ids.shape}", file=sys.stderr)

# ── Inference ──
print("\n=== Running inference ===", file=sys.stderr)
D = tok_emb.shape[1]  # 4096

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

all_logits = []

for start in range(0, N, BATCH_SIZE):
    end = min(start + BATCH_SIZE, N)
    batch_ids = input_ids[start:end].to(device)
    B = batch_ids.shape[0]
    S = batch_ids.shape[1]
    
    # Embeddings
    x = tok_emb[batch_ids]
    x = x + pos_emb[:S, :].unsqueeze(0)
    
    # ── Per-block streaming ──
    for bid in range(num_blocks):
        bw = block_weights[bid]
        
        # Load block weights to GPU
        w_qkv = weight_data_cpu[bw["qkv"]].float().to(device)
        w_proj = weight_data_cpu[bw["proj"]].float().to(device)
        w_ff1 = weight_data_cpu[bw["ff1"]].float().to(device)
        w_ff2 = weight_data_cpu[bw["ff2"]].float().to(device)
        
        # LN1 params (shared, from other_weights)
        ln1_w = other_weights.get(f"blocks.{bid}.ln1.weight")
        ln1_b = other_weights.get(f"blocks.{bid}.ln1.bias")
        if ln1_w is not None:
            ln1_w = ln1_w.float().to(device)
        if ln1_b is not None:
            ln1_b = ln1_b.float().to(device)
        
        # ── LayerNorm 1 + Self-Attention ──
        residual = x
        if ln1_w is not None:
            x = layer_norm(x, ln1_w, ln1_b)
        
        # QKV matmul (w_qkv: [4096, 12288], x: [B, S, 4096] → [B, S, 12288])
        qkv = x @ w_qkv
        
        # Split QKV (the model likely splits the last dim into 3)
        # w_qkv shape is [4096, 12288] = D × 3D, so qkv output is [B, S, 3*D]
        D_head = D  # assuming D = head_dim * num_heads
        q, k, v = qkv.chunk(3, dim=-1)  # each [B, S, D]
        
        # Attention
        scale = D ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn.float(), dim=-1).to(x.dtype)
        attn_out = attn @ v
        
        # Output projection (w_proj: [4096, 4096])
        attn_out = attn_out @ w_proj
        x = residual + attn_out
        
        # Free attention weights
        del w_qkv, w_proj
        torch.cuda.empty_cache()
        
        # ── LayerNorm 2 + FFN ──
        ln2_w = other_weights.get(f"blocks.{bid}.ln2.weight")
        ln2_b = other_weights.get(f"blocks.{bid}.ln2.bias")
        if ln2_w is not None:
            ln2_w = ln2_w.float().to(device)
        if ln2_b is not None:
            ln2_b = ln2_b.float().to(device)
        
        residual = x
        if ln2_w is not None:
            x = layer_norm(x, ln2_w, ln2_b)
        
        # FFN (w_ff1: [4096, 16384], w_ff2: [16384, 4096])
        x = x @ w_ff1
        x = gelu(x)
        x = x @ w_ff2
        x = residual + x
        
        # Free FFN weights
        del w_ff1, w_ff2, ln1_w, ln1_b, ln2_w, ln2_b
        torch.cuda.empty_cache()
    
    # ── Final LayerNorm ──
    ln_f_w = other_weights.get("ln_f.weight")
    ln_f_b = other_weights.get("ln_f.bias")
    if ln_f_w is not None:
        ln_f_w = ln_f_w.float().to(device)
        if ln_f_b is not None:
            ln_f_b = ln_f_b.float().to(device)
        x = layer_norm(x, ln_f_w, ln_f_b)
    
    # ── Head ──
    if head_w is not None:
        head_w_gpu = head_w.to(device)
        if head_b is not None:
            head_b_gpu = head_b.to(device)
            x = x.float() @ head_w_gpu + head_b_gpu
        else:
            x = x.float() @ head_w_gpu
        del head_w_gpu
        if head_b is not None:
            del head_b_gpu
    
    all_logits.append(x.cpu().numpy())
    
    if start == 0:
        dt = time.perf_counter() - t0
        print(f"  First batch: {dt:.1f}s, shape={all_logits[-1].shape}", file=sys.stderr)

# ── Debug shapes ──
print(f"\n  Num batches: {len(all_logits)}", file=sys.stderr)
print(f"  Shape[0]: {all_logits[0].shape}", file=sys.stderr)
print(f"  Shape[-1]: {all_logits[-1].shape}", file=sys.stderr)

# ── Results ──
logits = np.concatenate(all_logits, axis=0).astype(np.float32)
# Handle extra batch dim if present
if logits.ndim == 4:
    logits = logits.reshape(-1, logits.shape[-2], logits.shape[-1])
dt = time.perf_counter() - t0

print(f"\n  logits shape: {logits.shape}", file=sys.stderr)

golden = np.load(os.path.join(GOLDEN_DIR, "logits.npy"))
ok = np.allclose(logits, golden, rtol=1e-3, atol=1e-3)
md = np.max(np.abs(logits - golden))
print(f"\nTime: {dt:.1f}s | Precision: {'PASS' if ok else 'FAIL'} | MAX_DIFF: {md:.2e}", file=sys.stderr)
print(f"Logits shape: {logits.shape}", file=sys.stderr)
