"""Test selective TF32 on BigFormer."""
import torch, torch.nn.functional as F, time, numpy as np, os, sys, json
from collections import defaultdict
import onnx
from onnx.numpy_helper import to_array

ONNX_PATH = "/workspace/C3/testcases/models/bigformer_v1.onnx"
INPUT_DIR = "/workspace/C3/testcases/testdata/c35/bigformer_v1/input"
GOLDEN_DIR = "/workspace/C3/testcases/testdata/c35/bigformer_v1/golden"
BATCH_SIZE = 16
device = torch.device("cuda")

print("=== Selective TF32: FFN matmuls only ===")
t0 = time.perf_counter()

# Parse ONNX
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

# Load weights
init_data = {}
init_set = set()
for init in model.graph.initializer:
    arr = to_array(init)
    init_data[init.name] = torch.from_numpy(arr.copy()).pin_memory()
    init_set.add(init.name)

def get_param(name):
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
    bid = int(parts[1].split(".")[1])
    sub = parts[2]
    if bid not in block_weights: block_weights[bid] = {}
    for inp in node.input:
        r = resolve(inp)
        if r in init_set:
            block_weights[bid][sub] = r
            break

num_blocks = 24

# Pre-build blocks
block_w_cpu = []
for bid in range(num_blocks):
    bw = block_weights[bid]
    block_w_cpu.append({
        "qkv": init_data[bw["qkv"]],
        "proj": init_data[bw["proj"]],
        "ff1": init_data[bw["ff1"]],
        "ff2": init_data[bw["ff2"]],
        "qkv_b": get_param(f"blocks.{bid}.qkv.bias"),
        "proj_b": get_param(f"blocks.{bid}.proj.bias"),
        "ff1_b": get_param(f"blocks.{bid}.ff1.bias"),
        "ff2_b": get_param(f"blocks.{bid}.ff2.bias"),
        "ln1_w": get_param(f"blocks.{bid}.ln1.weight"),
        "ln1_b": get_param(f"blocks.{bid}.ln1.bias"),
        "ln2_w": get_param(f"blocks.{bid}.ln2.weight"),
        "ln2_b": get_param(f"blocks.{bid}.ln2.bias"),
    })

# Shared
tok_emb = get_param("tok_emb.weight").float().to(device)
pos_emb = get_param("pos_emb").float().to(device)
head_w = init_data[block_weights["head"]].float().to(device) if "head" in block_weights else None
head_b = get_param("head.bias")
if head_b is not None: head_b = head_b.float().to(device)
ln_f_w_g = get_param("ln_f.weight")
ln_f_b_g = get_param("ln_f.bias")
if ln_f_w_g is not None: ln_f_w_g = ln_f_w_g.float().to(device)
if ln_f_b_g is not None: ln_f_b_g = ln_f_b_g.float().to(device)

# Input
with open(os.path.join(INPUT_DIR, "manifest.json")) as f:
    manifest = json.load(f)
input_data = np.load(os.path.join(INPUT_DIR, manifest["tensors"][0]["file"]))
input_ids = torch.from_numpy(input_data).long()
N = input_ids.shape[0]
D = tok_emb.shape[1]

print(f"=== Inference (selective TF32) ===")

all_logits = []

for start in range(0, N, BATCH_SIZE):
    end = min(start + BATCH_SIZE, N)
    batch_ids = input_ids[start:end].to(device)
    B, S = batch_ids.shape[0], batch_ids.shape[1]
    
    x = tok_emb[batch_ids]
    x = x + pos_emb[:S, :].unsqueeze(0)
    
    # Pre-load block 0
    w_gpu = {}
    for key in ["qkv", "proj", "ff1", "ff2", "qkv_b", "proj_b", "ff1_b", "ff2_b",
                "ln1_w", "ln1_b", "ln2_w", "ln2_b"]:
        t = block_w_cpu[0][key]
        w_gpu[key] = t.to(device, non_blocking=True) if t is not None else None
    torch.cuda.synchronize()
    
    for bid in range(num_blocks):
        w = w_gpu
        
        # Pre-load next block
        if bid < num_blocks - 1:
            w_gpu = {}
            for key in ["qkv", "proj", "ff1", "ff2", "qkv_b", "proj_b", "ff1_b", "ff2_b",
                        "ln1_w", "ln1_b", "ln2_w", "ln2_b"]:
                t = block_w_cpu[bid+1][key]
                w_gpu[key] = t.to(device, non_blocking=True) if t is not None else None
        
        # LN1
        residual = x
        if w["ln1_w"] is not None:
            x = F.layer_norm(x.float(), [D], weight=w["ln1_w"].float(),
                           bias=w["ln1_b"].float() if w["ln1_b"] is not None else None, eps=1e-5).to(torch.float32)
        
        # QKV (FP32 — precision-sensitive)
        qkv = x @ w["qkv"].float()
        if w["qkv_b"] is not None: qkv = qkv + w["qkv_b"].float()
        
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, S, 32, 128).permute(0, 2, 1, 3)
        k = k.view(B, S, 32, 128).permute(0, 2, 1, 3)
        v = v.view(B, S, 32, 128).permute(0, 2, 1, 3)
        
        # Attention (FP32)
        attn = (q @ k.transpose(-2, -1)) * (128 ** -0.5)
        attn = F.softmax(attn.float(), dim=-1).to(torch.float32)
        attn_out = attn @ v
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, 4096)
        
        # Proj (FP32)
        attn_out = attn_out @ w["proj"].float()
        if w["proj_b"] is not None: attn_out = attn_out + w["proj_b"].float()
        x = residual + attn_out
        del w["qkv"], w["proj"]
        
        # LN2
        residual = x
        if w["ln2_w"] is not None:
            x = F.layer_norm(x.float(), [D], weight=w["ln2_w"].float(),
                           bias=w["ln2_b"].float() if w["ln2_b"] is not None else None, eps=1e-5).to(torch.float32)
        
        # FFN — ENABLE TF32 for these matmuls only
        torch.backends.cuda.matmul.allow_tf32 = True
        x = x @ w["ff1"].float()
        if w["ff1_b"] is not None: x = x + w["ff1_b"].float()
        x = F.gelu(x)
        x = x @ w["ff2"].float()
        if w["ff2_b"] is not None: x = x + w["ff2_b"].float()
        torch.backends.cuda.matmul.allow_tf32 = False  # disable for other ops
        
        x = residual + x
        del w["ff1"], w["ff2"]
    
    # Final LN + Head
    if ln_f_w_g is not None:
        x = F.layer_norm(x.float(), [D], weight=ln_f_w_g, bias=ln_f_b_g, eps=1e-5)
    
    x = x.float()
    if head_w is not None:
        x = x @ head_w.float()
        if head_b is not None:
            x = x + head_b
    
    all_logits.append(x.cpu().numpy())
    
    if start == 0:
        dt = time.perf_counter() - t0
        print(f"  First batch: {dt:.1f}s shape={all_logits[-1].shape}")

# Results
logits = np.concatenate(all_logits, axis=0).astype(np.float32)
if logits.ndim == 4:
    logits = logits.reshape(-1, logits.shape[-2], logits.shape[-1])
dt = time.perf_counter() - t0
golden = np.load(os.path.join(GOLDEN_DIR, "logits.npy"))
ok = np.allclose(logits, golden, rtol=1e-3, atol=1e-3)
md = np.max(np.abs(logits - golden))
print(f"\nTime: {dt:.1f}s | Precision: {'PASS' if ok else 'FAIL'} | MAX_DIFF: {md:.2e}")
