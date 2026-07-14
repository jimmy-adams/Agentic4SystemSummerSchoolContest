"""FP16 precision diagnostic: per-block error analysis.

Runs FP32 and FP16 inference, measures error accumulation per block.
Identifies which operations (attention, FFN, LayerNorm) are most sensitive.
"""
import onnx, numpy as np, torch, torch.nn.functional as F
import time, os, sys, json

ONNX_PATH = "/workspace/C3/testcases/models/bigformer_v1.onnx"
INPUT_DIR = "/workspace/C3/testcases/testdata/c35/bigformer_v1/input"
device = torch.device("cuda")

print("=== FP16 precision diagnostic ===")
model = onnx.load(ONNX_PATH)
identity_map = {n.output[0]: n.input[0] for n in model.graph.node if n.op_type == "Identity"}
def resolve(name):
    visited = set()
    while name in identity_map and name not in visited:
        visited.add(name); name = identity_map[name]
    return name

init_data = {}
for init in model.graph.initializer:
    init_data[init.name] = torch.from_numpy(onnx.numpy_helper.to_array(init).copy())

init_names = set(init_data.keys())
def gp(name):
    r = resolve(name)
    return init_data.get(r) if r in init_names else None

block_weights = {}
for node in model.graph.node:
    if node.op_type != "MatMul": continue
    parts = node.name.split("/")
    if len(parts) < 3 or not parts[1].startswith("blocks."):
        if "head" in node.name:
            for inp in node.input:
                r = resolve(inp)
                if r in init_names: block_weights["head"] = r
        continue
    bid, sub = int(parts[1].split(".")[1]), parts[2]
    if bid not in block_weights: block_weights[bid] = {}
    for inp in node.input:
        r = resolve(inp)
        if r in init_names: block_weights[bid][sub] = r; break

num_blocks = 24

def forward(x, block_weights, gp, device, num_blocks, D, use_fp16=False):
    """Run BigFormer forward, return list of (x_after_block, error_vs_ref)."""
    dtype = torch.float16 if use_fp16 else torch.float32
    B, S = x.shape[0], x.shape[1]
    
    ln_f_w = gp("ln_f.weight")
    ln_f_b = gp("ln_f.bias")
    head_w = init_data[block_weights["head"]]
    head_b = gp("head.bias")
    
    results = []
    
    for bid in range(num_blocks):
        x_before = x.clone()
        
        bw = block_weights[bid]
        w = {k: init_data[bw[k]].to(device, dtype=dtype) for k in ["qkv","proj","ff1","ff2"]}
        ln1_w = gp(f"blocks.{bid}.ln1.weight")
        ln1_b = gp(f"blocks.{bid}.ln1.bias")
        ln2_w = gp(f"blocks.{bid}.ln2.weight")
        ln2_b = gp(f"blocks.{bid}.ln2.bias")
        qkv_b = gp(f"blocks.{bid}.qkv.bias")
        proj_b = gp(f"blocks.{bid}.proj.bias")
        ff1_b = gp(f"blocks.{bid}.ff1.bias")
        ff2_b = gp(f"blocks.{bid}.ff2.bias")
        
        x = x.to(device, dtype=dtype)
        
        # LN1
        residual = x
        if ln1_w is not None:
            x_fp32 = F.layer_norm(x.float(), [D], weight=ln1_w.float().to(device), 
                                   bias=ln1_b.float().to(device) if ln1_b is not None else None, eps=1e-5)
            x = x_fp32.to(dtype)
        
        # QKV
        qkv = x @ w["qkv"]
        if qkv_b is not None: qkv = qkv + qkv_b.to(device, dtype=dtype)
        q,k,v = qkv.chunk(3, dim=-1)
        q = q.view(q.shape[0], q.shape[1], 32, 128).permute(0,2,1,3)
        k = k.view(k.shape[0], k.shape[1], 32, 128).permute(0,2,1,3)
        v = v.view(v.shape[0], v.shape[1], 32, 128).permute(0,2,1,3)
        
        # Attention (FP32 for stability)
        attn = (q.float() @ k.float().transpose(-2,-1)) * (128**-0.5)
        attn = F.softmax(attn, dim=-1)
        attn_out = (attn @ v.float()).to(dtype)
        attn_out = attn_out.permute(0,2,1,3).reshape(attn_out.shape[0], attn_out.shape[2], 4096)
        
        # Proj
        attn_out = attn_out @ w["proj"]
        if proj_b is not None: attn_out = attn_out + proj_b.to(device, dtype=dtype)
        x = residual + attn_out.to(dtype)
        
        # LN2
        residual = x
        if ln2_w is not None:
            x_fp32 = F.layer_norm(x.float(), [D], weight=ln2_w.float().to(device),
                                   bias=ln2_b.float().to(device) if ln2_b is not None else None, eps=1e-5)
            x = x_fp32.to(dtype)
        
        # FFN
        ffn1 = x @ w["ff1"]
        if ff1_b is not None: ffn1 = ffn1 + ff1_b.to(device, dtype=dtype)
        x = (0.5 * ffn1 * (1.0 + torch.erf(ffn1.float() / 1.41421356237))).to(dtype)
        x = x @ w["ff2"]
        if ff2_b is not None: x = x + ff2_b.to(device, dtype=dtype)
        x = residual + x.to(dtype)
        
        results.append(x.float().cpu())
        
        del w, q, k, v, attn, attn_out
        torch.cuda.empty_cache()
    
    # Final LN + Head
    if ln_f_w is not None:
        x = F.layer_norm(x.float(), [D], weight=ln_f_w.float().to(device),
                        bias=ln_f_b.float().to(device) if ln_f_b is not None else None, eps=1e-5)
    if head_w is not None:
        x = x.float() @ head_w.float().to(device)
        if head_b is not None: x = x + head_b.float().to(device)
    
    return [r.numpy() for r in results], x.float().cpu().numpy()

# Load input
with open(os.path.join(INPUT_DIR, "manifest.json")) as f: manifest = json.load(f)
input_ids = torch.from_numpy(np.load(os.path.join(INPUT_DIR, manifest["tensors"][0]["file"]))).long()
B = 16
batch_ids = input_ids[:B].to(device)
tok_emb = gp("tok_emb.weight").float().to(device)
pos_emb = gp("pos_emb").float().to(device)
D = tok_emb.shape[1]
x = tok_emb[batch_ids] + pos_emb[:32, :].unsqueeze(0)

# Run FP32 once to warm up
# Use batch=1 directly
batch_ids = input_ids[:1].to(device)
x = tok_emb[batch_ids] + pos_emb[:, :32, :]
print(f"x shape: {x.shape}", flush=True)
x0 = x.clone()
_, _ = forward(x0, block_weights, gp, device, num_blocks, D, use_fp16=False)
torch.cuda.synchronize()
print("Warmup done")

# Run diagnostic on small batch
x_diag = x[:1].clone()
print(f"Input shape: {x_diag.shape}")
fp32_outs, fp32_final = forward(x_diag, block_weights, gp, device, num_blocks, D, use_fp16=False)
fp16_outs, fp16_final = forward(x_diag, block_weights, gp, device, num_blocks, D, use_fp16=True)

print(f"\n{'Block':<8} {'FP16 error':<20} {'Error growth':<15}")
print("-" * 50)
prev_error = 0
for bid in range(num_blocks):
    error = np.max(np.abs(fp32_outs[bid] - fp16_outs[bid]))
    growth = error - prev_error
    bar = "|" * int(min(error * 500, 40))
    print(f"  {bid:<6} {error:<20.6f} {growth:+.6f}  {bar}")
    prev_error = error

final_error = np.max(np.abs(fp32_final - fp16_final))
print(f"\n  FINAL  {final_error:.6f}  (threshold: 0.001000)")

# Simple component test: run block 0 with specific parts in FP16
print(f"\n--- Component sensitivity (block 0, simple) ---")
B, S = 1, 32
x0 = x.clone()

# FP32 baseline (full block 0)
fp32_block0 = forward(x0.clone(), block_weights, gp, device, 1, D, use_fp16=False)[0][0]

# Test each component: convert one weight to FP16, compute, compare
bw = block_weights[0]
components = {
    "QKV weight": "qkv",
    "Proj weight": "proj", 
    "FF1 weight": "ff1",
    "FF2 weight": "ff2",
    "All 4 FP16": "all",
}

for label, key in components.items():
    if key == "all":
        fp16_parts = ["qkv", "proj", "ff1", "ff2"]
    else:
        fp16_parts = [key]
    
    # Run block 0 with selected parts in FP16
    w = {}
    for k in ["qkv", "proj", "ff1", "ff2"]:
        w[k] = init_data[bw[k]].to(device, dtype=torch.float16 if k in fp16_parts else torch.float32)
    
    x = x0.clone().to(device, dtype=torch.float32)
    residual = x
    ln1_w = gp("blocks.0.ln1.weight"); ln1_b = gp("blocks.0.ln1.bias")
    x = F.layer_norm(x.float(), [D], weight=ln1_w.float().to(device),
                    bias=ln1_b.float().to(device) if ln1_b is not None else None, eps=1e-5)
    
    # QKV
    _dtype = torch.float16 if "qkv" in fp16_parts else torch.float32
    qkv = x.to(_dtype) @ w["qkv"]
    q,k,v = qkv.float().chunk(3, dim=-1)
    q = q.view(B,S,32,128).permute(0,2,1,3)
    k = k.view(B,S,32,128).permute(0,2,1,3)
    v = v.view(B,S,32,128).permute(0,2,1,3)
    attn_out = (F.softmax((q @ k.transpose(-2,-1)) * (128**-0.5), dim=-1) @ v)
    attn_out = attn_out.permute(0,2,1,3).reshape(B,S,4096)
    
    # Proj
    _dtype = torch.float16 if "proj" in fp16_parts else torch.float32
    attn_out = attn_out.to(_dtype) @ w["proj"]
    x = residual + attn_out.float()
    
    # LN2
    residual = x
    ln2_w = gp("blocks.0.ln2.weight"); ln2_b = gp("blocks.0.ln2.bias")
    x = F.layer_norm(x.float(), [D], weight=ln2_w.float().to(device),
                    bias=ln2_b.float().to(device) if ln2_b is not None else None, eps=1e-5)
    
    # FF1
    _dtype = torch.float16 if "ff1" in fp16_parts else torch.float32
    x = x.to(_dtype) @ w["ff1"]
    x = F.gelu(x.float())
    
    # FF2
    _dtype = torch.float16 if "ff2" in fp16_parts else torch.float32
    x = x.to(_dtype) @ w["ff2"]
    x = residual + x.float()
    
    out = x.cpu().numpy()
    err = np.max(np.abs(fp32_block0 - out))
    bar = "|" * int(min(err * 500, 50))
    print(f"  {label:<18s} error={err:.6f}  {bar}")

# Also test: attention in FP16 with FP32 weights
print(f"\n  -- Attention ops in FP16 (weights FP32) --")
x = x0.clone().to(device)
residual = x
ln1_w = gp("blocks.0.ln1.weight"); ln1_b = gp("blocks.0.ln1.bias")
x = F.layer_norm(x.float(), [D], weight=ln1_w.float().to(device),
                bias=ln1_b.float().to(device) if ln1_b is not None else None, eps=1e-5)

qkv = x @ init_data[bw["qkv"]].float().to(device)
q,k,v = qkv.chunk(3, dim=-1)
q = q.view(B,S,32,128).permute(0,2,1,3).float()
k = k.view(B,S,32,128).permute(0,2,1,3).float()
v = v.view(B,S,32,128).permute(0,2,1,3).float()

# Test FP16 in attention matmul
attn_fp32 = F.softmax((q @ k.transpose(-2,-1)) * (128**-0.5), dim=-1) @ v
attn_fp16 = F.softmax((q.half() @ k.half().transpose(-2,-1)).float() * (128**-0.5), dim=-1) @ v.half()
print(f"  Attention(SD) FP16:    err={np.max(np.abs(attn_fp32.float().cpu().numpy() - attn_fp16.float().cpu().numpy())):.6f}")
attn_fp32_2 = (F.softmax((q @ k.transpose(-2,-1)) * (128**-0.5), dim=-1) @ v)
print(f"  Attention(SD) FP32x2:  err={np.max(np.abs(attn_fp32.cpu().numpy() - attn_fp32_2.cpu().numpy())):.6f} (baseline)")

# Test softmax sensitivity  
scores = (q @ k.transpose(-2,-1)) * (128**-0.5)
softmax_fp32 = F.softmax(scores, dim=-1)
softmax_fp16 = F.softmax(scores.half(), dim=-1).float()
print(f"  Softmax FP16:          err={np.max(np.abs(softmax_fp32.cpu().numpy() - softmax_fp16.cpu().numpy())):.6f}")

