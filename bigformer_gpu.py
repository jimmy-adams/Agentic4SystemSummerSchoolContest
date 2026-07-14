"""BigFormer ONNX → PyTorch with per-layer GPU streaming.

Strategy:
  1. Parse ONNX, group weights by layer (blocks.0 ~ blocks.23)
  2. Keep all weights on CPU, stream one layer at a time to GPU
  3. Run forward pass manually using PyTorch ops
  4. 512 samples × 32 seq → activations ~MB, fits easily alongside one layer
"""
import onnx
from onnx.numpy_helper import to_array
import numpy as np
import torch
import torch.nn.functional as F
import time, os, sys, json, gc

ONNX_PATH = "/workspace/C3/testcases/models/bigformer_v1.onnx"
INPUT_DIR = "/workspace/C3/testcases/testdata/c35/bigformer_v1/input"
OUTPUT_DIR = "/tmp/bf_gpu_test"

BATCH_SIZE = 16

print("=== Phase 1: Parse ONNX and load all weights to CPU ===", file=sys.stderr)
t0 = time.perf_counter()

model = onnx.load(ONNX_PATH)

# Load all initializers to CPU numpy dict
weights_cpu = {}
for init in model.graph.initializer:
    arr = to_array(init)
    weights_cpu[init.name] = torch.from_numpy(arr.copy())  # copy to avoid mmap ref
    
dt = time.perf_counter() - t0
total_gb = sum(w.numel() * w.element_size() for w in weights_cpu.values()) / 1e9
print(f"  Weights loaded to CPU: {total_gb:.1f} GB in {dt:.1f}s", file=sys.stderr)

# Analyze layer structure
print("\n=== Phase 2: Identify layer structure ===", file=sys.stderr)

# Group initializers by prefix
from collections import defaultdict
layer_weights = defaultdict(dict)
shared_weights = {}

for name, tensor in weights_cpu.items():
    if name.startswith("blocks."):
        parts = name.split(".")
        block_id = int(parts[1])
        key = ".".join(parts[2:])  # e.g. "qkv.weight"
        layer_weights[block_id][key] = tensor
    else:
        shared_weights[name] = tensor

num_layers = max(layer_weights.keys()) + 1
print(f"  Layers: {num_layers} (blocks.0 to blocks.{num_layers-1})", file=sys.stderr)

# Compute per-layer weight size
for bid in range(num_layers):
    lw = layer_weights[bid]
    size_mb = sum(t.numel() * t.element_size() for t in lw.values()) / 1e6
    keys = list(lw.keys())
    print(f"  Block {bid}: {size_mb:.1f} MB  [{', '.join(keys[:4])}...]", file=sys.stderr)

# Shared weights
shared_mb = sum(t.numel() * t.element_size() for t in shared_weights.values()) / 1e6
print(f"  Shared: {shared_mb:.1f} MB  [{', '.join(shared_weights.keys())}]", file=sys.stderr)

# Load shared weights to GPU
print("\n=== Phase 3: Move shared weights to GPU ===", file=sys.stderr)
device = torch.device("cuda")
for name in shared_weights:
    shared_weights[name] = shared_weights[name].to(device)

tok_emb = shared_weights["tok_emb.weight"].float()
pos_emb = shared_weights["pos_emb.weight"].float()
head_weight = shared_weights.get("head.weight", None)
head_bias = shared_weights.get("head.bias", None)
if head_weight is not None:
    head_weight = head_weight.float().to(device)
if head_bias is not None:
    head_bias = head_bias.float().to(device)

print(f"  Shared weights on GPU", file=sys.stderr)
gc.collect()
torch.cuda.empty_cache()

# Load input
print("\n=== Phase 4: Load input ===", file=sys.stderr)
with open(os.path.join(INPUT_DIR, "manifest.json")) as f:
    manifest = json.load(f)
input_data = np.load(os.path.join(INPUT_DIR, manifest["tensors"][0]["file"]))
input_ids = torch.from_numpy(input_data).long()
N = input_ids.shape[0]
print(f"  Input: {input_ids.shape}, {N} samples", file=sys.stderr)

# ── Transformer block forward ──
def transformer_block(x, block_id, layer_weights, device):
    """Single transformer block: self-attention + FFN."""
    w = {k: v.to(device).float() for k, v in layer_weights[block_id].items()}
    
    B, S, D = x.shape
    
    # LayerNorm 1
    ln1_w = w["ln1.weight"]
    ln1_b = w.get("ln1.bias", None)
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    x_norm = (x - mean) / torch.sqrt(var + 1e-5)
    x_norm = x_norm * ln1_w
    if ln1_b is not None:
        x_norm = x_norm + ln1_b
    
    # Self-attention: QKV
    qkv_w = w["qkv.weight"].view(3, D, D)  # [3, D, D]
    qkv_b = w.get("qkv.bias", None)
    if qkv_b is not None:
        qkv_b = qkv_b.view(3, D)
    
    q = x_norm @ qkv_w[0]
    k = x_norm @ qkv_w[1]
    v = x_norm @ qkv_w[2]
    if qkv_b is not None:
        q = q + qkv_b[0]
        k = k + qkv_b[1]
        v = v + qkv_b[2]
    
    # Attention scores
    scale = D ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = F.softmax(attn, dim=-1)
    attn_out = attn @ v
    
    # Output projection
    proj_w = w["proj.weight"]
    proj_b = w.get("proj.bias", None)
    attn_out = attn_out @ proj_w
    if proj_b is not None:
        attn_out = attn_out + proj_b
    
    x = x + attn_out  # residual
    
    # LayerNorm 2
    ln2_w = w.get("ln2.weight", ln1_w)  # some layers share LN params
    ln2_b = w.get("ln2.bias", ln1_b)
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    x_norm2 = (x - mean) / torch.sqrt(var + 1e-5)
    x_norm2 = x_norm2 * ln2_w
    if ln2_b is not None:
        x_norm2 = x_norm2 + ln2_b
    
    # FFN
    ff1_w = w["ff1.weight"]
    ff1_b = w.get("ff1.bias", None)
    ff1_out = x_norm2 @ ff1_w
    if ff1_b is not None:
        ff1_out = ff1_out + ff1_b
    
    # GELU
    ff1_out = 0.5 * ff1_out * (1.0 + torch.erf(ff1_out / 1.41421356237))
    
    ff2_w = w["ff2.weight"]
    ff2_b = w.get("ff2.bias", None)
    ff2_out = ff1_out @ ff2_w
    if ff2_b is not None:
        ff2_out = ff2_out + ff2_b
    
    x = x + ff2_out  # residual
    
    # Free GPU weights
    del w
    torch.cuda.empty_cache()
    
    return x


# ── Full forward pass ──
print("\n=== Phase 5: Run inference ===", file=sys.stderr)

os.makedirs(OUTPUT_DIR, exist_ok=True)
all_logits = []

for start in range(0, N, BATCH_SIZE):
    end = min(start + BATCH_SIZE, N)
    batch_ids = input_ids[start:end].to(device)
    B = batch_ids.shape[0]
    S = batch_ids.shape[1]
    D = tok_emb.shape[1]  # hidden dim
    
    # Embeddings
    x = tok_emb[batch_ids].float()  # [B, S, D]
    x = x + pos_emb[:S, :]  # positional encoding
    
    # Transformer blocks - one at a time
    for bid in range(num_layers):
        x = transformer_block(x, bid, layer_weights, device)
    
    # Final LayerNorm (shared weights, Identity nodes)
    ln_f_w = shared_weights.get("ln_f.weight", None)
    ln_f_b = shared_weights.get("ln_f.bias", None)
    if ln_f_w is not None:
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + 1e-5)
        x = x * ln_f_w.float()
        if ln_f_b is not None:
            x = x + ln_f_b.float()
    
    # Head projection
    if head_weight is not None:
        x = x @ head_weight.T
        if head_bias is not None:
            x = x + head_bias
    
    all_logits.append(x.cpu().numpy())
    
    if start == 0:
        t1 = time.perf_counter()
        print(f"  First batch done in {t1 - t0:.1f}s", file=sys.stderr)

# Concatenate and save
logits = np.concatenate(all_logits, axis=0).astype(np.float32)
np.save(os.path.join(OUTPUT_DIR, "logits.npy"), logits)

manifest_out = {"tensors": [{"name": "logits", "file": "logits.npy", "dtype": "float32", "shape": list(logits.shape)}]}
with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
    json.dump(manifest_out, f, indent=2)

t2 = time.perf_counter()
print(f"\nTotal: {t2 - t0:.1f}s for {N} samples", file=sys.stderr)

# Verify precision
golden = np.load("/workspace/C3/testcases/testdata/c35/bigformer_v1/golden/logits.npy")
ok = np.allclose(logits, golden, rtol=1e-3, atol=1e-3)
md = np.max(np.abs(logits - golden))
print(f"Precision: {'PASS' if ok else 'FAIL'}  MAX_DIFF: {md:.2e}", file=sys.stderr)
