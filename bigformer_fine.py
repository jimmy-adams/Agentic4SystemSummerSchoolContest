"""BigFormer v6: Fine-grained pipeline — split each block into attention + FFN phases.

Phase A: load QKV+Proj (268MB) → compute attention
Phase B: load FF1+FF2 (536MB) on transfer stream during Phase A → compute FFN

Result: weight transfer critical path ~268MB instead of 804MB.
"""
import onnx, numpy as np, torch, torch.nn.functional as F
import time, os, sys, json
from collections import defaultdict

ONNX_PATH = "/workspace/C3/testcases/models/bigformer_v1.onnx"
INPUT_DIR = "/workspace/C3/testcases/testdata/c35/bigformer_v1/input"
GOLDEN_DIR = "/workspace/C3/testcases/testdata/c35/bigformer_v1/golden"
BATCH_SIZE, device = 16, torch.device("cuda")

print("=== Fine-grained pipeline ===")
t0 = time.perf_counter()

# Parse & load (same as before)
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
    init_data[init.name] = torch.from_numpy(arr.copy()).pin_memory()
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

# Pre-build per-block weight names (split into attention + FFN groups)
block_attn = []  # QKV + Proj + LN1
block_ffn = []   # FF1 + FF2 + LN2
for bid in range(num_blocks):
    bw = block_weights[bid]
    block_attn.append({
        "qkv": init_data[bw["qkv"]], "proj": init_data[bw["proj"]],
        "qkv_b": gp(f"blocks.{bid}.qkv.bias"), "proj_b": gp(f"blocks.{bid}.proj.bias"),
        "ln1_w": gp(f"blocks.{bid}.ln1.weight"), "ln1_b": gp(f"blocks.{bid}.ln1.bias"),
    })
    block_ffn.append({
        "ff1": init_data[bw["ff1"]], "ff2": init_data[bw["ff2"]],
        "ff1_b": gp(f"blocks.{bid}.ff1.bias"), "ff2_b": gp(f"blocks.{bid}.ff2.bias"),
        "ln2_w": gp(f"blocks.{bid}.ln2.weight"), "ln2_b": gp(f"blocks.{bid}.ln2.bias"),
    })

print(f"  Loaded: {time.perf_counter()-t0:.1f}s", file=sys.stderr)

# Shared → GPU
tok_emb = gp("tok_emb.weight").float().to(device)
pos_emb = gp("pos_emb").float().to(device)
head_w = init_data[block_weights["head"]].float().to(device) if "head" in block_weights else None
head_b = gp("head.bias"); 
if head_b is not None: head_b = head_b.float().to(device)
ln_f_w = gp("ln_f.weight"); ln_f_b = gp("ln_f.bias")
if ln_f_w is not None: ln_f_w = ln_f_w.float().to(device)
if ln_f_b is not None: ln_f_b = ln_f_b.float().to(device)

# Input
with open(os.path.join(INPUT_DIR, "manifest.json")) as f: manifest = json.load(f)
input_ids = torch.from_numpy(np.load(os.path.join(INPUT_DIR, manifest["tensors"][0]["file"]))).long()
N, D = input_ids.shape[0], tok_emb.shape[1]

# Streams
t_stream = torch.cuda.Stream()  # dedicated transfer stream

def load_dict_gpu(d, stream=None):
    """Load a dict of CPU tensors to GPU, optionally on a specific stream."""
    ctx = torch.cuda.stream(stream) if stream else torch.enable_grad()
    with ctx:
        return {k: v.to(device, non_blocking=True) if v is not None else None for k, v in d.items()}

print(f"=== Inference ===", file=sys.stderr)
all_logits = []

for start in range(0, N, BATCH_SIZE):
    end = min(start + BATCH_SIZE, N)
    batch_ids = input_ids[start:end].to(device)
    B, S = batch_ids.shape[0], batch_ids.shape[1]
    x = tok_emb[batch_ids] + pos_emb[:S, :].unsqueeze(0)
    
    # Load block 0 attention weights (sync — first transfer has no overlap)
    w_attn = load_dict_gpu(block_attn[0])
    # Pre-load block 0 FFN weights async
    w_ffn_next = load_dict_gpu(block_ffn[0], t_stream)
    # Pre-load block 1 attention weights async
    w_attn_next = load_dict_gpu(block_attn[1], t_stream) if num_blocks > 1 else None
    
    for bid in range(num_blocks):
        wa = w_attn
        
        # ── Attention (using pre-loaded weights) ──
        residual = x
        if wa["ln1_w"] is not None:
            x = F.layer_norm(x.float(), [D], weight=wa["ln1_w"].float(),
                           bias=wa["ln1_b"].float() if wa["ln1_b"] is not None else None, eps=1e-5)
        qkv = x @ wa["qkv"].float()
        if wa["qkv_b"] is not None: qkv = qkv + wa["qkv_b"].float()
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, S, 32, 128).permute(0, 2, 1, 3)
        k = k.view(B, S, 32, 128).permute(0, 2, 1, 3)
        v = v.view(B, S, 32, 128).permute(0, 2, 1, 3)
        attn_out = (F.softmax((q @ k.transpose(-2, -1)) * (128 ** -0.5), dim=-1) @ v)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, 4096)
        attn_out = attn_out @ wa["proj"].float()
        if wa["proj_b"] is not None: attn_out = attn_out + wa["proj_b"].float()
        x = residual + attn_out
        del wa["qkv"], wa["proj"]
        
        # ── Wait for FFN weights (started loading during attention) ──
        t_stream.synchronize()
        wf = w_ffn_next
        
        # ── Kick off NEXT transfers while computing FFN ──
        if bid < num_blocks - 1:
            w_ffn_next = load_dict_gpu(block_ffn[bid + 1], t_stream)
            w_attn_next = load_dict_gpu(block_attn[bid + 1], t_stream)
        
        # ── FFN ──
        residual = x
        if wf["ln2_w"] is not None:
            x = F.layer_norm(x.float(), [D], weight=wf["ln2_w"].float(),
                           bias=wf["ln2_b"].float() if wf["ln2_b"] is not None else None, eps=1e-5)
        torch.backends.cuda.matmul.allow_tf32 = True
        x = x @ wf["ff1"].float()
        if wf["ff1_b"] is not None: x = x + wf["ff1_b"].float()
        x = F.gelu(x)
        x = x @ wf["ff2"].float()
        if wf["ff2_b"] is not None: x = x + wf["ff2_b"].float()
        torch.backends.cuda.matmul.allow_tf32 = False
        x = residual + x
        del wf["ff1"], wf["ff2"]
        
        # Swap for next iteration
        w_attn = w_attn_next
    
    # Final LN + Head
    if ln_f_w is not None:
        x = F.layer_norm(x.float(), [D], weight=ln_f_w, bias=ln_f_b, eps=1e-5)
    if head_w is not None:
        x = x.float() @ head_w.float()
        if head_b is not None: x = x + head_b
    all_logits.append(x.cpu().numpy())
    if start == 0: print(f"  First batch: {time.perf_counter()-t0:.1f}s", file=sys.stderr)

logits = np.concatenate(all_logits, axis=0).astype(np.float32)
if logits.ndim == 4: logits = logits.reshape(-1, logits.shape[-2], logits.shape[-1])
dt = time.perf_counter() - t0
golden = np.load(os.path.join(GOLDEN_DIR, "logits.npy"))
print(f"\nTime: {dt:.1f}s | Precision: {'PASS' if np.allclose(logits,golden,rtol=1e-3,atol=1e-3) else 'FAIL'} | MAX_DIFF: {np.max(np.abs(logits-golden)):.2e}", file=sys.stderr)
