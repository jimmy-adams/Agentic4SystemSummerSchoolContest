"""BigFormer v5: mmap streaming — ZERO upfront weight loading.

Eliminates the 42s ONNX weight load. Weights read directly from .data file
via numpy memmap on demand, per-block. Combined with pipeline + selective TF32.
"""
import onnx
import numpy as np
import torch
import torch.nn.functional as F
import time, os, sys, json
from collections import defaultdict

ONNX_PATH = "/workspace/C3/testcases/models/bigformer_v1.onnx"
DATA_PATH = ONNX_PATH + ".data"
INPUT_DIR = "/workspace/C3/testcases/testdata/c35/bigformer_v1/input"
GOLDEN_DIR = "/workspace/C3/testcases/testdata/c35/bigformer_v1/golden"
BATCH_SIZE = 16
device = torch.device("cuda")

print("=== mmap streaming ===")
t0 = time.perf_counter()

# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Parse graph (NO weight data loaded, 0.001s)
# ═══════════════════════════════════════════════════════════════════════
model = onnx.load(ONNX_PATH, load_external_data=False)

# Build offset table from external_data
DTYPE_MAP = {1: np.float32, 6: np.int32, 7: np.int64}
offset_table = {}
for init in model.graph.initializer:
    if init.data_location == onnx.TensorProto.EXTERNAL:
        ext = {e.key: e.value for e in init.external_data}
        offset_table[init.name] = {
            'offset': int(ext.get('offset', 0)),
            'length': int(ext.get('length', 0)),
            'dtype': DTYPE_MAP.get(init.data_type, np.float32),
            'shape': tuple(init.dims)
        }
    else:
        # Embedded (small tensors like biases)
        from onnx.numpy_helper import to_array
        arr = to_array(init)
        offset_table[init.name] = {
            'embedded': True,
            'data': torch.from_numpy(arr.copy()).pin_memory(),
            'dtype': arr.dtype,
            'shape': arr.shape
        }

# mmap the data file (no actual read yet)
data_mmap = np.memmap(DATA_PATH, dtype='uint8', mode='r')

def load_weight(name):
    """Load a single weight tensor from .data file on demand."""
    info = offset_table.get(name)
    if info is None: return None
    if info.get('embedded'):
        return info['data']  # already pinned
    raw = data_mmap[info['offset']:info['offset']+info['length']]
    arr = np.frombuffer(raw.tobytes(), dtype=info['dtype']).reshape(info['shape'])
    return torch.from_numpy(arr.copy()).pin_memory()

print(f"  Graph parsed + mmap ready: {time.perf_counter()-t0:.2f}s", file=sys.stderr)

# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Identity resolution + block mapping
# ═══════════════════════════════════════════════════════════════════════
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

# Build block→weight mapping (names only, no data loaded)
block_weights = {}
for node in model.graph.node:
    if node.op_type != "MatMul": continue
    parts = node.name.split("/")
    if len(parts) < 3 or not parts[1].startswith("blocks."):
        if "head" in node.name:
            for inp in node.input:
                r = resolve(inp)
                if r in offset_table: block_weights["head"] = r
        continue
    bid = int(parts[1].split(".")[1])
    sub = parts[2]
    if bid not in block_weights: block_weights[bid] = {}
    for inp in node.input:
        r = resolve(inp)
        if r in offset_table:
            block_weights[bid][sub] = r
            break

num_blocks = 24

# Build per-block weight NAME lists (no data loaded yet)
def resolve_param(pattern):
    r = resolve(pattern)
    return r if r in offset_table else None

block_cpu_names = []
for bid in range(num_blocks):
    bw = block_weights[bid]
    block_cpu_names.append({
        "qkv": bw["qkv"], "proj": bw["proj"], "ff1": bw["ff1"], "ff2": bw["ff2"],
        "qkv_b": resolve_param(f"blocks.{bid}.qkv.bias"),
        "proj_b": resolve_param(f"blocks.{bid}.proj.bias"),
        "ff1_b": resolve_param(f"blocks.{bid}.ff1.bias"),
        "ff2_b": resolve_param(f"blocks.{bid}.ff2.bias"),
        "ln1_w": resolve_param(f"blocks.{bid}.ln1.weight"),
        "ln1_b": resolve_param(f"blocks.{bid}.ln1.bias"),
        "ln2_w": resolve_param(f"blocks.{bid}.ln2.weight"),
        "ln2_b": resolve_param(f"blocks.{bid}.ln2.bias"),
    })

# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Load shared params only (tiny: embeddings, head)
# ═══════════════════════════════════════════════════════════════════════
def load_and_pin(name):
    t = load_weight(name)
    return t if t is not None else None

tok_emb = load_and_pin("tok_emb.weight")
pos_emb = load_and_pin("pos_emb")
head_w_name = block_weights.get("head")
head_w = load_and_pin(head_w_name) if head_w_name else None
head_b = load_and_pin("head.bias")
ln_f_w = load_and_pin("ln_f.weight")
ln_f_b = load_and_pin("ln_f.bias")

# Move to GPU (shared params are tiny)
if tok_emb is not None: tok_emb = tok_emb.float().to(device)
if pos_emb is not None: pos_emb = pos_emb.float().to(device)
if head_w is not None: head_w = head_w.float().to(device)
if head_b is not None: head_b = head_b.float().to(device)
if ln_f_w is not None: ln_f_w = ln_f_w.float().to(device)
if ln_f_b is not None: ln_f_b = ln_f_b.float().to(device)

print(f"  Shared params loaded: {time.perf_counter()-t0:.2f}s", file=sys.stderr)

# ═══════════════════════════════════════════════════════════════════════
# Phase 4: Input
# ═══════════════════════════════════════════════════════════════════════
with open(os.path.join(INPUT_DIR, "manifest.json")) as f:
    manifest = json.load(f)
input_data = np.load(os.path.join(INPUT_DIR, manifest["tensors"][0]["file"]))
input_ids = torch.from_numpy(input_data).long()
N = input_ids.shape[0]
D = tok_emb.shape[1]

# ═══════════════════════════════════════════════════════════════════════
# Phase 5: Inference — load weights on demand per block
# ═══════════════════════════════════════════════════════════════════════
print(f"=== Inference (mmap streaming) ===", file=sys.stderr)
compute_stream = torch.cuda.default_stream()
transfer_stream = torch.cuda.Stream()

def load_block_gpu(block_dict):
    """Load one block's weights from mmap to GPU."""
    result = {}
    for key in block_dict:
        name = block_dict[key]
        if name is None:
            result[key] = None
            continue
        t = load_weight(name)
        if t is not None:
            with torch.cuda.stream(transfer_stream):
                result[key] = t.to(device, non_blocking=True)
        else:
            result[key] = None
    return result

all_logits = []

for start in range(0, N, BATCH_SIZE):
    end = min(start + BATCH_SIZE, N)
    batch_ids = input_ids[start:end].to(device)
    B, S = batch_ids.shape[0], batch_ids.shape[1]
    
    x = tok_emb[batch_ids]
    x = x + pos_emb[:S, :].unsqueeze(0)
    
    # Load block 0 (sync — first block has no overlap)
    w_curr = {}
    for key in block_cpu_names[0]:
        name = block_cpu_names[0][key]
        w_curr[key] = load_weight(name).to(device) if name else None
    
    # Preload block 1 on transfer stream
    w_next = load_block_gpu(block_cpu_names[1]) if num_blocks > 1 else None
    
    for bid in range(num_blocks):
        w = w_curr
        
        # Kick off next block load while computing current
        if bid < num_blocks - 1:
            w_next = load_block_gpu(block_cpu_names[bid + 1])
        
        # ── Compute block N (identical to double-buffer version) ──
        residual = x
        if w["ln1_w"] is not None:
            x = F.layer_norm(x.float(), [D], weight=w["ln1_w"].float(),
                           bias=w["ln1_b"].float() if w["ln1_b"] is not None else None, eps=1e-5)
        
        qkv = x @ w["qkv"].float()
        if w["qkv_b"] is not None: qkv = qkv + w["qkv_b"].float()
        
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, S, 32, 128).permute(0, 2, 1, 3)
        k = k.view(B, S, 32, 128).permute(0, 2, 1, 3)
        v = v.view(B, S, 32, 128).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * (128 ** -0.5)
        attn = F.softmax(attn.float(), dim=-1)
        attn_out = attn @ v
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, 4096)
        
        attn_out = attn_out @ w["proj"].float()
        if w["proj_b"] is not None: attn_out = attn_out + w["proj_b"].float()
        x = residual + attn_out
        del w["qkv"], w["proj"]
        
        residual = x
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
        del w["ff1"], w["ff2"]
        
        # Wait for next block, swap
        if bid < num_blocks - 1:
            transfer_stream.synchronize()
            w_curr = w_next
    
    # Final LN + Head
    if ln_f_w is not None:
        x = F.layer_norm(x.float(), [D], weight=ln_f_w, bias=ln_f_b, eps=1e-5)
    if head_w is not None:
        x = x.float() @ head_w.float()
        if head_b is not None: x = x + head_b
    
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
