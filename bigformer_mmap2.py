"""BigFormer: mmap .data file directly → FP16 tensors. Skip ONNX to_array().

Key: .data file is sequential FP32 bytes. mmap as float32, then for each
initializer create a view at the right offset, convert to FP16, pin.
This avoids the 19s onnx.load() data parsing entirely.
"""
import onnx, numpy as np, torch, torch.nn.functional as F
import time, os, sys, json

ONNX_PATH = "/workspace/C3/testcases/models/bigformer_v1.onnx"
DATA_PATH = ONNX_PATH + ".data"
INPUT_DIR = "/workspace/C3/testcases/testdata/c35/bigformer_v1/input"
GOLDEN_DIR = "/workspace/C3/testcases/testdata/c35/bigformer_v1/golden"
BATCH_SIZE, device = 16, torch.device("cuda")

print("=== mmap direct → FP16 ===")
t0 = time.perf_counter()

# ── Parse graph only (0.001s) ──
model = onnx.load(ONNX_PATH, load_external_data=False)

identity_map = {n.output[0]: n.input[0] for n in model.graph.node if n.op_type == "Identity"}
def resolve(name):
    visited = set()
    while name in identity_map and name not in visited:
        visited.add(name); name = identity_map[name]
    return name

# ── mmap the entire .data file as float32 ──
total_bytes = os.path.getsize(DATA_PATH)
total_floats = total_bytes // 4
data_mmap = np.memmap(DATA_PATH, dtype=np.float32, mode='r', shape=(total_floats,))
print(f"  mmap ready: {total_bytes/1e9:.1f}GB in {time.perf_counter()-t0:.2f}s")

# ── Build offset table from ONNX without loading data ──
offset_table = {}
for init in model.graph.initializer:
    if init.data_location == onnx.TensorProto.EXTERNAL:
        ext = {e.key: e.value for e in init.external_data}
        offset = int(ext['offset']) // 4  # byte offset → float32 index
        length = int(ext['length']) // 4
        shape = tuple(init.dims)
        offset_table[init.name] = (offset, length, shape)
    else:
        from onnx.numpy_helper import to_array
        arr = to_array(init)
        offset_table[init.name] = ('embedded', arr)

init_names = set(offset_table.keys())

def load_weight_fp16(name):
    """Load a single weight as FP16 from mmap."""
    info = offset_table.get(name)
    if info is None: return None
    if info[0] == 'embedded':
        arr = info[1]
        return torch.from_numpy(arr.copy()).half().pin_memory()
    offset, length, shape = info
    # Read from mmap at offset, convert to FP16
    arr = np.array(data_mmap[offset:offset+length], dtype=np.float32).reshape(shape)
    return torch.from_numpy(arr).half().pin_memory()

def rp(name):
    r = resolve(name)
    return r if r in init_names else None

# ── Block mapping ──
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

# ── Load all weights as FP16 from mmap ──
print(f"  Loading weights from mmap...")
loaded = {}
for name in offset_table:
    if name not in loaded:
        loaded[name] = load_weight_fp16(name)

total_gb = sum(t.numel()*t.element_size() for t in loaded.values()) / 1e9
print(f"  Loaded: {total_gb:.1f}GB (FP16) in {time.perf_counter()-t0:.1f}s")

# ── Move blocks to GPU ──
blocks_gpu = []
for bid in range(num_blocks):
    bw = block_weights[bid]
    def g(n):
        name = rp(f"blocks.{bid}.{n}")
        t = loaded.get(name) if name else None
        return t.to(device) if t is not None else None
    
    blocks_gpu.append({
        "qkv": loaded[bw["qkv"]].to(device),
        "proj": loaded[bw["proj"]].to(device),
        "ff1": loaded[bw["ff1"]].to(device),
        "ff2": loaded[bw["ff2"]].to(device),
        "qkv_b": g("qkv.bias"), "proj_b": g("proj.bias"),
        "ff1_b": g("ff1.bias"), "ff2_b": g("ff2.bias"),
        "ln1_w": g("ln1.weight"), "ln1_b": g("ln1.bias"),
        "ln2_w": g("ln2.weight"), "ln2_b": g("ln2.bias"),
    })

# Shared
tok_emb = loaded.get(rp("tok_emb.weight"))
pos_emb = loaded.get(rp("pos_emb"))
head_w_name = block_weights.get("head")
head_w = loaded.get(head_w_name).to(device) if head_w_name else None
head_b = loaded.get(rp("head.bias"))
ln_f_w = loaded.get(rp("ln_f.weight"))
ln_f_b = loaded.get(rp("ln_f.bias"))

for t in [tok_emb, pos_emb, head_b, ln_f_w, ln_f_b]:
    if t is not None: t.data = t.float().to(device)
if tok_emb is not None: tok_emb = tok_emb.float().to(device)
if pos_emb is not None: pos_emb = pos_emb.float().to(device)

print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
print(f"  Total load: {time.perf_counter()-t0:.1f}s")

# Input
with open(os.path.join(INPUT_DIR, "manifest.json")) as f: manifest = json.load(f)
input_ids = torch.from_numpy(np.load(os.path.join(INPUT_DIR, manifest["tensors"][0]["file"]))).long()
N, D = input_ids.shape[0], tok_emb.shape[1]

print(f"=== Inference ===")
all_logits = []

for start in range(0, N, BATCH_SIZE):
    end = min(start + BATCH_SIZE, N)
    batch_ids = input_ids[start:end].to(device)
    B, S = batch_ids.shape[0], batch_ids.shape[1]
    x = tok_emb[batch_ids] + pos_emb[:S, :].unsqueeze(0)
    
    for bid in range(num_blocks):
        w = blocks_gpu[bid]
        
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
