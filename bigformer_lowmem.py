"""BigFormer: serialize FP16→FP32 conversions to minimize peak GPU memory.

Instead of converting all 4 weights at once (804MB peak), convert one at a time
and free immediately after use. Peak FP32 memory: max(268MB) instead of 804MB.
"""
import onnx, numpy as np, torch, torch.nn.functional as F, time, os, sys, json

BIGFORMER_PATH = "/workspace/C3/testcases/models/bigformer_v1.onnx"
INPUT_DIR = "/workspace/C3/testcases/testdata/c35/bigformer_v1/input"
GOLDEN_DIR = "/workspace/C3/testcases/testdata/c35/bigformer_v1/golden"
device = torch.device("cuda"); BATCH_SIZE = 16

print("=== Low-memory: serialized FP32 conversion ===")
t0 = time.perf_counter()

model = onnx.load(BIGFORMER_PATH, load_external_data=False)
identity_map = {n.output[0]: n.input[0] for n in model.graph.node if n.op_type == "Identity"}
def resolve(name):
    visited = set()
    while name in identity_map and name not in visited:
        visited.add(name); name = identity_map[name]
    return name

data_path = BIGFORMER_PATH + ".data"
data_mmap = np.memmap(data_path, dtype=np.float32, mode='r', shape=(os.path.getsize(data_path)//4,))
offset_table = {}
for init in model.graph.initializer:
    if init.data_location == onnx.TensorProto.EXTERNAL:
        ext = {e.key: e.value for e in init.external_data}
        offset_table[init.name] = (int(ext['offset'])//4, int(ext['length'])//4, tuple(init.dims))
    else:
        offset_table[init.name] = ('embedded', onnx.numpy_helper.to_array(init))

def load_fp16(name):
    info = offset_table.get(name)
    if info is None: return None
    if info[0] == 'embedded': return torch.from_numpy(info[1].copy()).half()
    off, length, shape = info
    return torch.from_numpy(np.array(data_mmap[off:off+length], dtype=np.float32).reshape(shape)).half()

init_names = set(offset_table.keys())
def rp(name):
    r = resolve(name)
    return r if r in init_names else None

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

# Pre-load FP16 to GPU
blocks_fp16 = []
for bid in range(24):
    bw = block_weights[bid]
    def g(n):
        name = rp(f"blocks.{bid}.{n}")
        t = load_fp16(name) if name else None
        return t.to(device) if t is not None else None
    blocks_fp16.append({
        "qkv": load_fp16(bw["qkv"]).to(device), "proj": load_fp16(bw["proj"]).to(device),
        "ff1": load_fp16(bw["ff1"]).to(device), "ff2": load_fp16(bw["ff2"]).to(device),
        "qkv_b": g("qkv.bias"), "proj_b": g("proj.bias"), "ff1_b": g("ff1.bias"), "ff2_b": g("ff2.bias"),
        "ln1_w": g("ln1.weight"), "ln1_b": g("ln1.bias"), "ln2_w": g("ln2.weight"), "ln2_b": g("ln2.bias"),
    })

tok_emb = load_fp16(rp("tok_emb.weight")).float().to(device); pos_emb = load_fp16(rp("pos_emb")).float().to(device)
head_w = load_fp16(block_weights["head"]).to(device) if "head" in block_weights else None
head_b = load_fp16(rp("head.bias")).float().to(device) if rp("head.bias") else None
ln_f_w = load_fp16(rp("ln_f.weight")).float().to(device) if rp("ln_f.weight") else None
ln_f_b = load_fp16(rp("ln_f.bias")).float().to(device) if rp("ln_f.bias") else None

print(f"  Base GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")

with open(os.path.join(INPUT_DIR, "manifest.json")) as f: manifest = json.load(f)
input_ids = torch.from_numpy(np.load(os.path.join(INPUT_DIR, manifest["tensors"][0]["file"]))).long()
N, D = 512, tok_emb.shape[1]

print(f"=== Inference (serialized conversions) ===")
all_logits = []
torch.cuda.reset_peak_memory_stats()

for start in range(0, N, BATCH_SIZE):
    end = min(start + BATCH_SIZE, N)
    batch = input_ids[start:end].to(device); B, S = batch.shape
    x = tok_emb[batch] + pos_emb[:, :S, :]
    
    for bid in range(24):
        w = blocks_fp16[bid]
        
        # ── Convert QKV ONLY, use, free ──
        w_qkv = w["qkv"].float()
        
        residual = x
        if w["ln1_w"] is not None:
            x = F.layer_norm(x.float(), [D], weight=w["ln1_w"].float(), bias=w["ln1_b"].float() if w["ln1_b"] is not None else None, eps=1e-5)
        
        qkv = x @ w_qkv
        if w["qkv_b"] is not None: qkv = qkv + w["qkv_b"].float()
        del w_qkv  # free QKV FP32 immediately
        
        q,k,v = qkv.chunk(3,dim=-1)
        q=q.view(B,S,32,128).permute(0,2,1,3); k=k.view(B,S,32,128).permute(0,2,1,3); v=v.view(B,S,32,128).permute(0,2,1,3)
        attn_out = (F.softmax((q@k.transpose(-2,-1))*(128**-0.5),dim=-1)@v).permute(0,2,1,3).reshape(B,S,4096)
        
        # ── Convert Proj ONLY, use, free ──
        w_proj = w["proj"].float()
        attn_out = attn_out @ w_proj
        if w["proj_b"] is not None: attn_out = attn_out + w["proj_b"].float()
        del w_proj
        
        x = residual + attn_out
        
        # ── LN2 ──
        residual = x
        if w["ln2_w"] is not None:
            x = F.layer_norm(x.float(), [D], weight=w["ln2_w"].float(), bias=w["ln2_b"].float() if w["ln2_b"] is not None else None, eps=1e-5)
        
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # ── Convert FF1 ONLY, use, free before FF2 ──
        w_ff1 = w["ff1"].float()
        x = x @ w_ff1
        if w["ff1_b"] is not None: x = x + w["ff1_b"].float()
        x = F.gelu(x)
        del w_ff1  # free FF1 before converting FF2
        
        # ── Convert FF2 ONLY, use, free ──
        w_ff2 = w["ff2"].float()
        x = x @ w_ff2
        if w["ff2_b"] is not None: x = x + w["ff2_b"].float()
        del w_ff2
        
        torch.backends.cuda.matmul.allow_tf32 = False
        x = residual + x
    
    x = F.layer_norm(x.float(), [D], weight=ln_f_w, bias=ln_f_b, eps=1e-5) if ln_f_w is not None else x
    x = x.float() @ head_w.float() + (head_b if head_b is not None else 0) if head_w is not None else x
    all_logits.append(x.cpu().numpy())
    if start == 0: print(f"  First batch: {time.perf_counter()-t0:.1f}s")

peak_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"  Peak GPU memory: {peak_gb:.1f}GB")

logits = np.concatenate(all_logits, axis=0).astype(np.float32)
if logits.ndim == 4: logits = logits.reshape(-1, logits.shape[-2], logits.shape[-1])
dt = time.perf_counter() - t0
golden = np.load(os.path.join(GOLDEN_DIR, "logits.npy"))
ok = np.allclose(logits, golden, rtol=1e-3, atol=1e-3)
print(f"\nTime: {dt:.1f}s | Peak: {peak_gb:.1f}GB | Precision: {'PASS' if ok else 'FAIL'} | MAX_DIFF: {np.max(np.abs(logits-golden)):.2e}")
