"""Profile BigFormer inference in worker context."""
import onnx, numpy as np, torch, torch.nn.functional as F, time, os, sys, json

BIGFORMER_PATH = "/workspace/C3/testcases/models/bigformer_v1.onnx"
INPUT_DIR = "/workspace/C3/testcases/testdata/c35/bigformer_v1/input"
device = torch.device("cuda")

# ── Fast load (copy from worker) ──
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
    arr = np.array(data_mmap[off:off+length], dtype=np.float32).reshape(shape)
    return torch.from_numpy(arr).half()

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

# Pre-load to GPU
blocks = []
for bid in range(24):
    bw = block_weights[bid]
    def g(n):
        name = rp(f"blocks.{bid}.{n}")
        t = load_fp16(name) if name else None
        return t.to(device) if t is not None else None
    blocks.append({
        "qkv": load_fp16(bw["qkv"]).to(device), "proj": load_fp16(bw["proj"]).to(device),
        "ff1": load_fp16(bw["ff1"]).to(device), "ff2": load_fp16(bw["ff2"]).to(device),
        "qkv_b": g("qkv.bias"), "proj_b": g("proj.bias"), "ff1_b": g("ff1.bias"), "ff2_b": g("ff2.bias"),
        "ln1_w": g("ln1.weight"), "ln1_b": g("ln1.bias"), "ln2_w": g("ln2.weight"), "ln2_b": g("ln2.bias"),
    })

tok_emb = load_fp16(rp("tok_emb.weight")).float().to(device)
pos_emb = load_fp16(rp("pos_emb")).float().to(device)
head_w = load_fp16(block_weights["head"]).to(device) if "head" in block_weights else None
head_b = load_fp16(rp("head.bias")).float().to(device) if rp("head.bias") else None
ln_f_w = load_fp16(rp("ln_f.weight")).float().to(device) if rp("ln_f.weight") else None
ln_f_b = load_fp16(rp("ln_f.bias")).float().to(device) if rp("ln_f.bias") else None

with open(os.path.join(INPUT_DIR, "manifest.json")) as f: manifest = json.load(f)
input_ids = torch.from_numpy(np.load(os.path.join(INPUT_DIR, manifest["tensors"][0]["file"]))).long()
N, D = 512, tok_emb.shape[1]

print("=== Per-block timing ===")
torch.cuda.synchronize()

# Warmup one full forward
batch = input_ids[:16].to(device); B, S = batch.shape
x = tok_emb[batch] + pos_emb[:, :S, :]
for w in blocks:
    residual = x.float()
    x = F.layer_norm(x.float(), [D], weight=w["ln1_w"].float(), bias=w["ln1_b"].float() if w["ln1_b"] is not None else None, eps=1e-5)
    qkv = x @ w["qkv"].float(); q,k,v = qkv.chunk(3,dim=-1)
    q=q.view(B,S,32,128).permute(0,2,1,3); k=k.view(B,S,32,128).permute(0,2,1,3); v=v.view(B,S,32,128).permute(0,2,1,3)
    attn_out = (F.softmax((q@k.transpose(-2,-1))*(128**-0.5),dim=-1)@v).permute(0,2,1,3).reshape(B,S,4096)
    x = residual + attn_out @ w["proj"].float()
    residual = x.float()
    x = F.layer_norm(x.float(),[D],weight=w["ln2_w"].float(),bias=w["ln2_b"].float() if w["ln2_b"] is not None else None,eps=1e-5)
    x = F.gelu(x @ w["ff1"].float()) @ w["ff2"].float()
    x = residual + x
torch.cuda.synchronize()

# Time per block
timings = {"qkv": [], "attn": [], "proj": [], "ln2+ff1": [], "gelu": [], "ff2": [], "total": []}

for run in range(3):
    batch = input_ids[run*16:(run+1)*16].to(device); B, S = batch.shape
    x = tok_emb[batch] + pos_emb[:, :S, :]
    
    for bid in range(24):
        w = blocks[bid]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        
        residual = x.float()
        x = F.layer_norm(x.float(), [D], weight=w["ln1_w"].float(), bias=w["ln1_b"].float() if w["ln1_b"] is not None else None, eps=1e-5)
        
        t1 = time.perf_counter()
        qkv = x @ w["qkv"].float()
        q,k,v = qkv.chunk(3,dim=-1)
        q=q.view(B,S,32,128).permute(0,2,1,3); k=k.view(B,S,32,128).permute(0,2,1,3); v=v.view(B,S,32,128).permute(0,2,1,3)
        torch.cuda.synchronize(); t2 = time.perf_counter()
        
        attn_out = (F.softmax((q@k.transpose(-2,-1))*(128**-0.5),dim=-1)@v)
        attn_out = attn_out.permute(0,2,1,3).reshape(B,S,4096)
        torch.cuda.synchronize(); t3 = time.perf_counter()
        
        attn_out = attn_out @ w["proj"].float()
        x = residual + attn_out
        torch.cuda.synchronize(); t4 = time.perf_counter()
        
        residual = x.float()
        x = F.layer_norm(x.float(),[D],weight=w["ln2_w"].float(),bias=w["ln2_b"].float() if w["ln2_b"] is not None else None,eps=1e-5)
        
        t5 = time.perf_counter()
        ffn1 = x @ w["ff1"].float()
        torch.cuda.synchronize(); t6 = time.perf_counter()
        
        x = F.gelu(ffn1)
        torch.cuda.synchronize(); t7 = time.perf_counter()
        
        x = x @ w["ff2"].float()
        torch.cuda.synchronize(); t8 = time.perf_counter()
        
        x = residual + x
        torch.cuda.synchronize(); t9 = time.perf_counter()
        
        timings["qkv"].append((t2-t1)*1000)
        timings["attn"].append((t3-t2)*1000)
        timings["proj"].append((t4-t3)*1000)
        timings["ln2+ff1"].append((t6-t5)*1000)
        timings["gelu"].append((t7-t6)*1000)
        timings["ff2"].append((t8-t7)*1000)
        timings["total"].append((t9-t0)*1000)

print(f"\n{'Op':<20s} {'Avg(ms)':<10s} {'Sum/24blocks':<15s} {'Pct':<10s}")
print("-" * 55)
total_per_block = sum(sum(v)/len(v) for v in timings.values() if v) / 3
for key in ["qkv", "attn", "proj", "ln2+ff1", "gelu", "ff2"]:
    avg = sum(timings[key]) / len(timings[key])
    total_24 = avg * 24 / 1000
    pct = avg / (sum(sum(timings[k])/len(timings[k]) for k in timings if k != "total") / 3) * 100
    print(f"  {key:<18s} {avg:>8.1f}ms  {total_24:>12.1f}s  {pct:>5.1f}%")

avg_total = sum(timings["total"]) / len(timings["total"])
print(f"\n  Per-block total: {avg_total:.1f}ms")
print(f"  24 blocks: {avg_total*24/1000:.1f}s")
print(f"  32 batches: {avg_total*24*32/1000:.0f}s  (estimated total compute)")

# Also count .float() conversions
print(f"\n=== .float() conversions per block ===")
# In current loop: ln1_w.float(), ln1_b.float(), w["qkv"].float(), w["proj"].float(), 
#                  ln2_w.float(), ln2_b.float(), w["ff1"].float(), w["ff2"].float()
# = 8 conversions per block × 24 blocks × 32 batches = 6144 conversions
print(f"  8 weight .float() calls per block")
print(f"  24 blocks × 32 batches = 6144 FP16→FP32 conversions total")
print(f"  Each ~804MB/4 weights ≈ 200MB copied per conversion")
