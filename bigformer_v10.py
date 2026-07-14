"""BigFormer v10: micro-batch 64 + TF32 FFN-only on first 18 blocks."""
import onnx, numpy as np, torch, torch.nn.functional as F, time, os, sys, json

BIGFORMER_PATH = "/workspace/C3/testcases/models/bigformer_v1.onnx"
INPUT_DIR = "/workspace/C3/testcases/testdata/c35/bigformer_v1/input"
GOLDEN_DIR = "/workspace/C3/testcases/testdata/c35/bigformer_v1/golden"
device = torch.device("cuda")

print("=== v10: inner=64, TF32 FFN-only on first 18 blocks ===")
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

with open(os.path.join(INPUT_DIR, "manifest.json")) as f: manifest = json.load(f)
input_ids = torch.from_numpy(np.load(os.path.join(INPUT_DIR, manifest["tensors"][0]["file"]))).long()
N, D = 512, tok_emb.shape[1]
INNER = 64
TF32_BLOCKS = 18

print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB  Load: {time.perf_counter()-t0:.1f}s")

xs = []
for s in range(0, N, INNER):
    e = min(s+INNER, N); batch = input_ids[s:e].to(device)
    xs.append(tok_emb[batch] + pos_emb[:, :batch.shape[1], :])

torch.cuda.reset_peak_memory_stats()

for bid, w in enumerate(blocks_fp16):
    use_tf32 = bid < TF32_BLOCKS
    
    w_qkv=w["qkv"].float();w_proj=w["proj"].float();w_ff1=w["ff1"].float();w_ff2=w["ff2"].float()
    l1w=w["ln1_w"].float() if w["ln1_w"] is not None else None; l1b=w["ln1_b"].float() if w["ln1_b"] is not None else None
    l2w=w["ln2_w"].float() if w["ln2_w"] is not None else None; l2b=w["ln2_b"].float() if w["ln2_b"] is not None else None
    qb=w["qkv_b"].float() if w["qkv_b"] is not None else None; pb=w["proj_b"].float() if w["proj_b"] is not None else None
    f1b=w["ff1_b"].float() if w["ff1_b"] is not None else None; f2b=w["ff2_b"].float() if w["ff2_b"] is not None else None
    
    for si, x in enumerate(xs):
        B, S = x.shape[0], x.shape[1]
        r=x; x=F.layer_norm(x,[D],weight=l1w,bias=l1b,eps=1e-5) if l1w is not None else x
        qkv=x@w_qkv; qkv=qkv+qb if qb is not None else qkv
        q,k,v=qkv.chunk(3,dim=-1); q=q.view(B,S,32,128).permute(0,2,1,3); k=k.view(B,S,32,128).permute(0,2,1,3); v=v.view(B,S,32,128).permute(0,2,1,3)
        ao=(F.softmax((q@k.transpose(-2,-1))*(128**-0.5),dim=-1)@v).permute(0,2,1,3).reshape(B,S,4096)
        ao=ao@w_proj; ao=ao+pb if pb is not None else ao; x=r+ao
        r=x; x=F.layer_norm(x,[D],weight=l2w,bias=l2b,eps=1e-5) if l2w is not None else x
        if use_tf32: torch.backends.cuda.matmul.allow_tf32 = True
        x=x@w_ff1; x=x+f1b if f1b is not None else x; x=F.gelu(x)
        x=x@w_ff2; x=x+f2b if f2b is not None else x
        if use_tf32: torch.backends.cuda.matmul.allow_tf32 = False
        x=r+x
        xs[si]=x
    
    del w_qkv,w_proj,w_ff1,w_ff2

for si, x in enumerate(xs):
    x=F.layer_norm(x,[D],weight=ln_f_w,bias=ln_f_b,eps=1e-5) if ln_f_w is not None else x
    x=x@head_w.float()+(head_b if head_b is not None else 0) if head_w is not None else x
    xs[si]=x.cpu().numpy()

peak_gb = torch.cuda.max_memory_allocated() / 1e9
logits = np.concatenate(xs, axis=0).astype(np.float32)
dt = time.perf_counter() - t0
golden = np.load(os.path.join(GOLDEN_DIR, "logits.npy"))
ok = np.allclose(logits, golden, rtol=1e-3, atol=1e-3)
print(f"\nTime: {dt:.1f}s | Peak: {peak_gb:.1f}GB | {'PASS' if ok else 'FAIL'} | MAX_DIFF: {np.max(np.abs(logits-golden)):.2e}")
