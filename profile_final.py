"""Final optimization check: profile every operation in the inference loop."""
import onnx, numpy as np, torch, torch.nn.functional as F, time, os, sys, json

BIGFORMER_PATH = "/workspace/C3/testcases/models/bigformer_v1.onnx"
INPUT_DIR = "/workspace/C3/testcases/testdata/c35/bigformer_v1/input"
device = torch.device("cuda")

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
D = tok_emb.shape[1]

print("=== Line-by-line profile ===")
torch.cuda.synchronize()

batch = input_ids[:16].to(device); B, S = batch.shape
x = tok_emb[batch] + pos_emb[:, :S, :]
N_RUNS = 50

# Warmup
w = blocks_fp16[0]
w_qkv=w["qkv"].float();w_proj=w["proj"].float();w_ff1=w["ff1"].float();w_ff2=w["ff2"].float()
r=x;xn=F.layer_norm(x,[D],weight=w["ln1_w"].float(),bias=w["ln1_b"].float() if w["ln1_b"] is not None else None,eps=1e-5)
qkv=xn@w_qkv;q,k,v=qkv.chunk(3,dim=-1);q=q.view(B,S,32,128).permute(0,2,1,3);k=k.view(B,S,32,128).permute(0,2,1,3);v=v.view(B,S,32,128).permute(0,2,1,3)
ao=(F.softmax((q@k.transpose(-2,-1))*(128**-0.5),dim=-1)@v).permute(0,2,1,3).reshape(B,S,4096)
x=r+ao@w_proj;r=x;x=F.layer_norm(x,[D],weight=w["ln2_w"].float(),bias=w["ln2_b"].float() if w["ln2_b"] is not None else None,eps=1e-5)
x=x@w_ff1;x=F.gelu(x);x=x@w_ff2;x=r+x
torch.cuda.synchronize()

events = {}
w_qkv=w["qkv"].float();w_proj=w["proj"].float();w_ff1=w["ff1"].float();w_ff2=w["ff2"].float()

# FP16→FP32
torch.cuda.synchronize();t0=time.perf_counter()
for _ in range(N_RUNS): wq=w["qkv"].float();wp=w["proj"].float();wf1=w["ff1"].float();wf2=w["ff2"].float()
torch.cuda.synchronize()
events['FP16->FP32 x4'] = (time.perf_counter()-t0)/N_RUNS*1000

# LN1
ln1w=w["ln1_w"].float();ln1b=w["ln1_b"].float() if w["ln1_b"] is not None else None
torch.cuda.synchronize();t0=time.perf_counter()
for _ in range(N_RUNS): _ = F.layer_norm(x, [D], weight=ln1w, bias=ln1b, eps=1e-5)
torch.cuda.synchronize()
events['LayerNorm'] = (time.perf_counter()-t0)/N_RUNS*1000

xn = F.layer_norm(x, [D], weight=ln1w, bias=ln1b, eps=1e-5)

# QKV matmul
torch.cuda.synchronize();t0=time.perf_counter()
for _ in range(N_RUNS): _ = xn @ w_qkv
torch.cuda.synchronize()
events['QKV MatMul'] = (time.perf_counter()-t0)/N_RUNS*1000

qkv = xn @ w_qkv
q,k,v = qkv.chunk(3,dim=-1)
q=q.view(B,S,32,128).permute(0,2,1,3);k=k.view(B,S,32,128).permute(0,2,1,3);v=v.view(B,S,32,128).permute(0,2,1,3)

# Attention
torch.cuda.synchronize();t0=time.perf_counter()
for _ in range(N_RUNS): _ = (F.softmax((q@k.transpose(-2,-1))*(128**-0.5),dim=-1)@v)
torch.cuda.synchronize()
events['Attention(SD)'] = (time.perf_counter()-t0)/N_RUNS*1000

ao = (F.softmax((q@k.transpose(-2,-1))*(128**-0.5),dim=-1)@v).permute(0,2,1,3).reshape(B,S,4096)

# Proj matmul
torch.cuda.synchronize();t0=time.perf_counter()
for _ in range(N_RUNS): _ = ao @ w_proj
torch.cuda.synchronize()
events['Proj MatMul'] = (time.perf_counter()-t0)/N_RUNS*1000

# LN2 + FF1
ln2w=w["ln2_w"].float();ln2b=w["ln2_b"].float() if w["ln2_b"] is not None else None
xn2 = F.layer_norm((x+ao@w_proj), [D], weight=ln2w, bias=ln2b, eps=1e-5)
torch.cuda.synchronize();t0=time.perf_counter()
for _ in range(N_RUNS):
    _ = xn2 @ w_ff1; _ = F.gelu(_); _ = _ @ w_ff2
torch.cuda.synchronize()
events['FF1+GELU+FF2'] = (time.perf_counter()-t0)/N_RUNS*1000

# FF1 alone
torch.cuda.synchronize();t0=time.perf_counter()
for _ in range(N_RUNS): _ = xn2 @ w_ff1
torch.cuda.synchronize()
events['FF1 MatMul'] = (time.perf_counter()-t0)/N_RUNS*1000

# FF2 alone
ff1_out = xn2 @ w_ff1; gelu_out = F.gelu(ff1_out)
torch.cuda.synchronize();t0=time.perf_counter()
for _ in range(N_RUNS): _ = gelu_out @ w_ff2
torch.cuda.synchronize()
events['FF2 MatMul'] = (time.perf_counter()-t0)/N_RUNS*1000

# GELU
torch.cuda.synchronize();t0=time.perf_counter()
for _ in range(N_RUNS): _ = F.gelu(ff1_out)
torch.cuda.synchronize()
events['GELU'] = (time.perf_counter()-t0)/N_RUNS*1000

total_ms = sum(v for v in events.values())
print(f"\n{'Operation':<28s} {'ms':<8s} {'x24':<8s} {'%':<6s}")
print("-" * 52)
for key, val in sorted(events.items(), key=lambda x: -x[1]):
    print(f"  {key:<26s} {val:>6.2f}  {val*24:>6.0f}  {val/total_ms*100:>5.1f}")
print(f"  {'Per block TOTAL':<26s} {total_ms:>6.2f}  {total_ms*24:>6.0f}")
print(f"  {'32 batches TOTAL':<26s} {'':<6s}  {total_ms*24*32/1000:>6.0f}s")
