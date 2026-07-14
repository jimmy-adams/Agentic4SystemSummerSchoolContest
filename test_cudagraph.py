"""Test CUDA Graph: capture block computation, replay 24 times."""
import onnx, numpy as np, torch, torch.nn.functional as F, time, os, sys, json

BIGFORMER_PATH = "/workspace/C3/testcases/models/bigformer_v1.onnx"
INPUT_DIR = "/workspace/C3/testcases/testdata/c35/bigformer_v1/input"
GOLDEN_DIR = "/workspace/C3/testcases/testdata/c35/bigformer_v1/golden"
device = torch.device("cuda"); BATCH_SIZE = 16

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

print("=== Testing CUDA Graph ===")

# Test: can we capture a single block's computation as a graph?
# CUDA Graph requires static shapes and no CPU-GPU sync points.

w0 = blocks_fp16[0]
with open(os.path.join(INPUT_DIR, "manifest.json")) as f: manifest = json.load(f)
input_ids = torch.from_numpy(np.load(os.path.join(INPUT_DIR, manifest["tensors"][0]["file"]))).long()
batch = input_ids[:16].to(device); B, S = batch.shape
D = tok_emb.shape[1]

# Create static input/output tensors for graph capture
x_static = torch.zeros(B, S, D, device=device, dtype=torch.float32)
w_qkv_static = w0["qkv"].float()
w_proj_static = w0["proj"].float()
w_ff1_static = w0["ff1"].float()
w_ff2_static = w0["ff2"].float()
ln1w = w0["ln1_w"].float() if w0["ln1_w"] is not None else None
ln1b = w0["ln1_b"].float() if w0["ln1_b"] is not None else None  
ln2w = w0["ln2_w"].float() if w0["ln2_w"] is not None else None
ln2b = w0["ln2_b"].float() if w0["ln2_b"] is not None else None

# Warmup first
x = tok_emb[batch] + pos_emb[:, :S, :]
residual = x
if ln1w is not None: x = F.layer_norm(x, [D], weight=ln1w, bias=ln1b, eps=1e-5)
qkv = x @ w_qkv_static; q,k,v = qkv.chunk(3,dim=-1)
q=q.view(B,S,32,128).permute(0,2,1,3); k=k.view(B,S,32,128).permute(0,2,1,3); v=v.view(B,S,32,128).permute(0,2,1,3)
ao=(F.softmax((q@k.transpose(-2,-1))*(128**-0.5),dim=-1)@v).permute(0,2,1,3).reshape(B,S,4096)
ao = ao @ w_proj_static; x = residual + ao
residual = x
if ln2w is not None: x = F.layer_norm(x, [D], weight=ln2w, bias=ln2b, eps=1e-5)
x = x @ w_ff1_static; x = F.gelu(x); x = x @ w_ff2_static; x = residual + x
torch.cuda.synchronize()

# Try to capture
try:
    g = torch.cuda.CUDAGraph()
    x_static.copy_(x)  # initialize with real data
    
    with torch.cuda.graph(g):
        residual = x_static
        if ln1w is not None: x_out = F.layer_norm(x_static, [D], weight=ln1w, bias=ln1b, eps=1e-5)
        else: x_out = x_static
        qkv = x_out @ w_qkv_static
        q,k,v = qkv.chunk(3,dim=-1)
        q=q.view(B,S,32,128).permute(0,2,1,3); k=k.view(B,S,32,128).permute(0,2,1,3); v=v.view(B,S,32,128).permute(0,2,1,3)
        ao=(F.softmax((q@k.transpose(-2,-1))*(128**-0.5),dim=-1)@v).permute(0,2,1,3).reshape(B,S,4096)
        ao = ao @ w_proj_static
        x_mid = residual + ao
        residual2 = x_mid
        if ln2w is not None: x_mid = F.layer_norm(x_mid, [D], weight=ln2w, bias=ln2b, eps=1e-5)
        x_mid = x_mid @ w_ff1_static
        x_mid = F.gelu(x_mid)
        x_mid = x_mid @ w_ff2_static
        x_out_final = residual2 + x_mid
        x_static.copy_(x_out_final)
    
    print("  CUDA Graph captured successfully!")
    
    # Benchmark: eager vs graph
    N_RUNS = 100
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(N_RUNS):
        x_test = x.clone()
        r = x_test
        if ln1w is not None: x_test = F.layer_norm(x_test, [D], weight=ln1w, bias=ln1b, eps=1e-5)
        qkv = x_test @ w_qkv_static; q,k,v = qkv.chunk(3,dim=-1)
        q=q.view(B,S,32,128).permute(0,2,1,3); k=k.view(B,S,32,128).permute(0,2,1,3); v=v.view(B,S,32,128).permute(0,2,1,3)
        ao=(F.softmax((q@k.transpose(-2,-1))*(128**-0.5),dim=-1)@v).permute(0,2,1,3).reshape(B,S,4096)
        ao = ao @ w_proj_static; x_test = r + ao
        r = x_test
        if ln2w is not None: x_test = F.layer_norm(x_test, [D], weight=ln2w, bias=ln2b, eps=1e-5)
        x_test = x_test @ w_ff1_static; x_test = F.gelu(x_test); x_test = x_test @ w_ff2_static; x_test = r + x_test
    torch.cuda.synchronize()
    t_eager = (time.perf_counter()-t0)/N_RUNS*1000
    
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(N_RUNS):
        x_static.copy_(x)
        g.replay()
    torch.cuda.synchronize()
    t_graph = (time.perf_counter()-t0)/N_RUNS*1000
    
    print(f"  Eager: {t_eager:.2f}ms/block")
    print(f"  Graph: {t_graph:.2f}ms/block")
    print(f"  Speedup: {t_eager/t_graph:.1f}x")
    print(f"  Savings per block: {t_eager-t_graph:.1f}ms")
    print(f"  Total savings (768 blocks): {(t_eager-t_graph)*768/1000:.1f}s")
    
except Exception as e:
    print(f"  CUDA Graph FAILED: {e}")
    print("  (Graph requires static shapes and no dynamic allocs)")
