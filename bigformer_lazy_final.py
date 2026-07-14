"""BigFormer v7 final: Lazy loading — sync per-block, no pipeline complexity.

Loads only block 0 upfront (1.2s vs 45s). Remaining blocks loaded on demand.
"""
import onnx, numpy as np, torch, torch.nn.functional as F
import time, os, sys, json

ONNX_PATH = "/workspace/C3/testcases/models/bigformer_v1.onnx"
DATA_PATH = ONNX_PATH + ".data"
INPUT_DIR = "/workspace/C3/testcases/testdata/c35/bigformer_v1/input"
GOLDEN_DIR = "/workspace/C3/testcases/testdata/c35/bigformer_v1/golden"
BATCH_SIZE, device = 16, torch.device("cuda")

print("=== Lazy loading ===")
t0 = time.perf_counter()

model = onnx.load(ONNX_PATH, load_external_data=False)
identity_map = {n.output[0]: n.input[0] for n in model.graph.node if n.op_type == "Identity"}
def resolve(name):
    visited = set()
    while name in identity_map and name not in visited:
        visited.add(name); name = identity_map[name]
    return name

DTYPES = {1: np.float32, 6: np.int32, 7: np.int64}
offset_table = {}
for init in model.graph.initializer:
    if init.data_location == onnx.TensorProto.EXTERNAL:
        ext = {e.key: e.value for e in init.external_data}
        offset_table[init.name] = {'offset': int(ext['offset']), 'length': int(ext['length']),
                                   'dtype': DTYPES.get(init.data_type, np.float32), 'shape': tuple(init.dims)}
    else:
        from onnx.numpy_helper import to_array
        offset_table[init.name] = {'embedded': True, 'data': torch.from_numpy(to_array(init).copy()).pin_memory()}

data_mmap = np.memmap(DATA_PATH, dtype='uint8', mode='r')
init_names = set(offset_table.keys())

def load_one(name):
    info = offset_table.get(name)
    if info is None: return None
    if info.get('embedded'): return info['data']
    raw = data_mmap[info['offset']:info['offset']+info['length']]
    arr = np.frombuffer(raw.tobytes(), dtype=info['dtype']).reshape(info['shape'])
    return torch.from_numpy(arr.copy()).pin_memory()

def rp(name):
    r = resolve(name)
    return r if r in init_names else None

def load_block(names_dict):
    return {k: load_one(v).to(device) if v else None for k, v in names_dict.items()}

# Block mapping
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
block_names = []
for bid in range(num_blocks):
    bw = block_weights[bid]
    def g(n): return rp(f"blocks.{bid}.{n}")
    block_names.append({
        "qkv": bw["qkv"], "proj": bw["proj"], "ff1": bw["ff1"], "ff2": bw["ff2"],
        "qkv_b": g("qkv.bias"), "proj_b": g("proj.bias"), "ff1_b": g("ff1.bias"), "ff2_b": g("ff2.bias"),
        "ln1_w": g("ln1.weight"), "ln1_b": g("ln1.bias"), "ln2_w": g("ln2.weight"), "ln2_b": g("ln2.bias"),
    })

# Shared params
def ld(name):
    t = load_one(rp(name))
    return t.float().to(device) if t is not None else None

tok_emb = ld("tok_emb.weight")
pos_emb = ld("pos_emb")
head_w = load_one(block_weights["head"]).float().to(device) if "head" in block_weights else None
head_b = load_one(rp("head.bias")).float().to(device) if rp("head.bias") else None
ln_f_w = load_one(rp("ln_f.weight")).float().to(device) if rp("ln_f.weight") else None
ln_f_b = load_one(rp("ln_f.bias")).float().to(device) if rp("ln_f.bias") else None

print(f"  Graph + shared: {time.perf_counter()-t0:.1f}s")
print(f"  Loading block 0...")
w_curr = load_block(block_names[0])
print(f"  Ready: {time.perf_counter()-t0:.1f}s")

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
    
    w_curr = load_block(block_names[0])
    
    for bid in range(num_blocks):
        w = w_curr
        if bid < num_blocks - 1:
            w_next = load_block(block_names[bid + 1])
        
        # Attention
        residual = x
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
        
        # FFN
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
        
        if bid < num_blocks - 1:
            del w_curr
            w_curr = w_next
    
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
