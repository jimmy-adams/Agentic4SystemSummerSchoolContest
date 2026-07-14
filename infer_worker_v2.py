"""C3.5 Persistent Worker v2 — ORT for normal models, mmap+GPU for BigFormer."""
import json, os, sys, time, traceback, numpy as np, onnx, onnxruntime as ort, torch, torch.nn.functional as F

print("READY", flush=True)
print("[worker] Ready", file=sys.stderr)

session_cache = {}

def load_inputs(input_dir):
    with open(os.path.join(input_dir, "manifest.json")) as f:
        manifest = json.load(f)
    return {e["name"]: np.load(os.path.join(input_dir, e["file"])) for e in manifest["tensors"]}

def write_outputs(output_dir, outputs):
    os.makedirs(output_dir, exist_ok=True)
    manifest = {"tensors": []}
    for name, data in outputs.items():
        if data.dtype != np.float32: data = data.astype(np.float32)
        np.save(os.path.join(output_dir, f"{name}.npy"), data)
        manifest["tensors"].append({"name": name, "file": f"{name}.npy", "dtype": "float32", "shape": list(data.shape)})
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

def infer_ort(onnx_path, input_tensors, batch_size, cache):
    if onnx_path in cache:
        session = cache[onnx_path]
    else:
        providers = [("CUDAExecutionProvider", {"device_id": "0"}), ("CPUExecutionProvider", {})]
        opts = ort.SessionOptions(); opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        session = ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)
        cache[onnx_path] = session
    input_names = [m.name for m in session.get_inputs()]
    output_names = [m.name for m in session.get_outputs()]
    onnx_to_np = {}
    for onnx_name in input_names:
        for k, v in input_tensors.items():
            if k.lower() == onnx_name.lower(): onnx_to_np[onnx_name] = v; break
    N = list(onnx_to_np.values())[0].shape[0]; B = min(batch_size, N)
    all_out = {n: [] for n in output_names}
    for s in range(0, N, B):
        e = min(s+B, N); feed = {n: onnx_to_np[n][s:e] for n in onnx_to_np}
        results = session.run(output_names, feed)
        for n, arr in zip(output_names, results): all_out[n].append(arr)
    return {n: np.concatenate(chunks, axis=0) for n, chunks in all_out.items()}

# ═══════════════════════════════════════════════════════════════
# BigFormer mmap preload (one-time, ~12s)
# ═══════════════════════════════════════════════════════════════
BIGFORMER_PATH = "/workspace/C3/testcases/models/bigformer_v1.onnx"
BF_LOADED = False
BF_BLOCKS = None
BF_TOK, BF_POS, BF_HEAD_W, BF_HEAD_B, BF_LNF_W, BF_LNF_B = None, None, None, None, None, None
BF_DEVICE = torch.device("cuda")

def init_bigformer():
    global BF_LOADED, BF_BLOCKS, BF_TOK, BF_POS, BF_HEAD_W, BF_HEAD_B, BF_LNF_W, BF_LNF_B
    if BF_LOADED: return
    print("[worker] Preloading BigFormer (mmap→FP16→GPU)...", file=sys.stderr)
    
    model = onnx.load(BIGFORMER_PATH, load_external_data=False)
    identity_map = {n.output[0]: n.input[0] for n in model.graph.node if n.op_type == "Identity"}
    def resolve(name):
        visited = set()
        while name in identity_map and name not in visited:
            visited.add(name); name = identity_map[name]
        return name
    
    data_path = BIGFORMER_PATH + ".data"
    total_bytes = os.path.getsize(data_path)
    data_mmap = np.memmap(data_path, dtype=np.float32, mode='r', shape=(total_bytes//4,))
    
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
    
    # Load all weights to GPU as FP16
    BF_BLOCKS = []
    for bid in range(24):
        bw = block_weights[bid]
        def g(n):
            name = rp(f"blocks.{bid}.{n}")
            t = load_fp16(name) if name else None
            return t.to(BF_DEVICE) if t is not None else None
        BF_BLOCKS.append({
            "qkv": load_fp16(bw["qkv"]).to(BF_DEVICE), "proj": load_fp16(bw["proj"]).to(BF_DEVICE),
            "ff1": load_fp16(bw["ff1"]).to(BF_DEVICE), "ff2": load_fp16(bw["ff2"]).to(BF_DEVICE),
            "qkv_b": g("qkv.bias"), "proj_b": g("proj.bias"), "ff1_b": g("ff1.bias"), "ff2_b": g("ff2.bias"),
            "ln1_w": g("ln1.weight"), "ln1_b": g("ln1.bias"), "ln2_w": g("ln2.weight"), "ln2_b": g("ln2.bias"),
        })
    
    BF_TOK = load_fp16(rp("tok_emb.weight")).float().to(BF_DEVICE)
    BF_POS = load_fp16(rp("pos_emb")).float().to(BF_DEVICE)
    BF_HEAD_W = load_fp16(block_weights["head"]).to(BF_DEVICE) if "head" in block_weights else None
    BF_HEAD_B = load_fp16(rp("head.bias")).float().to(BF_DEVICE) if rp("head.bias") else None
    BF_LNF_W = load_fp16(rp("ln_f.weight")).float().to(BF_DEVICE) if rp("ln_f.weight") else None
    BF_LNF_B = load_fp16(rp("ln_f.bias")).float().to(BF_DEVICE) if rp("ln_f.bias") else None
    
    BF_LOADED = True
    print(f"[worker] BigFormer ready: {torch.cuda.memory_allocated()/1e9:.1f}GB GPU", file=sys.stderr)

def infer_bigformer(input_tensors, batch_size):
    init_bigformer()
    input_ids = torch.from_numpy(input_tensors["input_ids"]).long()
    N, D = input_ids.shape[0], BF_TOK.shape[1]; B = min(batch_size, N)
    all_logits = []
    for s in range(0, N, B):
        e = min(s+B, N); batch = input_ids[s:e].to(BF_DEVICE); BS, SS = batch.shape
        x = BF_TOK[batch] + BF_POS[:, :SS, :]
        for w in BF_BLOCKS:
            residual = x.float()
            if w["ln1_w"] is not None:
                x = F.layer_norm(x.float(), [D], weight=w["ln1_w"].float(), bias=w["ln1_b"].float() if w["ln1_b"] is not None else None, eps=1e-5)
            qkv = x @ w["qkv"].float()
            if w["qkv_b"] is not None: qkv = qkv + w["qkv_b"].float()
            q,k,v = qkv.chunk(3,dim=-1); q=q.view(BS,SS,32,128).permute(0,2,1,3); k=k.view(BS,SS,32,128).permute(0,2,1,3); v=v.view(BS,SS,32,128).permute(0,2,1,3)
            attn_out = (F.softmax((q@k.transpose(-2,-1))*(128**-0.5),dim=-1)@v).permute(0,2,1,3).reshape(BS,SS,4096)
            attn_out = attn_out @ w["proj"].float()
            if w["proj_b"] is not None: attn_out = attn_out + w["proj_b"].float()
            x = residual + attn_out
            residual = x.float()
            if w["ln2_w"] is not None:
                x = F.layer_norm(x.float(),[D],weight=w["ln2_w"].float(),bias=w["ln2_b"].float() if w["ln2_b"] is not None else None,eps=1e-5)
            torch.backends.cuda.matmul.allow_tf32 = True
            x = x @ w["ff1"].float(); 
            if w["ff1_b"] is not None: x = x + w["ff1_b"].float()
            x = F.gelu(x); x = x @ w["ff2"].float()
            if w["ff2_b"] is not None: x = x + w["ff2_b"].float()
            torch.backends.cuda.matmul.allow_tf32 = False
            x = residual + x
        if BF_LNF_W is not None: x = F.layer_norm(x.float(),[D],weight=BF_LNF_W,bias=BF_LNF_B,eps=1e-5)
        if BF_HEAD_W is not None: x = x.float() @ BF_HEAD_W.float() + (BF_HEAD_B if BF_HEAD_B is not None else 0)
        all_logits.append(x.cpu().numpy())
    return {"logits": np.concatenate(all_logits,axis=0).astype(np.float32)}

# ═══════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    try: task = json.loads(line)
    except: continue
    
    if task.get("cmd") == "exit":
        print("[worker] Exit", file=sys.stderr)
        break
    
    onnx_path = task["onnx"]; input_dir = task["input"]; output_dir = task["output"]
    batch_size = task.get("batch_size", 256)
    
    try:
        t0 = time.perf_counter()
        input_tensors = load_inputs(input_dir)
        N = list(input_tensors.values())[0].shape[0]
        
        # Route: BigFormer → GPU mmap, others → ORT
        is_bigformer = "bigformer" in onnx_path.lower() or os.path.getsize(onnx_path + ".data" if os.path.exists(onnx_path + ".data") else onnx_path) > 10*1024**3
        
        if is_bigformer:
            print(f"[worker] BigFormer GPU", file=sys.stderr)
            outputs = infer_bigformer(input_tensors, batch_size)
        else:
            outputs = infer_ort(onnx_path, input_tensors, batch_size, session_cache)
        
        write_outputs(output_dir, outputs)
        dt = time.perf_counter() - t0
        print(json.dumps({"status": "ok", "samples": N}), flush=True)
        print(f"[worker] Done: {N} samples in {dt:.1f}s", file=sys.stderr)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"status": "error", "error": str(e)}), flush=True)

sys.exit(0)
