"""C3.5 Worker v3 — ORT for normal models, teammate's BigFormer for large models."""
import json, os, sys, time, traceback, numpy as np, onnx, onnxruntime as ort

BIGFORMER_PATH = "/workspace/C3/testcases/models/bigformer_v1.onnx"

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

# Teammate's BigFormer executor (lazy init)
BF_EXECUTOR = None

def init_bigformer():
    global BF_EXECUTOR
    if BF_EXECUTOR is None:
        from bigformer_streaming import BigFormerStreamingExecutor
        print("[worker] Preloading BigFormer (teammate executor)...", file=sys.stderr)
        BF_EXECUTOR = BigFormerStreamingExecutor(BIGFORMER_PATH)
        print("[worker] BigFormer ready", file=sys.stderr)
    return BF_EXECUTOR

def infer_bigformer(input_tensors, batch_size):
    executor = init_bigformer()
    return executor.run(input_tensors, batch_size)

# Main loop
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
        
        is_bf = "bigformer" in onnx_path.lower()
        
        if is_bf:
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
