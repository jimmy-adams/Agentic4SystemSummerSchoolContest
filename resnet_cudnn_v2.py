"""ResNet via PyTorch cuDNN — proper memory management."""
import onnx, numpy as np, torch, torch.nn.functional as F, time, json, os, gc

ONNX_PATH = "/workspace/C3/testcases/models/resnet_v1.onnx"
INPUT_DIR = "/workspace/C3/testcases/testdata/c35/resnet_v1/input"
GOLDEN_DIR = "/workspace/C3/testcases/testdata/c35/resnet_v1/golden"
device = torch.device("cuda")

print("=== Build ResNet from ONNX ===")
t0 = time.perf_counter()

model = onnx.load(ONNX_PATH)

# Load all initializers to GPU
weights = {}
for init in model.graph.initializer:
    w = onnx.numpy_helper.to_array(init)
    weights[init.name] = torch.from_numpy(w.copy()).float().to(device)

# Pre-parse node attributes (strides, pads, etc.)  
node_attrs = {}
for node in model.graph.node:
    attrs = {}
    for a in node.attribute:
        if a.name == "strides": attrs["stride"] = tuple(a.ints)
        elif a.name == "pads": attrs["pads"] = list(a.ints)
        elif a.name == "group": attrs["group"] = a.i
        elif a.name == "dilations": attrs["dilation"] = tuple(a.ints)
    node_attrs[node.name] = attrs

# Build consumer map: how many times each tensor is consumed
consumer_count = {}
for node in model.graph.node:
    for inp in node.input:
        consumer_count[inp] = consumer_count.get(inp, 0) + 1

input_name = model.graph.input[0].name
output_name = model.graph.output[0].name

print(f"  Ready: {time.perf_counter()-t0:.1f}s")

# Load input
with open(os.path.join(INPUT_DIR, "manifest.json")) as f: m = json.load(f)
data = torch.from_numpy(np.load(os.path.join(INPUT_DIR, m["tensors"][0]["file"]))).float()
N = data.shape[0]
BS = 256

def run_inference(data, batch_size):
    """Run ResNet inference, pad last batch to avoid cuDNN algorithm change."""
    all_out = []
    for st in range(0, data.shape[0], batch_size):
        end = min(st+batch_size, data.shape[0])
        batch = data[st:end]
        actual = batch.shape[0]
        
        # Pad to full batch_size if needed
        if actual < batch_size:
            pad = torch.zeros(batch_size - actual, *batch.shape[1:], dtype=batch.dtype)
            batch = torch.cat([batch, pad], dim=0)
        
        batch = batch.to(device)
        reg = {input_name: batch}
        remaining = dict(consumer_count)
        
        for node in model.graph.node:
            op = node.op_type
            inp = node.input
            out = node.output
            
            if op == "Conv":
                x = reg[inp[0]]; w = weights[inp[1]]
                b = weights.get(inp[2]) if len(inp) > 2 else None
                attrs = node_attrs.get(node.name, {})
                stride = attrs.get("stride", (1, 1))
                pads = attrs.get("pads", [0,0,0,0])
                p = (pads[0], pads[2]) if len(pads) >= 4 else (pads[0], pads[0]) if len(pads) >= 2 else 0
                groups = attrs.get("group", 1)
                result = F.conv2d(x, w, b, stride=stride, padding=p, groups=groups)
                
            elif op == "Relu":
                result = F.relu(reg[inp[0]])
                
            elif op == "Add":
                result = reg[inp[0]] + reg[inp[1]]
                
            elif op == "GlobalAveragePool":
                result = reg[inp[0]].mean(dim=[2, 3])
                
            elif op == "Flatten":
                result = reg[inp[0]].reshape(reg[inp[0]].shape[0], -1)
                
            elif op == "Gemm":
                x = reg[inp[0]]; w = weights[inp[1]]
                b = weights.get(inp[2]) if len(inp) > 2 else None
                result = F.linear(x, w, b)
            else:
                continue
            
            reg[out[0]] = result
            
            for i_name in inp:
                if i_name in remaining and i_name in reg:
                    remaining[i_name] -= 1
                    if remaining[i_name] <= 0 and i_name != input_name:
                        del reg[i_name]
        
        result = reg[output_name][:actual]  # trim padding
        all_out.append(result.cpu())
        del reg; gc.collect()
    
    return torch.cat(all_out, dim=0)

# Warmup
_ = run_inference(data[:256], 256)
torch.cuda.synchronize()

# Time
t0 = time.perf_counter()
with torch.no_grad():
    out = run_inference(data, BS)
torch.cuda.synchronize()
dt = time.perf_counter() - t0

gold = np.load(os.path.join(GOLDEN_DIR, "logits.npy"))
out_np = out.numpy()
ok = np.allclose(out_np, gold, rtol=1e-3, atol=1e-3)
print(f"\nTime: {dt:.1f}s  PREC={'PASS' if ok else 'FAIL'}  MAX_DIFF: {np.max(np.abs(out_np-gold)):.2e}")
