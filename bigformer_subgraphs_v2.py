"""BigFormer ORT GPU via subgraph splitting.

Splits the 1037-node ONNX graph into 24 per-block subgraphs (~804MB each),
plus pre-block (embeddings) and post-block (head). Each subgraph runs on ORT GPU.
"""
import onnx
from onnx import helper, numpy_helper
from onnx.numpy_helper import to_array
import numpy as np
import onnxruntime as ort
import time, os, sys, json, gc
from collections import defaultdict

ONNX_PATH = "/workspace/C3/testcases/models/bigformer_v1.onnx"
INPUT_DIR = "/workspace/C3/testcases/testdata/c35/bigformer_v1/input"
GOLDEN_DIR = "/workspace/C3/testcases/testdata/c35/bigformer_v1/golden"
BATCH_SIZE = 16

# ═══════════════════════════════════════════════════════════════════════
# 1. Parse & build Identity resolution
# ═══════════════════════════════════════════════════════════════════════
print("=== Parse ONNX ===", file=sys.stderr)
t0 = time.perf_counter()
model = onnx.load(ONNX_PATH)

identity_map = {}
for node in model.graph.node:
    if node.op_type == "Identity":
        identity_map[node.output[0]] = node.input[0]

def resolve(name):
    visited = set()
    while name in identity_map and name not in visited:
        visited.add(name)
        name = identity_map[name]
    return name

init_set = {i.name for i in model.graph.initializer}

# ═══════════════════════════════════════════════════════════════════════
# 2. Load weights
# ═══════════════════════════════════════════════════════════════════════
print("=== Load weights ===", file=sys.stderr)
weight_data = {}
for init in model.graph.initializer:
    weight_data[init.name] = to_array(init)
print(f"  {sum(a.nbytes for a in weight_data.values())/1e9:.1f} GB", file=sys.stderr)

# ═══════════════════════════════════════════════════════════════════════
# 3. Helper: create resolved subgraph
# ═══════════════════════════════════════════════════════════════════════
providers = [("CUDAExecutionProvider", {"device_id": "0"}), ("CPUExecutionProvider", {})]
sess_opt = ort.SessionOptions()
sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

def make_subgraph_session(node_list, input_name, input_shape, output_name, output_shape, tag):
    """Create ORT session for a node subgraph with Identity resolution."""
    # Resolve Identity aliases in node inputs
    resolved_nodes = []
    needed_inits = set()
    
    for node in node_list:
        new_inputs = []
        for inp in node.input:
            r = resolve(inp)
            if r in init_set:
                new_inputs.append(r)
                needed_inits.add(r)
            else:
                new_inputs.append(inp)
        
        attrs = {}
        for a in node.attribute:
            attrs[a.name] = helper.get_attribute_value(a)
        
        resolved_nodes.append(helper.make_node(
            node.op_type, new_inputs, list(node.output),
            name=node.name, **attrs
        ))
    
    sub_inits = [numpy_helper.from_array(weight_data[n], name=n) for n in needed_inits]
    sub_input = helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, input_shape)
    sub_output = helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, output_shape)
    sub_graph = helper.make_graph(resolved_nodes, tag, [sub_input], [sub_output], sub_inits)
    sub_model = helper.make_model(sub_graph, producer_name="bf")
    sub_model.opset_import[0].version = 26  # override opset 27
    
    serialized = sub_model.SerializeToString()
    sess = ort.InferenceSession(serialized, sess_options=sess_opt, providers=providers)
    print(f"  [{tag}] {len(resolved_nodes)} nodes, {len(needed_inits)} weights", file=sys.stderr)
    return sess, input_name, output_name

# ═══════════════════════════════════════════════════════════════════════
# 4. Split graph into blocks
# ═══════════════════════════════════════════════════════════════════════
print("=== Build subgraphs ===", file=sys.stderr)

# Find LN node indices
ln_indices = [i for i, n in enumerate(model.graph.node) if n.op_type == 'LayerNormalization']
print(f"  {len(ln_indices)} LayerNorms", file=sys.stderr)

# Pre-block: start → first LN (inclusive)
pre_end = ln_indices[0] + 1  # include first LN
pre_nodes = [model.graph.node[i] for i in range(pre_end)]

# Need special handling: first input is input_ids (int64), not float
pre_input = helper.make_tensor_value_info("input_ids", onnx.TensorProto.INT64, [BATCH_SIZE, 32])
first_ln_out = model.graph.node[ln_indices[0]].output[0]
pre_output = helper.make_tensor_value_info(first_ln_out, onnx.TensorProto.FLOAT, [BATCH_SIZE, 32, 4096])

# Resolve pre-block
pre_resolved = []
pre_needed = set()
for node in pre_nodes:
    new_inputs = []
    for inp in node.input:
        r = resolve(inp)
        if r in init_set:
            new_inputs.append(r)
            pre_needed.add(r)
        else:
            new_inputs.append(inp)
    attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    pre_resolved.append(helper.make_node(node.op_type, new_inputs, list(node.output), name=node.name, **attrs))

pre_inits = [numpy_helper.from_array(weight_data[n], name=n) for n in pre_needed]
pre_graph = helper.make_graph(pre_resolved, "pre", [pre_input], [pre_output], pre_inits)
pre_model = helper.make_model(pre_graph, producer_name="bf")
pre_model.opset_import[0].version = 26
pre_sess = ort.InferenceSession(pre_model.SerializeToString(), sess_options=sess_opt, providers=providers)
print(f"  [pre] {len(pre_resolved)} nodes, {len(pre_needed)} weights", file=sys.stderr)

# Post-block: final LN + head (nodes from last LN to end of graph)
post_start = ln_indices[-1]  # final LN
post_nodes = [model.graph.node[i] for i in range(post_start, len(model.graph.node))]
post_inp_name = post_nodes[0].input[0]  # input to final LN

# Find final output name
post_out_name = None
for node in reversed(post_nodes):
    if node.output:
        post_out_name = node.output[0]
        break

# Determine output shape from head weight
head_w = weight_data.get("onnx::MatMul_1477")
vocab_size = head_w.shape[1] if head_w is not None else 4096

post_sess, _, _ = make_subgraph_session(post_nodes, post_inp_name, [BATCH_SIZE, 32, 4096],
                                         post_out_name, [BATCH_SIZE, 32, vocab_size], "head")
print(f"  output: {post_out_name} shape [B,32,{vocab_size}]", file=sys.stderr)

# ═══════════════════════════════════════════════════════════════════════
# 5. Inference (streaming: create/destroy ORT sessions per block)
# ═══════════════════════════════════════════════════════════════════════
print("=== Inference ===", file=sys.stderr)

with open(os.path.join(INPUT_DIR, "manifest.json")) as f:
    manifest = json.load(f)
input_data = np.load(os.path.join(INPUT_DIR, manifest["tensors"][0]["file"]))
input_ids = input_data.astype(np.int64)
N = input_ids.shape[0]

all_logits = []

for start in range(0, N, BATCH_SIZE):
    end = min(start + BATCH_SIZE, N)
    batch = input_ids[start:end]
    
    # Pre-block
    out = pre_sess.run(None, {"input_ids": batch})
    activation = out[0]
    
    # Per-block: create session, run, destroy
    for bid in range(24):
        start_ln = ln_indices[bid * 2]
        if bid < 24 - 1:
            end_idx = ln_indices[(bid + 1) * 2]
        else:
            end_idx = ln_indices[-1]
        
        sub_nodes = [model.graph.node[i] for i in range(start_ln, end_idx)]
        inp_name = sub_nodes[0].input[0]
        out_name = sub_nodes[-1].output[0]
        
        sess, sn_in, sn_out = make_subgraph_session(
            sub_nodes, inp_name, list(activation.shape),
            out_name, list(activation.shape), f"b{bid}")
        
        out = sess.run(None, {sn_in: activation})
        activation = out[0]
        
        # Free session immediately
        del sess
        gc.collect()
    
    # Post-block (head)
    out = post_sess.run(None, {post_inp_name: activation})
    logits_batch = out[0]
    
    all_logits.append(logits_batch)
    
    if start == 0:
        dt = time.perf_counter() - t0
        print(f"  First batch: {dt:.1f}s shape={logits_batch.shape}", file=sys.stderr)

# ═══════════════════════════════════════════════════════════════════════
# 6. Results
# ═══════════════════════════════════════════════════════════════════════
logits = np.concatenate(all_logits, axis=0).astype(np.float32)
if logits.ndim == 4:
    logits = logits.reshape(-1, logits.shape[-2], logits.shape[-1])
dt = time.perf_counter() - t0
golden = np.load(os.path.join(GOLDEN_DIR, "logits.npy"))
ok = np.allclose(logits, golden, rtol=1e-3, atol=1e-3)
md = np.max(np.abs(logits - golden))
print(f"\nTime: {dt:.1f}s | Precision: {'PASS' if ok else 'FAIL'} | MAX_DIFF: {md:.2e}", file=sys.stderr)
