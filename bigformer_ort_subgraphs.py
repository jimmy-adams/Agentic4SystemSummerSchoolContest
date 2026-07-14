"""BigFormer GPU: split into per-block ONNX subgraphs, run with ORT GPU.

Strategy: extract each transformer block as an independent ONNX model.
Each block has ~804MB weights → fits in 16GB GPU.
ORT handles all ops correctly (multi-head attention, Identity aliasing, etc).
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

print("=== Phase 1: Parse ONNX graph ===", file=sys.stderr)
t0 = time.perf_counter()
model = onnx.load(ONNX_PATH)

# Find block boundaries: each block starts at a LayerNormalization
# and ends at the next LayerNormalization (or before next block)
# Pattern per block: LayerNorm → attention ops → LayerNorm → FFN ops

# First, find all LN node indices
ln_indices = []
for i, node in enumerate(model.graph.node):
    if node.op_type == 'LayerNormalization':
        ln_indices.append(i)

print(f"  Total LayerNorm nodes: {len(ln_indices)}", file=sys.stderr)

# The first 2 LNs are block 0 (ln1 + ln2), next 2 are block 1, etc.
# 48 LNs for 24 blocks, plus 1 final LN = 49 total
# Extract blocks: block N = nodes from LN[2N] to before LN[2N+2]

# But the graph has shared initializers accessed via Identity nodes.
# For each subgraph, we need to:
# 1. Identify which initializers are used (directly or via Identity)
# 2. Include them in the subgraph

# Build Identity resolution map
identity_map = {}
for node in model.graph.node:
    if node.op_type == 'Identity':
        identity_map[node.output[0]] = node.input[0]

def resolve_name(name):
    visited = set()
    while name in identity_map and name not in visited:
        visited.add(name)
        name = identity_map[name]
    return name

# Build initializer set
init_set = {i.name for i in model.graph.initializer}
init_data_map = {i.name: i for i in model.graph.initializer}

def get_needed_initializers(node_list):
    """Collect all initializer names needed by a set of nodes."""
    needed = set()
    for node in node_list:
        for inp in node.input:
            resolved = resolve_name(inp)
            if resolved in init_set:
                needed.add(resolved)
    return needed

# ── Extract blocks ──
# Each block: between LN[bid*2] and LN[bid*2+2] (exclusive end)
# Input: activation from previous block
# Output: activation after block

num_blocks = 24  # blocks.0 to blocks.23
blocks = []  # [(start_node_idx, end_node_idx, input_name, output_name)]

for bid in range(num_blocks):
    start_ln = ln_indices[bid * 2]   # first LN of this block (ln1)
    # End: either the first LN of next block, or the final LN
    if bid < num_blocks - 1:
        end_idx = ln_indices[(bid + 1) * 2]  # first LN of next block
    else:
        end_idx = ln_indices[-1]  # final LN (ln_f)
    
    # Input: the residual tensor entering this block's first LN
    # Output: the output of the last Add (residual) before next block's LN
    start_node = model.graph.node[start_ln]
    end_node = model.graph.node[end_idx - 1]
    
    block_input = start_node.input[0]
    block_output = end_node.output[0]
    
    blocks.append((start_ln, end_idx, block_input, block_output))

print(f"  Blocks extracted: {len(blocks)}", file=sys.stderr)

# ── Pre-load all initializer DATA to CPU ──
print("\n=== Phase 2: Load weights to CPU ===", file=sys.stderr)
weight_data = {}
total_w = 0
for init in model.graph.initializer:
    arr = to_array(init)
    weight_data[init.name] = arr
    total_w += arr.nbytes
print(f"  {total_w/1e9:.1f} GB on CPU", file=sys.stderr)

# ── Input loading ──
print("\n=== Phase 3: Load input ===", file=sys.stderr)
with open(os.path.join(INPUT_DIR, "manifest.json")) as f:
    manifest = json.load(f)
input_data = np.load(os.path.join(INPUT_DIR, manifest["tensors"][0]["file"]))
input_ids = input_data.astype(np.int64)
N = input_ids.shape[0]

# ── Initial embeddings (pre-block) ──
# Nodes before first LN: Gather(tok_emb) → Add(pos_emb)
# Let's do this manually since it's simple

tok_emb_w = weight_data["tok_emb.weight"]
pos_emb_w = weight_data["pos_emb"]

# ── Extract final head (after last block) ──
# Nodes after last LN: MatMul(head) → Add(bias)
head_nodes = []
final_ln_idx = ln_indices[-1]
for i in range(final_ln_idx, len(model.graph.node)):
    node = model.graph.node[i]
    if node.op_type in ('MatMul', 'Add') and 'head' in node.name.lower():
        head_nodes.append(node)

print(f"  Head nodes: {len(head_nodes)}", file=sys.stderr)
head_w = weight_data.get("onnx::MatMul_1477")
head_b = weight_data.get("head.bias")

# ── Phase 4: Build ORT sub-sessions ──
print("\n=== Phase 4: Build ORT GPU sub-sessions ===", file=sys.stderr)

ort_sessions = []
providers = [("CUDAExecutionProvider", {"device_id": "0"}), ("CPUExecutionProvider", {})]
sess_opt = ort.SessionOptions()
sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

for bid, (start_idx, end_idx, inp_name, out_name) in enumerate(blocks):
    sub_nodes = [model.graph.node[i] for i in range(start_idx, end_idx)]
    
    # Collect needed initializers
    needed_inits = get_needed_initializers(sub_nodes)
    
    # Build subgraph
    sub_inits = []
    for name in needed_inits:
        if name in weight_data:
            tensor_proto = numpy_helper.from_array(weight_data[name], name=name)
            sub_inits.append(tensor_proto)
    
    # Input: the activation tensor entering this block
    # We need to find its shape from context
    # For first block: [B, S, 4096] (after embeddings)
    # For subsequent blocks: same shape
    
    # Create input value info
    sub_input = helper.make_tensor_value_info(inp_name, onnx.TensorProto.FLOAT, [BATCH_SIZE, 32, 4096])
    sub_output = helper.make_tensor_value_info(out_name, onnx.TensorProto.FLOAT, [BATCH_SIZE, 32, 4096])
    
    sub_graph = helper.make_graph(sub_nodes, f"block_{bid}", [sub_input], [sub_output], sub_inits)
    sub_model = helper.make_model(sub_graph, producer_name="bf_subgraph")
    # Override opset to 26 (ORT doesn't fully support opset 27)
    sub_model.opset_import[0].version = 26
    
    # Serialize to bytes (avoid file I/O)
    serialized = sub_model.SerializeToString()
    sess = ort.InferenceSession(serialized, sess_options=sess_opt, providers=providers)
    ort_sessions.append((sess, inp_name, out_name))
    
    if bid < 3:
        print(f"  Block {bid}: {len(sub_nodes)} nodes, {len(needed_inits)} init, "
              f"in={inp_name} out={out_name}", file=sys.stderr)

# ── Also build embedding + pre-LN subgraph ──
# Nodes: Gather(tok_emb) → Add(pos_emb) → first LN
pre_nodes = []
pre_end = ln_indices[0] + 1  # include first LN
for i in range(0, pre_end):
    pre_nodes.append(model.graph.node[i])

pre_needed = get_needed_initializers(pre_nodes)
pre_inits = [numpy_helper.from_array(weight_data[n], name=n) for n in pre_needed if n in weight_data]

pre_input = helper.make_tensor_value_info("input_ids", onnx.TensorProto.INT64, [BATCH_SIZE, 32])
# First LN output is the input to first attention
first_ln_out = model.graph.node[ln_indices[0]].output[0]
pre_output = helper.make_tensor_value_info(first_ln_out, onnx.TensorProto.FLOAT, [BATCH_SIZE, 32, 4096])

pre_graph = helper.make_graph(pre_nodes, "pre_block", [pre_input], [pre_output], pre_inits)
pre_model = helper.make_model(pre_graph, producer_name="bf_pre")
pre_model.opset_import[0].version = 26
pre_sess = ort.InferenceSession(pre_model.SerializeToString(), sess_options=sess_opt, providers=providers)
print(f"\n  Pre-block: {len(pre_nodes)} nodes, {len(pre_needed)} init", file=sys.stderr)

# ── Post-block (head) subgraph ──
post_start = final_ln_idx
post_nodes = [model.graph.node[i] for i in range(post_start, len(model.graph.node))]
post_needed = get_needed_initializers(post_nodes)
post_inits = [numpy_helper.from_array(weight_data[n], name=n) for n in post_needed if n in weight_data]

post_input_name = model.graph.node[post_start].input[0]
post_input = helper.make_tensor_value_info(post_input_name, onnx.TensorProto.FLOAT, [BATCH_SIZE, 32, 4096])
last_out = model.graph.node[-1].output[0] if model.graph.node[-1].output else model.graph.node[-2].output[0]
post_output = helper.make_tensor_value_info(last_out, onnx.TensorProto.FLOAT, [BATCH_SIZE, 32, 14])

post_graph = helper.make_graph(post_nodes, "post_block", [post_input], [post_output], post_inits)
post_model = helper.make_model(post_graph, producer_name="bf_post")
post_model.opset_import[0].version = 26
post_sess = ort.InferenceSession(post_model.SerializeToString(), sess_options=sess_opt, providers=providers)
print(f"  Post-block: {len(post_nodes)} nodes, {len(post_needed)} init", file=sys.stderr)

# ── Phase 5: Run inference ──
print("\n=== Phase 5: Inference ===", file=sys.stderr)

all_logits = []

for start in range(0, N, BATCH_SIZE):
    end = min(start + BATCH_SIZE, N)
    batch = input_ids[start:end]
    B = batch.shape[0]
    
    # Pre-block: token embedding + positional encoding + first LN
    pre_out = pre_sess.run(None, {"input_ids": batch})
    activation = pre_out[0]  # name = first LN output
    
    # Per-block execution
    for bid, (sess, inp_name, out_name) in enumerate(ort_sessions):
        feed = {inp_name: activation}
        out = sess.run(None, feed)
        activation = out[0]
    
    # Post-block: final LN + head
    post_out = post_sess.run(None, {post_input_name: activation})
    logits_batch = post_out[0]
    
    all_logits.append(logits_batch)
    
    if start == 0:
        dt = time.perf_counter() - t0
        print(f"  First batch: {dt:.1f}s shape={logits_batch.shape}", file=sys.stderr)

# ── Results ──
logits = np.concatenate(all_logits, axis=0).astype(np.float32)
dt = time.perf_counter() - t0
golden = np.load(os.path.join(GOLDEN_DIR, "logits.npy"))
ok = np.allclose(logits, golden, rtol=1e-3, atol=1e-3)
md = np.max(np.abs(logits - golden))
print(f"\nTime: {dt:.1f}s | Precision: {'PASS' if ok else 'FAIL'} | MAX_DIFF: {md:.2e}", file=sys.stderr)
