"""Analyze BigFormer: map weights to layers via graph topology."""
import onnx
from onnx.numpy_helper import to_array
from collections import defaultdict

m = onnx.load('/workspace/C3/testcases/models/bigformer_v1.onnx')

# Build a map: initializer name -> which MatMul nodes use it
init_to_nodes = defaultdict(list)
node_to_inits = defaultdict(list)

for node in m.graph.node:
    if node.op_type == 'MatMul':
        for inp in node.input:
            if inp in {i.name for i in m.graph.initializer}:
                init_to_nodes[inp].append(node.name)
                node_to_inits[node.name].append(inp)

# Find the ordering: topo sort MatMul nodes
# Build adjacency: if node A's output is used as input by node B
output_to_producer = {}
for node in m.graph.node:
    for out in node.output:
        output_to_producer[out] = node.name

consumers = defaultdict(set)
for node in m.graph.node:
    for inp in node.input:
        if inp in output_to_producer:
            consumers[output_to_producer[inp]].add(node.name)

# Simple approach: group consecutive MatMul nodes
matmul_order = [n.name for n in m.graph.node if n.op_type == 'MatMul']
print(f"Total MatMul nodes: {len(matmul_order)}")
print(f"First 10: {matmul_order[:10]}")
print(f"Last 10: {matmul_order[-10:]}")

# Check pattern: how many MatMuls between LayerNorm nodes?
ln_positions = []
for i, node in enumerate(m.graph.node):
    if node.op_type == 'LayerNormalization':
        ln_positions.append(i)

print(f"\nLayerNorm positions (first 10): {ln_positions[:10]}")
print(f"LayerNorm count: {len(ln_positions)}")

# Count MatMuls between consecutive LayerNorms  
for i in range(min(5, len(ln_positions)-1)):
    start = ln_positions[i]
    end = ln_positions[i+1]
    mm_count = sum(1 for j in range(start, end) if m.graph.node[j].op_type == 'MatMul')
    print(f"  LN {i}→{i+1}: {mm_count} MatMuls")

# Let's find the repeating pattern
print("\n--- Node pattern around first few LNs ---")
for i in range(min(3, len(ln_positions))):
    pos = ln_positions[i]
    print(f"\nAround LN at position {pos}:")
    for j in range(max(0, pos-2), min(len(m.graph.node), pos+10)):
        node = m.graph.node[j]
        init_inputs = [inp for inp in node.input if inp in init_to_nodes]
        print(f"  {'>>>' if j==pos else '   '} {j}: {node.op_type:20s} {node.name[:50]}  weights={init_inputs[:2]}")

# Now map each MatMul weight to its layer based on LN boundaries
print("\n\n=== Layer assignment ===")
layer_id = -1
layer_weights = defaultdict(list)
for i, node in enumerate(m.graph.node):
    if node.op_type == 'LayerNormalization':
        layer_id += 1
    if node.op_type == 'MatMul':
        for inp in node.input:
            if inp in init_to_nodes:
                layer_weights[layer_id].append(inp)

for lid in sorted(layer_weights.keys()):
    weights = layer_weights[lid]
    print(f"Layer {lid}: {len(weights)} weight tensors")
    for w in weights[:3]:
        init = [i for i in m.graph.initializer if i.name == w][0]
        arr = to_array(init)
        print(f"  {w}: shape={list(arr.shape)} size={arr.nbytes/1e6:.1f}MB")
