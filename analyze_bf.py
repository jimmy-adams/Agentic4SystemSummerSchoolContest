import onnx
from onnx.numpy_helper import to_array
from collections import defaultdict

m = onnx.load('/workspace/C3/testcases/models/bigformer_v1.onnx')

matmul_nodes = [n for n in m.graph.node if n.op_type == 'MatMul']
ln_nodes = [n for n in m.graph.node if n.op_type == 'LayerNormalization']
softmax_nodes = [n for n in m.graph.node if n.op_type == 'Softmax']

print(f'Nodes: MatMul={len(matmul_nodes)}  LayerNorm={len(ln_nodes)}  Softmax={len(softmax_nodes)}')
print(f'Total nodes: {len(m.graph.node)}')

total = 0
weight_sizes = []
for init in m.graph.initializer:
    arr = to_array(init)
    size = arr.nbytes
    total += size
    weight_sizes.append((init.name, size))

weight_sizes.sort(key=lambda x: -x[1])
print(f'\nTotal weights: {total/1e9:.2f} GB ({len(weight_sizes)} initializers)')

# Group by name prefix to find layers
layer_weights = defaultdict(int)
for name, size in weight_sizes:
    parts = name.split('/')
    if len(parts) >= 3:
        layer_key = '/'.join(parts[:3])
    else:
        layer_key = name
    layer_weights[layer_key] += size

print('\nWeight by component (MB):')
for key, size in sorted(layer_weights.items(), key=lambda x: -x[1]):
    print(f'  {key}: {size/1e6:.1f} MB')

# Check per-layer size
if layer_weights:
    per_layer = list(layer_weights.values())
    print(f'\nMax layer: {max(per_layer)/1e9:.2f} GB')
    print(f'Min layer: {min(per_layer)/1e6:.1f} MB')
    print(f'Num components: {len(per_layer)}')

# Check node input/output patterns for layer boundaries
print('\n--- First 10 nodes ---')
for i, node in enumerate(m.graph.node[:10]):
    print(f'  {i}: {node.op_type}  in={node.input[:2]}...  out={node.output[:1]}')
