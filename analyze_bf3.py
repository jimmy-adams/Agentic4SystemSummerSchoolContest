"""Quick fix: map ALL 6 MatMuls per block correctly."""
import onnx
from onnx.numpy_helper import to_array

m = onnx.load('/workspace/C3/testcases/models/bigformer_v1.onnx')

# List all MatMul nodes per block with their weight initializers
init_names = {i.name for i in m.graph.initializer}

block_mm = {}
for node in m.graph.node:
    if node.op_type != 'MatMul':
        continue
    parts = node.name.split('/')
    if len(parts) < 3 or not parts[1].startswith('blocks.'):
        continue
    bid = int(parts[1].split('.')[1])
    sub = parts[2] if len(parts) > 2 else 'unknown'
    
    if bid not in block_mm:
        block_mm[bid] = []
    
    weight_inps = [inp for inp in node.input if inp in init_names]
    block_mm[bid].append((node.name, sub, weight_inps))

print("Block MatMul structure:")
for bid in sorted(block_mm.keys())[:2]:
    print(f"\nBlock {bid}:")
    for name, sub, weights in block_mm[bid]:
        w_info = []
        for w in weights:
            init = [i for i in m.graph.initializer if i.name == w][0]
            arr = to_array(init)
            w_info.append(f"{w}: {list(arr.shape)} {arr.nbytes/1e6:.1f}MB")
        print(f"  {name} ({sub}): {', '.join(w_info)}")

# Also show the head
for node in m.graph.node:
    if node.op_type == 'MatMul' and 'head' in node.name:
        weight_inps = [inp for inp in node.input if inp in init_names]
        print(f"\nHead: {node.name}")
        for w in weight_inps:
            init = [i for i in m.graph.initializer if i.name == w][0]
            arr = to_array(init)
            print(f"  {w}: {list(arr.shape)} {arr.nbytes/1e6:.1f}MB")
