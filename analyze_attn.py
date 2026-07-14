"""Analyze multi-head attention structure in BigFormer."""
import onnx

m = onnx.load('/workspace/C3/testcases/models/bigformer_v1.onnx')

# Find Reshape/Transpose/Constant nodes around first attention block
# Block 0 attention: nodes between LN[0] and LN[1]
for i, node in enumerate(m.graph.node):
    if node.name == '/blocks.0/ln1/LayerNormalization':
        ln1_idx = i
    if node.name == '/blocks.0/ln2/LayerNormalization':
        ln2_idx = i

print("=== Block 0 attention (ln1 → ln2) ===")
for i in range(ln1_idx, ln2_idx + 1):
    node = m.graph.node[i]
    if node.op_type in ('Reshape', 'Transpose', 'Split', 'Constant', 'MatMul', 'Softmax'):
        attrs = {}
        for a in node.attribute:
            if a.name == 'value':
                from onnx.numpy_helper import to_array
                try:
                    attrs['value'] = list(to_array(a.t).flatten())
                except:
                    attrs['value'] = '?'
            elif a.type == 7:  # INTS
                attrs[a.name] = list(a.ints)
            elif a.type == 2:  # INT
                attrs[a.name] = a.i
        print(f"  {node.name}: {node.op_type} attrs={attrs}")
        print(f"    in={node.input[:2]}  out={node.output[:2]}")
