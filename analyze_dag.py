"""BigFormer DAG analysis: graph structure, critical path, fusion opportunities."""
import onnx, time, sys
from collections import defaultdict

PATH = "/workspace/C3/testcases/models/bigformer_v1.onnx"
print("=== BigFormer DAG Analysis ===\n", file=sys.stderr)

m = onnx.load(PATH, load_external_data=False)

# 1. Node type distribution
op_counts = defaultdict(int)
for node in m.graph.node:
    op_counts[node.op_type] += 1

print("--- Operator Distribution ---", file=sys.stderr)
for op, cnt in sorted(op_counts.items(), key=lambda x: -x[1]):
    bar = "█" * (cnt // 5)
    print(f"  {op:25s} {cnt:4d}  {bar}", file=sys.stderr)

# 2. Graph structure
nodes = list(m.graph.node)
edges = []
for ni, node in enumerate(nodes):
    for inp in node.input:
        # Find producer
        for nj, other in enumerate(nodes):
            if inp in other.output:
                edges.append((nj, ni))
                break

print(f"\n--- Graph Stats ---", file=sys.stderr)
print(f"  Nodes: {len(nodes)}", file=sys.stderr)
print(f"  Edges: {len(edges)}", file=sys.stderr)
print(f"  Inputs: {len(m.graph.input)}", file=sys.stderr)
print(f"  Outputs: {len(m.graph.output)}", file=sys.stderr)

# 3. Layer structure
block_counts = defaultdict(int)
for node in nodes:
    parts = node.name.split("/")
    if len(parts) >= 2 and parts[1].startswith("blocks."):
        bid = parts[1]
        block_counts[bid] += 1

print(f"\n--- Blocks ---", file=sys.stderr)
print(f"  Unique blocks: {len(block_counts)}", file=sys.stderr)
first_block_nodes = block_counts.get("blocks.0", 0)
print(f"  Nodes per block: ~{first_block_nodes} (blocks.0)", file=sys.stderr)

# Show block.0 structure
print(f"\n--- Block 0 Structure ---", file=sys.stderr)
for node in nodes:
    if "/blocks.0/" in node.name:
        print(f"  {node.op_type:25s} {node.name}", file=sys.stderr)

# 4. Critical path length (longest chain)
# Build topological order and longest path
in_degree = {i: 0 for i in range(len(nodes))}
adj = defaultdict(list)
for src, dst in edges:
    adj[src].append(dst)
    in_degree[dst] += 1

# Topo sort
from collections import deque
q = deque([i for i in range(len(nodes)) if in_degree[i] == 0])
topo = []
while q:
    u = q.popleft()
    topo.append(u)
    for v in adj[u]:
        in_degree[v] -= 1
        if in_degree[v] == 0:
            q.append(v)

# Longest path (DP)
dist = [1] * len(nodes)
for u in topo:
    for v in adj[u]:
        dist[v] = max(dist[v], dist[u] + 1)

critical_path = max(dist)
print(f"\n--- Critical Path ---", file=sys.stderr)
print(f"  Longest chain: {critical_path} nodes", file=sys.stderr)
print(f"  Total nodes: {len(nodes)}", file=sys.stderr)
print(f"  Parallelism: {len(nodes)/critical_path:.1f}x (avg width)", file=sys.stderr)

# 5. Fusion candidates
print(f"\n--- Fusion Candidates ---", file=sys.stderr)

# Elementwise chains (Add, Mul, Div, Sub, Erf)
ew_ops = {"Add", "Mul", "Div", "Sub", "Erf", "Relu"}
ew_chains = []
current_chain = []
for node in nodes:
    if node.op_type in ew_ops:
        current_chain.append(node.name)
    else:
        if len(current_chain) >= 3:
            ew_chains.append(current_chain)
        current_chain = []

print(f"  EW chains (>=3): {len(ew_chains)}", file=sys.stderr)

# MatMul + Add (MatMulBias)
mm_add_pairs = 0
for node in nodes:
    if node.op_type == "MatMul":
        for out_name in node.output:
            for consumer in nodes:
                if out_name in consumer.input and consumer.op_type == "Add":
                    mm_add_pairs += 1
print(f"  MatMul+Bias pairs: {mm_add_pairs}", file=sys.stderr)

# Identity aliasing (weight sharing)
identities = sum(1 for n in nodes if n.op_type == "Identity")
print(f"  Identity (shared weight) nodes: {identities}", file=sys.stderr)

# 6. Memory: intermediate tensor count
init_names = {i.name for i in m.graph.initializer}
intermediate_count = sum(1 for node in nodes for out in node.output if out not in init_names)
print(f"\n--- Memory ---", file=sys.stderr)
print(f"  Intermediate tensors: {intermediate_count}", file=sys.stderr)

# 7. Data flow: read-after-write distance per tensor
tensor_first_use = {}
tensor_last_use = {}
for ni, node in enumerate(nodes):
    for out_name in node.output:
        if out_name not in init_names:
            tensor_first_use[out_name] = ni
            tensor_last_use[out_name] = ni
    for inp in node.input:
        if inp in tensor_first_use:
            tensor_last_use[inp] = max(tensor_last_use.get(inp, 0), ni)

lifetimes = {t: (f, tensor_last_use.get(t, f)) for t, f in tensor_first_use.items()}
avg_lifetime = sum(l - f for f, l in lifetimes.values()) / max(len(lifetimes), 1)
print(f"  Avg tensor lifetime: {avg_lifetime:.1f} nodes", file=sys.stderr)
print(f"  Tensors with lifetime > 10: {sum(1 for f,l in lifetimes.values() if l-f > 10)}", file=sys.stderr)
