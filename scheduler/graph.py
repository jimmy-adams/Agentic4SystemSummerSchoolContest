# scheduler/graph.py
"""Graph representation for ONNX models."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import onnx


@dataclass
class Graph:
    """Parsed ONNX computation graph as DAG with helper methods."""

    format_version: str = "1.0"
    graph_inputs: List[dict] = field(default_factory=list)
    graph_outputs: List[dict] = field(default_factory=list)
    nodes: List[dict] = field(default_factory=list)
    edges: List[dict] = field(default_factory=list)
    initializer_names: Set[str] = field(default_factory=set)  # weights/constants

    # Internal maps (built post-parse)
    _node_by_name: Dict[str, dict] = field(default_factory=dict)
    _consumers: Dict[str, List[str]] = field(default_factory=dict)  # tensor -> [node_name, …]
    _producer: Dict[str, Optional[str]] = field(default_factory=dict)  # tensor -> node_name | None

    def __post_init__(self):
        self._rebuild_index()

    def _rebuild_index(self):
        self._node_by_name = {n["name"]: n for n in self.nodes}
        self._consumers = {}
        self._producer = {}

        # Producers: nodes
        for n in self.nodes:
            for o in n.get("outputs", []):
                self._producer[o] = n["name"]
        # Input & initializer tensors
        for inp in self.graph_inputs:
            self._producer[inp["name"]] = None
        for name in self.initializer_names:
            self._producer[name] = None

        # Consumers
        for n in self.nodes:
            for i in n.get("inputs", []):
                self._consumers.setdefault(i, []).append(n["name"])

    def get_node(self, name: str) -> Optional[dict]:
        return self._node_by_name.get(name)

    def topological_order(self) -> List[str]:
        """Return node names in topological order."""
        indeg: Dict[str, int] = {n["name"]: 0 for n in self.nodes}
        for n in self.nodes:
            for i in n.get("inputs", []):
                for c in self._consumers.get(i, []):
                    indeg[c] = indeg.get(c, 0)  # consumer is a node
        # Rebuild indegree from edges
        indeg = {n["name"]: 0 for n in self.nodes}
        for e in self.edges:
            dst = e["dst_node"]
            if dst in indeg:
                indeg[dst] += 1

        # Start with nodes that have no incoming edges (inputs go to them)
        q = [n for n, d in indeg.items() if d == 0]
        order = []
        while q:
            u = q.pop(0)
            order.append(u)
            for o in self._node_by_name[u].get("outputs", []):
                for c in self._consumers.get(o, []):
                    indeg[c] -= 1
                    if indeg[c] == 0:
                        q.append(c)
        return order

    def validate(self) -> Tuple[bool, str]:
        """Graph soundness check."""
        # Check no cycles (topological sort should include all nodes)
        order = self.topological_order()
        if len(order) != len(self.nodes):
            return False, "Graph has cycles"

        # Check all referenced tensors have producers or are graph inputs/weights
        input_names = {i["name"] for i in self.graph_inputs}
        input_names |= self.initializer_names
        all_producers = set(self._producer.keys())
        for n in self.nodes:
            for i in n.get("inputs", []):
                if i not in all_producers and i not in input_names:
                    return False, f"Tensor '{i}' in node '{n['name']}' has no producer"
        return True, "ok"


def import_onnx_graph(model_path: str) -> Graph:
    """Load an ONNX model and return a Graph DAG.

    This is the public entry-point called by the evaluation script:
        graph = import_onnx_graph("model.onnx")
    """
    model = onnx.load(model_path)
    g = model.graph
    init_names = set(init.name for init in g.initializer)

    DTYPE_MAP = {1: "FLOAT", 6: "INT32", 7: "INT64", 10: "FLOAT16"}

    graph_inputs = []
    for inp in g.input:
        if inp.name in init_names:
            continue
        shape = [d.dim_value if d.dim_value else d.dim_param
                 for d in inp.type.tensor_type.shape.dim]
        dtype = DTYPE_MAP.get(inp.type.tensor_type.elem_type, "UNKNOWN")
        graph_inputs.append({"name": inp.name, "dtype": dtype, "shape": shape})

    graph_outputs = []
    for out in g.output:
        shape = [d.dim_value if d.dim_value else d.dim_param
                 for d in out.type.tensor_type.shape.dim]
        dtype = DTYPE_MAP.get(out.type.tensor_type.elem_type, "UNKNOWN")
        graph_outputs.append({"name": out.name, "dtype": dtype, "shape": shape})

    nodes = []
    for node in g.node:
        nodes.append({
            "name": node.name,
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
        })

    # Build edges
    tensor_producer = {}
    for inp in g.input:
        if inp.name not in init_names:
            tensor_producer[inp.name] = None
    for init in g.initializer:
        tensor_producer[init.name] = None
    for node in g.node:
        for out_name in node.output:
            tensor_producer[out_name] = node.name

    edges = []
    for node in g.node:
        for inp_name in node.input:
            src = tensor_producer.get(inp_name)
            if src is not None:
                edges.append({"src_node": src, "dst_node": node.name, "tensor": inp_name})

    return Graph(
        format_version="1.0",
        graph_inputs=graph_inputs,
        graph_outputs=graph_outputs,
        nodes=nodes,
        edges=edges,
        initializer_names=init_names,
    )
