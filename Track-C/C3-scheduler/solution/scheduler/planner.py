# scheduler/planner.py
"""End-to-end pipeline: ONNX → graph → decompose → fuse → decompose → execute."""

import copy
from typing import Dict, List

import numpy as np

from .graph import import_onnx_graph, Graph
from .strategy import strategy
from .graph_passes import GraphPassPipeline
from .executor import GPUExecutor


def decompose_graph(graph: Graph) -> list:
    """Decompose every graph node into a flat kernel sequence."""
    plan = []
    for node in graph.nodes:
        prec = strategy.select_precision(node, graph)
        kernels = strategy.decompose(node, graph, prec)
        for k in kernels:
            _ = strategy.tune_kernel(k, prec, 1024)
            plan.append({
                "name": k.name,
                "inputs": list(k.inputs),
                "outputs": list(k.outputs),
                "op_type": k.op_type,
            })
    return plan


def apply_fusion(graph: Graph, raw_plan: list) -> tuple:
    """Fuse at kernel level: replace fused node kernels with single fused kernel,
    preserving execution order."""
    pipeline = GraphPassPipeline(enable_fusion=True)
    pipeline.run(graph)
    stats = pipeline.pass_results["Fusion"]["stats"]
    fusion_log = stats["fusion_log"]

    if not fusion_log:
        return raw_plan, stats

    # Map: node_name → True if removed by fusion
    removed_nodes = set()
    # Map: first_removed_node → fusion_entry (for ordering)
    node_to_fusion = {}
    for f in fusion_log:
        for n in f["nodes_removed"]:
            removed_nodes.add(n)
            if n not in node_to_fusion:
                node_to_fusion[n] = f

    # Map kernel output → source node
    node_outputs = {}
    for n in graph.nodes:
        for o in n["outputs"]:
            node_outputs[o] = n["name"]

    # Build optimized plan: replace fused kernels in-place
    opt_plan = []
    seen_fusions = set()
    i = 0
    while i < len(raw_plan):
        k = raw_plan[i]
        src = None
        for o in k["outputs"]:
            if o in node_outputs:
                src = node_outputs[o]
                break

        if src and src in removed_nodes:
            # This kernel belongs to a fused node
            fusion = node_to_fusion.get(src)
            if fusion and id(fusion) not in seen_fusions:
                seen_fusions.add(id(fusion))
                opt_plan.append({
                    "name": fusion["fused_op"],
                    "inputs": fusion["inputs"],
                    "outputs": fusion["outputs"],
                    "op_type": fusion["fused_op"],
                })
            # Skip ALL kernels from this fusion's removed nodes
            # (only add fused kernel once at first encounter)
        else:
            opt_plan.append(k)
        i += 1

    return opt_plan, stats


def _rebuild_edges(g: Graph):
    """Rebuild edges for a graph from its nodes."""
    producer = {}
    for inp in g.graph_inputs:
        producer[inp["name"]] = None
    for name in g.initializer_names:
        producer[name] = None
    for n in g.nodes:
        for o in n.get("outputs", []):
            producer[o] = n["name"]

    edges = []
    for n in g.nodes:
        for inp_name in n.get("inputs", []):
            src = producer.get(inp_name)
            if src is not None:
                edges.append({"src_node": src, "dst_node": n["name"], "tensor": inp_name})
    g.edges = edges
    g._rebuild_index()


def run_pipeline(onnx_path: str, input_tensors: Dict[str, np.ndarray],
                 output_names: List[str], batch_size: int = 2048) -> tuple:
    """Full pipeline: ONNX → graph → decompose → fuse → decompose → execute."""
    # C3.1: Parse
    graph = import_onnx_graph(onnx_path)

    # C3.2: Raw decompose
    raw_plan = decompose_graph(graph)

    # C3.3: Fuse
    opt_plan, fusion_stats = apply_fusion(graph, raw_plan)

    # Execute
    executor = GPUExecutor(onnx_path)
    executor.load_inputs(input_tensors)
    results = executor.execute_plan(opt_plan, output_names)

    return results, {
        "raw_kernels": len(raw_plan),
        "opt_kernels": len(opt_plan),
        "fusion_reduction": f"{(1 - len(opt_plan)/max(len(raw_plan),1))*100:.0f}%",
        "launches": executor.launch_count,
    }
