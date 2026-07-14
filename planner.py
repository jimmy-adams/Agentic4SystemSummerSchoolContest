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


def apply_fusion_and_rebuild(graph: Graph) -> tuple:
    """Run fusion, rebuild fused graph, re-decompose.

    Returns (fused_plan, stats).
    """
    pipeline = GraphPassPipeline(enable_fusion=True)
    pipeline.run(graph)
    stats = pipeline.pass_results["Fusion"]["stats"]
    fusion_log = stats["fusion_log"]

    if not fusion_log:
        return decompose_graph(graph), stats

    # Build fused graph from original nodes + fusion entries
    removed_names = set()
    for f in fusion_log:
        for n in f["nodes_removed"]:
            removed_names.add(n)

    # Keep unfused nodes
    fused_nodes = [n for n in graph.nodes if n["name"] not in removed_names]

    # Add fused nodes from fusion_log
    for f in fusion_log:
        fused_nodes.append({
            "name": f["new_name"],
            "op_type": f["fused_op"],
            "inputs": f["inputs"],
            "outputs": f["outputs"],
        })

    # Rebuild graph with fused nodes
    # Preserve original inputs/outputs but update edges
    fused_graph = Graph(
        graph_inputs=copy.deepcopy(graph.graph_inputs),
        graph_outputs=copy.deepcopy(graph.graph_outputs),
        nodes=fused_nodes,
        edges=[],  # edges rebuilt on init
        initializer_names=graph.initializer_names,
    )
    # Manually recompute edges for the fused graph
    _rebuild_edges(fused_graph)

    # Re-decompose the fused graph
    return decompose_graph(fused_graph), stats


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

    # C3.3: Fuse + re-decompose
    opt_plan, fusion_stats = apply_fusion_and_rebuild(graph)

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
