#!/usr/bin/env python3
"""C3.1: ONNX Model -> DAG JSON exporter."""
import argparse
import json
import onnx

DTYPE_MAP = {
    1: "FLOAT", 2: "UINT8", 3: "INT8", 4: "UINT16", 5: "INT16",
    6: "INT32", 7: "INT64", 9: "BOOL", 10: "FLOAT16",
    11: "DOUBLE", 12: "UINT32", 13: "UINT64",
}

def export_dag(model_path, output_path):
    model = onnx.load(model_path)
    graph = model.graph
    init_names = set(init.name for init in graph.initializer)

    # Graph inputs (exclude weights/initializers)
    graph_inputs = []
    for inp in graph.input:
        if inp.name in init_names:
            continue
        shape = []
        for d in inp.type.tensor_type.shape.dim:
            shape.append(d.dim_value if d.dim_value else d.dim_param)
        dtype = DTYPE_MAP.get(inp.type.tensor_type.elem_type, "UNKNOWN")
        graph_inputs.append({"name": inp.name, "dtype": dtype, "shape": shape})

    # Graph outputs
    graph_outputs = []
    for out in graph.output:
        shape = []
        for d in out.type.tensor_type.shape.dim:
            shape.append(d.dim_value if d.dim_value else d.dim_param)
        dtype = DTYPE_MAP.get(out.type.tensor_type.elem_type, "UNKNOWN")
        graph_outputs.append({"name": out.name, "dtype": dtype, "shape": shape})

    # Nodes
    nodes = []
    for node in graph.node:
        nodes.append({
            "name": node.name,
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
        })

    # Build tensor -> producer map for edges
    tensor_producer = {}
    for inp in graph.input:
        if inp.name not in init_names:
            tensor_producer[inp.name] = None  # graph input, no producer node
    for init in graph.initializer:
        tensor_producer[init.name] = None  # weight constant
    for node in graph.node:
        for out_name in node.output:
            tensor_producer[out_name] = node.name

    # Edges
    edges = []
    for node in graph.node:
        for inp_name in node.input:
            src = tensor_producer.get(inp_name)
            if src is not None:
                edges.append({
                    "src_node": src,
                    "dst_node": node.name,
                    "tensor": inp_name,
                })

    dag = {
        "format_version": "1.0",
        "graph_inputs": graph_inputs,
        "graph_outputs": graph_outputs,
        "nodes": nodes,
        "edges": edges,
    }

    with open(output_path, "w") as f:
        json.dump(dag, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export_dag(args.onnx, args.output)
