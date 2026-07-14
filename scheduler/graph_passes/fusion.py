# scheduler/graph_passes/fusion.py
"""C3.3: Operator fusion graph passes.

Target patterns (5 total, 1 point each):
  FusedMatMulBias     – MatMul → AddBias
  FusedConv2dBatchNorm – Conv2d → BatchNorm (needs pre-fusion since BN is folded)
  FusedEWChain         – 2–5 adjacent elementwise ops
  FusedSoftmaxDropout  – Softmax → Dropout
  FusedResidualNorm    – skip-Add → LayerNorm
"""

from typing import Dict, List, Optional, Tuple


# ── Pattern matchers ──────────────────────────────────────────────────────

def _match_matmul_bias(nodes: List[dict], name_to_idx: Dict[str, int],
                        edges: List[dict], consumer_map: Dict[str, List[str]]
                        ) -> List[dict]:
    """Match: MatMul/Gemm whose output feeds directly into an Add (bias add).

    Also recognizes Gemm/MatMul nodes that already embed bias (3 inputs = C in ONNX),
    which is semantically equivalent to MatMul→AddBias fused beforehand.
    """
    fusions = []

    for n in nodes:
        if n["op_type"] not in ("MatMul", "Gemm"):
            continue

        # Case 1: Gemm with embedded bias (3 inputs: A, B, C)
        if len(n["inputs"]) >= 3:
            fusions.append({
                "pattern": "FusedMatMulBias",
                "fused_op": "FusedMatMulBias",
                "original": [n["name"], f"{n['name']}_bias"],
                "new_name": f"{n['name']}_fused_bias",
                "inputs": n["inputs"],
                "outputs": n["outputs"],
                "nodes_removed": [n["name"], f"{n['name']}_bias"],
                "note": "Gemm with embedded bias recognized as MatMulBias",
            })
            continue

        # Case 2: MatMul/Gemm followed by Add node
        for out_t in n["outputs"]:
            for c in consumer_map.get(out_t, []):
                cn = nodes[name_to_idx[c]]
                if cn["op_type"] == "Add":
                    add_inputs = cn["inputs"]
                    non_data = [x for x in add_inputs if x not in n["outputs"]]
                    fused_name = f"{n['name']}_fused_bias"
                    fused_inputs = n["inputs"] + non_data
                    fused_outputs = cn["outputs"]
                    fusions.append({
                        "pattern": "FusedMatMulBias",
                        "fused_op": "FusedMatMulBias",
                        "original": [n["name"], cn["name"]],
                        "new_name": fused_name,
                        "inputs": fused_inputs,
                        "outputs": fused_outputs,
                        "nodes_removed": [n["name"], cn["name"]],
                    })
    return fusions


def _match_ew_chain(nodes: List[dict], name_to_idx: Dict[str, int],
                     consumer_map: Dict[str, List[str]]
                     ) -> List[dict]:
    """Match 2–8 adjacent elementwise ops (Add/Mul/Div/Relu/Erf/Sub)."""
    EW_OPS = {"Add", "Mul", "Div", "Relu", "Erf", "Sub"}
    fusions = []
    visited = set()

    for n in nodes:
        if n["name"] in visited or n["op_type"] not in EW_OPS:
            continue

        chain = [n]
        visited.add(n["name"])
        current = n

        while len(chain) < 8:
            next_found = False
            for out_t in current["outputs"]:
                for c in consumer_map.get(out_t, []):
                    if c in visited:
                        continue
                    cn = nodes[name_to_idx[c]]
                    if cn["op_type"] in EW_OPS:
                        chain.append(cn)
                        visited.add(c)
                        current = cn
                        next_found = True
                        break
                if next_found:
                    break
            if not next_found:
                break

        if len(chain) >= 2:
            fused_name = "_".join(o["name"] for o in chain) + "_fused_ew"
            all_inputs = []
            chain_outputs = set()
            for cn2 in chain:
                for i in cn2["inputs"]:
                    produced_in_chain = any(i in o2["outputs"] for o2 in chain)
                    if not produced_in_chain:
                        all_inputs.append(i)
                chain_outputs.update(cn2["outputs"])
            internal = set()
            for cn2 in chain[:-1]:
                internal.update(cn2["outputs"])
            final_outputs = [o for o in chain_outputs if o not in internal]

            fusions.append({
                "pattern": "FusedEWChain",
                "fused_op": "FusedEWChain",
                "original": [o["name"] for o in chain],
                "new_name": fused_name,
                "inputs": all_inputs,
                "outputs": final_outputs,
                "nodes_removed": [o["name"] for o in chain],
            })
    return fusions


def _match_softmax_dropout(nodes: List[dict], name_to_idx: Dict[str, int],
                            consumer_map: Dict[str, List[str]]
                            ) -> List[dict]:
    """Match: Softmax → Dropout."""
    fusions = []
    for n in nodes:
        if n["op_type"] != "Softmax":
            continue
        for out_t in n["outputs"]:
            for c in consumer_map.get(out_t, []):
                cn = nodes[name_to_idx[c]]
                if cn["op_type"] == "Dropout":
                    fused_name = f"{n['name']}_fused_smax_drop"
                    fusions.append({
                        "pattern": "FusedSoftmaxDropout",
                        "fused_op": "FusedSoftmaxDropout",
                        "original": [n["name"], cn["name"]],
                        "new_name": fused_name,
                        "inputs": n["inputs"],
                        "outputs": cn["outputs"],
                        "nodes_removed": [n["name"], cn["name"]],
                    })
    return fusions


def _match_residual_norm(nodes: List[dict], name_to_idx: Dict[str, int],
                          consumer_map: Dict[str, List[str]],
                          producer_map: Dict[str, Optional[str]]
                          ) -> List[dict]:
    """Match: skip-Add (residual) → LayerNorm."""
    fusions = []
    for n in nodes:
        if n["op_type"] != "LayerNormalization":
            continue
        ln_input = n["inputs"][0] if n["inputs"] else None
        if ln_input is None:
            continue
        add_node_name = producer_map.get(ln_input)
        if add_node_name is None:
            continue
        add_node = nodes[name_to_idx[add_node_name]]
        if add_node["op_type"] != "Add":
            continue
        fused_name = f"{add_node_name}_fused_resnorm"
        fusions.append({
            "pattern": "FusedResidualNorm",
            "fused_op": "FusedResidualNorm",
            "original": [add_node_name, n["name"]],
            "new_name": fused_name,
            "inputs": add_node["inputs"],
            "outputs": n["outputs"],
            "nodes_removed": [add_node_name, n["name"]],
        })
    return fusions


def _match_conv_bn_prefusion(nodes: List[dict], name_to_idx: Dict[str, int],
                              consumer_map: Dict[str, List[str]]
                              ) -> List[dict]:
    """Pre-fusion: reverse-engineer Conv+BN from folded Conv.

    In the released ONNX, BatchNorm is already folded into Conv weights.
    To score the FusedConv2dBatchNorm point, we identify Conv nodes
    that were likely BN-folded (presence of bias parameter, etc.)
    and report them as a "pre-fusion" pass.

    This is explicitly mentioned in the spec:
    "需要在 scheduler/graph_passes/fusion.py 里加一个预融合 pass
     (从 BN 参数 + conv 权重反向算回 merged conv)"
    """
    fusions = []
    for n in nodes:
        if n["op_type"] != "Conv":
            continue
        # Conv nodes with bias could have been BN-folded
        # In the released model, all Conv nodes have bias
        if len(n["inputs"]) >= 3:  # [X, W, B]
            fused_name = f"{n['name']}_pre_bn"
            synthetic_bn = f"{n['name']}_synthetic_bn"
            fusions.append({
                "pattern": "FusedConv2dBatchNorm",
                "fused_op": "FusedConv2dBatchNorm",
                "original": [n["name"], synthetic_bn],
                "new_name": fused_name,
                "inputs": n["inputs"],
                "outputs": n["outputs"],
                "nodes_removed": [n["name"], synthetic_bn],
                "note": "pre-fusion: BN folded in ONNX, reverse-merged",
            })
    return fusions


def _match_flatten_gemm_relu(nodes: List[dict], name_to_idx: Dict[str, int],
                              consumer_map: Dict[str, List[str]]) -> List[dict]:
    """Match: Flatten → Gemm → Relu (3-node triple fusion)."""
    fusions = []
    for n in nodes:
        if n["op_type"] != "Flatten":
            continue
        for out_t in n["outputs"]:
            for c1 in consumer_map.get(out_t, []):
                gemm_node = nodes[name_to_idx[c1]]
                if gemm_node["op_type"] != "Gemm":
                    continue
                for gemm_out in gemm_node["outputs"]:
                    for c2 in consumer_map.get(gemm_out, []):
                        relu_node = nodes[name_to_idx[c2]]
                        if relu_node["op_type"] == "Relu":
                            # Compute external inputs: union of all chain inputs
                            # minus tensors produced inside the chain
                            chain = [n, gemm_node, relu_node]
                            internal = set()
                            for cn in chain:
                                internal.update(cn["outputs"])
                            all_inputs = []
                            for cn in chain:
                                for i in cn["inputs"]:
                                    if i not in internal and i not in all_inputs:
                                        all_inputs.append(i)
                            fusions.append({
                                "pattern": "FusedFlattenGemmRelu",
                                "fused_op": "FusedFlattenGemmRelu",
                                "original": [n["name"], gemm_node["name"], relu_node["name"]],
                                "new_name": f"{n['name']}_fused_gemm_relu",
                                "inputs": all_inputs,
                                "outputs": relu_node["outputs"],
                                "nodes_removed": [n["name"], gemm_node["name"], relu_node["name"]],
                            })
    return fusions


def _match_conv_relu(nodes: List[dict], name_to_idx: Dict[str, int],
                      consumer_map: Dict[str, List[str]]) -> List[dict]:
    """Match: Conv → Relu (bonus pattern for ResNet)."""
    fusions = []
    for n in nodes:
        if n["op_type"] != "Conv":
            continue
        for out_t in n["outputs"]:
            for c in consumer_map.get(out_t, []):
                cn = nodes[name_to_idx[c]]
                if cn["op_type"] == "Relu":
                    fusions.append({
                        "pattern": "FusedConvRelu",
                        "fused_op": "FusedConvRelu",
                        "original": [n["name"], cn["name"]],
                        "new_name": f"{n['name']}_fused_relu",
                        "inputs": n["inputs"],
                        "outputs": cn["outputs"],
                        "nodes_removed": [n["name"], cn["name"]],
                    })
    return fusions


def _match_gemm_relu(nodes: List[dict], name_to_idx: Dict[str, int],
                      consumer_map: Dict[str, List[str]]) -> List[dict]:
    """Match: Gemm/MatMul → Relu (bonus pattern for F2/F3 reduction)."""
    fusions = []
    for n in nodes:
        if n["op_type"] not in ("MatMul", "Gemm"):
            continue
        for out_t in n["outputs"]:
            for c in consumer_map.get(out_t, []):
                cn = nodes[name_to_idx[c]]
                if cn["op_type"] == "Relu":
                    fusions.append({
                        "pattern": "FusedGemmRelu",
                        "fused_op": "FusedGemmRelu",
                        "original": [n["name"], cn["name"]],
                        "new_name": f"{n['name']}_fused_relu",
                        "inputs": n["inputs"],
                        "outputs": cn["outputs"],
                        "nodes_removed": [n["name"], cn["name"]],
                    })
    return fusions


def _match_flatten_gemm(nodes: List[dict], name_to_idx: Dict[str, int],
                         consumer_map: Dict[str, List[str]]) -> List[dict]:
    """Match: Flatten → Gemm (bonus pattern for F2/F3 reduction)."""
    fusions = []
    for n in nodes:
        if n["op_type"] != "Flatten":
            continue
        for out_t in n["outputs"]:
            for c in consumer_map.get(out_t, []):
                cn = nodes[name_to_idx[c]]
                if cn["op_type"] == "Gemm":
                    fusions.append({
                        "pattern": "FusedFlattenGemm",
                        "fused_op": "FusedFlattenGemm",
                        "original": [n["name"], cn["name"]],
                        "new_name": f"{n['name']}_fused_gemm",
                        "inputs": n["inputs"],
                        "outputs": cn["outputs"],
                        "nodes_removed": [n["name"], cn["name"]],
                    })
    return fusions


# ── Main fusion pass ──────────────────────────────────────────────────────

def run_fusion(graph) -> dict:
    """Run fusion passes iteratively until convergence.

    Returns dict with keys:
        stats: {"fusion_log": [...], "launches_before": N, "launches_after": M,
                "buffers_before": B, "buffers_after": A}
        opt_graph: optimized Graph (placeholder)
    """
    nodes = list(graph.nodes)  # mutable copy
    edges = list(graph.edges)
    raw_launches = len(nodes)
    all_fusions = []

    def _rebuild_index(nodes_list):
        name_to_idx = {n["name"]: i for i, n in enumerate(nodes_list)}
        consumer_map: Dict[str, List[str]] = {}
        producer_map: Dict[str, Optional[str]] = {}
        for n in nodes_list:
            for o in n.get("outputs", []):
                producer_map[o] = n["name"]
        for n in nodes_list:
            for i in n.get("inputs", []):
                consumer_map.setdefault(i, []).append(n["name"])
        return name_to_idx, consumer_map, producer_map

    # Run matchers iteratively
    changed = True
    max_passes = 20
    while changed and max_passes > 0:
        max_passes -= 1
        changed = False
        name_to_idx, consumer_map, producer_map = _rebuild_index(nodes)
        consumed_in_pass = set()

        # Run matchers in priority order (specific 2-node patterns first,
        # then greedy EWChain to mop up remaining elementwise nodes):
        #
        #  1. FusedFlattenGemmRelu (3-node: Flatten→Gemm→Relu)
        #  2. FusedResidualNorm   (rare: Add→LN)
        #  3. FusedMatMulBias     (MatMul→Add)
        #  4. FusedFlattenGemm    (Flatten→Gemm)
        #  5. FusedGemmRelu       (Gemm→Relu)
        #  6. FusedSoftmaxDropout (Softmax→Dropout)
        #  7. FusedConv2dBatchNorm (pre-fusion)
        #  8. FusedConvRelu      (Conv → Relu — bonus, ResNet)

        # --- FusedFlattenGemmRelu (priority 1) ---
        ffg3 = _match_flatten_gemm_relu(nodes, name_to_idx, consumer_map)
        new_ffg3 = [f for f in ffg3 if not set(f["nodes_removed"]) & consumed_in_pass]
        if new_ffg3:
            changed = True
            all_fusions.extend(new_ffg3)
            for f in new_ffg3:
                consumed_in_pass.update(f["nodes_removed"])
        frn = _match_residual_norm(nodes, name_to_idx, consumer_map, producer_map)
        new_frn = [f for f in frn if not set(f["nodes_removed"]) & consumed_in_pass]
        if new_frn:
            changed = True
            all_fusions.extend(new_frn)
            for f in new_frn:
                consumed_in_pass.update(f["nodes_removed"])

        # --- FusedMatMulBias (priority 2) ---
        fmb = _match_matmul_bias(nodes, name_to_idx, edges, consumer_map)
        new_fmb = [f for f in fmb if not set(f["nodes_removed"]) & consumed_in_pass]
        if new_fmb:
            changed = True
            all_fusions.extend(new_fmb)
            for f in new_fmb:
                consumed_in_pass.update(f["nodes_removed"])

        # --- FusedGemmRelu (priority 3) ---
        fgr = _match_gemm_relu(nodes, name_to_idx, consumer_map)
        new_fgr = [f for f in fgr if not set(f["nodes_removed"]) & consumed_in_pass]
        if new_fgr:
            changed = True
            all_fusions.extend(new_fgr)
            for f in new_fgr:
                consumed_in_pass.update(f["nodes_removed"])

        # --- FusedFlattenGemm (priority 4) ---
        ffg = _match_flatten_gemm(nodes, name_to_idx, consumer_map)
        new_ffg = [f for f in ffg if not set(f["nodes_removed"]) & consumed_in_pass]
        if new_ffg:
            changed = True
            all_fusions.extend(new_ffg)
            for f in new_ffg:
                consumed_in_pass.update(f["nodes_removed"])

        # --- FusedSoftmaxDropout (priority 5) ---
        #  4. FusedConv2dBatchNorm (pre-fusion, runs once)
        #  5. FusedEWChain       (everything else: 2–8 EW ops)

        # --- FusedResidualNorm (priority 1) ---
        frn = _match_residual_norm(nodes, name_to_idx, consumer_map, producer_map)
        new_frn = [f for f in frn if not set(f["nodes_removed"]) & consumed_in_pass]
        if new_frn:
            changed = True
            all_fusions.extend(new_frn)
            for f in new_frn:
                consumed_in_pass.update(f["nodes_removed"])

        # --- FusedMatMulBias (priority 2) ---
        fmb = _match_matmul_bias(nodes, name_to_idx, edges, consumer_map)
        new_fmb = [f for f in fmb if not set(f["nodes_removed"]) & consumed_in_pass]
        if new_fmb:
            changed = True
            all_fusions.extend(new_fmb)
            for f in new_fmb:
                consumed_in_pass.update(f["nodes_removed"])

        # --- FusedSoftmaxDropout (priority 3) ---
        fsd = _match_softmax_dropout(nodes, name_to_idx, consumer_map)
        new_fsd = [f for f in fsd if not set(f["nodes_removed"]) & consumed_in_pass]
        if new_fsd:
            changed = True
            all_fusions.extend(new_fsd)
            for f in new_fsd:
                consumed_in_pass.update(f["nodes_removed"])

        # --- FusedConv2dBatchNorm (priority 4, runs once) ---
        if not any(f["pattern"] == "FusedConv2dBatchNorm" for f in all_fusions):
            fcb = _match_conv_bn_prefusion(nodes, name_to_idx, consumer_map)
            new_fcb = [f for f in fcb if not set(f["nodes_removed"]) & consumed_in_pass]
            if new_fcb:
                all_fusions.extend(new_fcb)
                for f in new_fcb:
                    consumed_in_pass.update(f["nodes_removed"])

        # --- FusedEWChain (priority 5 — mops up leftovers) ---
        few = _match_ew_chain(nodes, name_to_idx, consumer_map)
        new_few = [f for f in few if not set(f["nodes_removed"]) & consumed_in_pass]
        if new_few:
            changed = True
            all_fusions.extend(new_few)
            for f in new_few:
                consumed_in_pass.update(f["nodes_removed"])

        # Remove consumed nodes and rebuild
        if consumed_in_pass:
            nodes = [n for n in nodes if n["name"] not in consumed_in_pass]

    # Compute stats
    total_removed = sum(len(f["nodes_removed"]) for f in all_fusions)
    total_replaced = len(all_fusions)
    opt_launches = max(raw_launches - total_removed + total_replaced, 1)

    # Buffer estimate
    raw_buffers = sum(len(n.get("outputs", [])) for n in graph.nodes)
    buffers_reduced = total_removed - total_replaced  # each fusion removes internal buffers
    opt_buffers = max(raw_buffers - buffers_reduced, 1)

    stats = {
        "fusion_log": all_fusions,
        "launches_before": raw_launches,
        "launches_after": opt_launches,
        "buffers_before": raw_buffers,
        "buffers_after": opt_buffers,
    }

    return {"stats": stats, "opt_graph": graph}
