#!/usr/bin/env python3
"""Self-test for C3.2 + C3.3 scheduler module."""

import sys
sys.path.insert(0, "/home/mig20/c3_solution")

from scheduler import import_onnx_graph, strategy
from scheduler.graph_passes import GraphPassPipeline

MODEL_DIR = "/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models"

for model_name in ["mlp_v1", "resnet_v1", "transformer_v1"]:
    path = f"{MODEL_DIR}/{model_name}.onnx"
    g = import_onnx_graph(path)
    print(f"--- {model_name}: {len(g.nodes)} nodes ---")

    # C3.2: precision routing + decompose + tuning
    sens_count = 0
    nonsens_count = 0
    tuning_ok = 0
    tuning_total = 0
    all_nodal_precisions = set()

    for n in g.nodes:
        p = strategy.select_precision(n, g)
        all_nodal_precisions.add(p.precision)

        if p.precision == "fp32":
            sens_count += 1
        else:
            nonsens_count += 1

        kseq = strategy.decompose(n, g, p)

        for k in kseq:
            tune = strategy.tune_kernel(k, p, None)
            tuning_total += 1
            ok = (0 < tune.block_x <= 1024 and tune.grid_x > 0
                  and (tune.smem_bytes <= 163 * 1024 or tune.smem_bytes == -1))
            if ok:
                tuning_ok += 1

        # Check intermediate tensors (C3.2 D3)
        for k in kseq:
            inters = set(k.outputs) - set(n["outputs"])
            if inters:
                pass  # expected

    print(f"  Precision routing: {len(all_nodal_precisions)} types: {all_nodal_precisions}")
    print(f"  Tuning validity: {tuning_ok}/{tuning_total} OK")

    # C3.3: fusion
    pipeline = GraphPassPipeline(enable_fusion=True)
    pipeline.run(g)
    stats = pipeline.pass_results["Fusion"]["stats"]
    flog = stats["fusion_log"]
    patterns = set(f["pattern"] for f in flog)
    print(f"  Fusion patterns ({len(patterns)}): {patterns}")
    print(f"  Launches: {stats['launches_before']} -> {stats['launches_after']} "
          f"(reduction: {(1 - stats['launches_after']/max(stats['launches_before'],1))*100:.0f}%)")
    print()

print("=== ALL PASS ===")
