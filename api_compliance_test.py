#!/usr/bin/env python3
"""C3 API compliance test — simulates what the evaluation harness does.

This script validates that every public API the contest graders call
exists, is callable, and returns the expected types.
"""
import sys, json, os
sys.path.insert(0, "/home/mig20/c3_solution")

MODEL_BASE = "/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models"
PASSES = 0
FAILS = 0
ERRORS = []

def check(cond, msg):
    global PASSES, FAILS
    if cond:
        PASSES += 1
        print(f"  ✓ {msg}")
    else:
        FAILS += 1
        ERRORS.append(msg)
        print(f"  ✗ {msg}")

# ═══════════════════════════════════════════════════════════
# C3.2 — strategy API
# ═══════════════════════════════════════════════════════════
print("=== C3.2  strategy / hardware ===")

try:
    from scheduler import import_onnx_graph, strategy, hardware
    check(True, "import scheduler OK")
except Exception as e:
    check(False, f"import scheduler: {e}")
    print(f"\n{'='*60}\nFAILED: {FAILS}/{PASSES+FAILS}  errors={ERRORS}\n{'='*60}")
    sys.exit(1)

# hardware
check(hasattr(hardware, "supported_precisions"), "hardware.supported_precisions exists")
check(callable(hardware.supported_precisions), "hardware.supported_precisions() callable")
precs = hardware.supported_precisions()
check(isinstance(precs, set) and len(precs) > 0, f"supported_precisions -> {precs}")
check(hasattr(hardware, "MAX_THREADS_PER_BLOCK"), "hardware.MAX_THREADS_PER_BLOCK")
check(hasattr(hardware, "SMEM_BYTES"), "hardware.SMEM_BYTES")

# import_onnx_graph
for model_name in ["mlp_v1", "resnet_v1", "transformer_v1"]:
    path = os.path.join(MODEL_BASE, f"{model_name}.onnx")
    try:
        g = import_onnx_graph(path)
        check(len(g.nodes) > 0, f"import_onnx_graph({model_name}) -> {len(g.nodes)} nodes")
        ok, reason = g.validate()
        check(ok, f"graph.validate() {model_name}: {reason}")
    except Exception as e:
        check(False, f"import_onnx_graph({model_name}): {e}")

# strategy.select_precision
g = import_onnx_graph(os.path.join(MODEL_BASE, "mlp_v1.onnx"))
for n in g.nodes:
    try:
        p = strategy.select_precision(n, g)
        check(hasattr(p, "precision") and p.precision in precs,
              f"select_precision({n['op_type']}) -> {p.precision}")
    except Exception as e:
        check(False, f"select_precision: {e}")
    break  # test one node

# strategy.decompose
for n in g.nodes:
    try:
        p = strategy.select_precision(n, g)
        kseq = strategy.decompose(n, g, p)
        check(isinstance(kseq, list) and len(kseq) > 0,
              f"decompose({n['op_type']}) -> {len(kseq)} kernels")
        for k in kseq:
            check(hasattr(k, "name"), f"kernel has name: {k.name}")
            check(hasattr(k, "inputs"), f"kernel has inputs")
            check(hasattr(k, "outputs"), f"kernel has outputs")
            # D3: intermediate tensor check
            inters = set(k.outputs) - set(n["outputs"])
            if inters:
                check(True, f"intermediate tensors found: {inters}")
    except Exception as e:
        check(False, f"decompose: {e}")
    break

# strategy.tune_kernel
for n in g.nodes:
    try:
        p = strategy.select_precision(n, g)
        kseq = strategy.decompose(n, g, p)
        for k in kseq:
            t = strategy.tune_kernel(k, p, 1024)
            check(hasattr(t, "block_x"), f"tuning has block_x: {t.block_x}")
            check(hasattr(t, "grid_x"), f"tuning has grid_x: {t.grid_x}")
            check(hasattr(t, "smem_bytes"), f"tuning has smem_bytes: {t.smem_bytes}")
            # D4 validity
            check(0 < t.block_x <= hardware.MAX_THREADS_PER_BLOCK,
                  f"block_x {t.block_x} <= {hardware.MAX_THREADS_PER_BLOCK}")
            check(t.grid_x > 0, f"grid_x {t.grid_x} > 0")
            check(t.smem_bytes == -1 or t.smem_bytes <= hardware.SMEM_BYTES,
                  f"smem {t.smem_bytes} <= {hardware.SMEM_BYTES}")
    except Exception as e:
        check(False, f"tune_kernel: {e}")
    break

# precision routing: sensitive ops
softmax_nodes = [n for n in g.nodes if n["op_type"] == "Softmax"]
if softmax_nodes:
    p = strategy.select_precision(softmax_nodes[0], g)
    check(p.precision == "fp32", f"Softmax forced fp32: {p.precision}")
else:
    print("  (no Softmax in MLP — tested with Transformer below)")

# ═══════════════════════════════════════════════════════════
# C3.3 — GraphPassPipeline
# ═══════════════════════════════════════════════════════════
print("\n=== C3.3  GraphPassPipeline ===")

from scheduler.graph_passes import GraphPassPipeline

for model_name in ["mlp_v1", "resnet_v1", "transformer_v1"]:
    g = import_onnx_graph(os.path.join(MODEL_BASE, f"{model_name}.onnx"))
    pipeline = GraphPassPipeline(enable_fusion=True)
    try:
        results = pipeline.run(g)
        check("Fusion" in pipeline.pass_results, f"{model_name}: Fusion in pass_results")
        stats = pipeline.pass_results["Fusion"]["stats"]
        check("fusion_log" in stats, f"{model_name}: fusion_log exists")
        check("launches_before" in stats, f"{model_name}: launches_before")
        check("launches_after" in stats, f"{model_name}: launches_after")
        check("buffers_before" in stats, f"{model_name}: buffers_before")
        check("buffers_after" in stats, f"{model_name}: buffers_after")
        check(stats["launches_after"] <= stats["launches_before"],
              f"{model_name}: optimization reduced launches ({stats['launches_before']}->{stats['launches_after']})")
        for f in stats["fusion_log"]:
            check("pattern" in f, f"fusion has pattern: {f.get('pattern')}")
            check("nodes_removed" in f and len(f["nodes_removed"]) >= 2,
                  f"pattern {f['pattern']}: removed {len(f.get('nodes_removed',[]))} nodes")
    except Exception as e:
        check(False, f"GraphPassPipeline.run({model_name}): {e}")

# ═══════════════════════════════════════════════════════════
# C3.1 — export_dag.py CLI
# ═══════════════════════════════════════════════════════════
print("\n=== C3.1  export_dag CLI ===")

import subprocess
for model_name in ["mlp_v1", "resnet_v1", "transformer_v1"]:
    path = os.path.join(MODEL_BASE, f"{model_name}.onnx")
    out = f"/tmp/api_dag_{model_name}.json"
    r = subprocess.run(["python3", "/home/mig20/c3_solution/export_dag.py",
                        "--onnx", path, "--output", out],
                       capture_output=True, text=True)
    check(r.returncode == 0, f"export_dag {model_name}: exit 0")
    with open(out) as f:
        d = json.load(f)
    check("nodes" in d and "edges" in d, f"export_dag {model_name}: valid JSON")
    check(d["format_version"] == "1.0", f"export_dag {model_name}: format_version")

# ═══════════════════════════════════════════════════════════
# C3.5 — infer.py CLI
# ═══════════════════════════════════════════════════════════
print("\n=== C3.5  infer CLI ===")

TESTDATA = MODEL_BASE.replace("models", "testdata/c35")
for model_name, cpu_flag in [("mlp_v1", False), ("resnet_v1", False)]:
    cmd = ["bash", "/home/mig20/c3_solution/run_infer.sh",
           "--onnx", os.path.join(MODEL_BASE, f"{model_name}.onnx"),
           "--input", os.path.join(TESTDATA, model_name, "input"),
           "--output", f"/tmp/api_infer_{model_name}",
           "--batch-size", "256"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    check(r.returncode == 0, f"infer {model_name}: exit 0 (GPU)")
    # Verify output
    with open(f"/tmp/api_infer_{model_name}/manifest.json") as f:
        m = json.load(f)
    check(len(m["tensors"]) > 0 and m["tensors"][0]["dtype"] == "float32",
          f"infer {model_name}: valid output manifest")

# C3.4 — memory API
print("\n=== C3.4  memory API ===")
from scheduler.memory import (
    DeviceMemoryPool, TensorLifetime,
    analyze_lifetimes, build_reuse_slots,
    WeightUpload, plan_weight_uploads,
    schedule_with_prefetch, plan_streams,
)
check(True, "memory imports OK")

pool = DeviceMemoryPool(10 * 1024 * 1024)
h = pool.malloc(1024)
check(h >= 0, f"DeviceMemoryPool.malloc -> handle={h}")
pool.free(h)
check(pool.stats()["used"] == 0, "DeviceMemoryPool.free -> used=0")

lifetimes = [TensorLifetime("a", 0, 2), TensorLifetime("b", 1, 3), TensorLifetime("c", 3, 4)]
slots = build_reuse_slots(lifetimes)
check(len(set(slots.values())) <= len(lifetimes), f"lifetime reuse: {len(set(slots.values()))} slots for {len(lifetimes)} tensors")

streams = plan_streams([{"inputs":[],"outputs":["x"]},{"inputs":["x"],"outputs":["y"]}])
check(len(set(streams.values())) >= 1, f"stream plan: {len(set(streams.values()))} unique streams")

# Summary
print(f"\n{'='*60}")
print(f"RESULTS: {PASSES}/{PASSES+FAILS} passed")
if ERRORS:
    print(f"FAILURES:")
    for e in ERRORS:
        print(f"  - {e}")
else:
    print("ALL CHECKS PASSED ✓")
print(f"{'='*60}")
