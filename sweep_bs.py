#!/usr/bin/env python3
"""Batch size sweep for throughput optimization."""
import subprocess, time, os

MODELS = {
    "mlp_v1": "/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models/mlp_v1.onnx",
    "resnet_v1": "/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models/resnet_v1.onnx",
    "transformer_v1": "/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models/transformer_v1.onnx",
}
INPUT_BASE = "/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35"

for bs in [256, 512, 1024, 2048]:
    total = 0
    print(f"\n--- batch_size={bs} ---")
    for name, model in MODELS.items():
        inp = f"{INPUT_BASE}/{name}/input"
        t0 = time.perf_counter()
        r = subprocess.run(
            ["bash", "/home/mig20/c3_solution/run_infer.sh",
             "--onnx", model, "--input", inp,
             "--output", f"/tmp/bs_{bs}_{name}",
             "--batch-size", str(bs)],
            capture_output=True, text=True, timeout=120)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        total += elapsed
        print(f"  {name:20s}: {elapsed:.2f}s")
    print(f"  {'TOTAL':20s}: {total:.2f}s")
