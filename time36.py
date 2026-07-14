#!/usr/bin/env python3
import subprocess, time

MODELS = {
    "mlp_v1": 256,
    "resnet_v1": 256,
    "transformer_v1": 256,
}
BASE = "/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors"
SOL = "/home/mig20/c3_solution"

total = 0
for name, bs in MODELS.items():
    model = f"{BASE}/models/{name}.onnx"
    inp   = f"{BASE}/testdata/c35/{name}/input"
    out   = f"/tmp/timing_{name}"
    t0 = time.perf_counter()
    r = subprocess.run(
        ["bash", f"{SOL}/run_infer.sh",
         "--onnx", model, "--input", inp,
         "--output", out, "--batch-size", str(bs)],
        capture_output=True, text=True)
    t1 = time.perf_counter()
    e = t1 - t0
    total += e
    # Extract provider info
    for line in r.stderr.split("\n"):
        if "providers" in line or "Inference" in line:
            print(f"  {line.strip()}")
    print(f"{name:20s}: {e:.2f}s")
print(f"{'TOTAL':20s}: {total:.2f}s")
