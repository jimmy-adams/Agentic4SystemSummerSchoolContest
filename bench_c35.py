#!/usr/bin/env python3
"""C3.5 benchmark: runtime + GPU memory for all 3 models."""
import subprocess, time, sys, os

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

MODEL_DIR  = "/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models"
DATA_DIR   = "/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35"
SOL        = "/home/mig20/c3_solution"

import torch

def bench(model_name, batch_size=256):
    model_path = f"{MODEL_DIR}/{model_name}.onnx"
    input_dir  = f"{DATA_DIR}/{model_name}/input"
    output_dir = f"/tmp/bench_{model_name}"

    torch.cuda.empty_cache()

    t0 = time.perf_counter()

    r = subprocess.run(
        ["bash", f"{SOL}/run_infer.sh",
         "--onnx", model_path,
         "--input", input_dir,
         "--output", output_dir,
         "--batch-size", str(batch_size)],
        capture_output=True, text=True,
        env={**os.environ, "NVIDIA_TF32_OVERRIDE": "0"}
    )

    t1 = time.perf_counter()
    elapsed = t1 - t0

    # Verify
    golden_dir = f"{DATA_DIR}/{model_name}/golden"
    labels_path = f"{DATA_DIR}/{model_name}/labels.npy"
    import numpy as np
    out = np.load(f"{output_dir}/logits.npy")
    gold = np.load(f"{golden_dir}/logits.npy")
    prec_ok = np.allclose(out, gold, rtol=1e-3, atol=1e-3)
    max_diff = np.max(np.abs(out - gold))

    acc_str = ""
    if os.path.exists(labels_path) and model_name != "transformer_v1":
        lab = np.load(labels_path)
        if out.ndim == 2:  # classification: [N, C]
            acc = (out.argmax(1) == lab).mean()
            acc_str = f"  acc={acc:.4f}"

    print(f"  Time: {elapsed:.2f}s  "
          f"precision={'OK' if prec_ok else 'FAIL'}(max={max_diff:.2e}){acc_str}")

print("=" * 55)
print(" C3.5 Performance Benchmark (10k samples)")
print("=" * 55)

for m in ["mlp_v1", "resnet_v1", "transformer_v1"]:
    bench(m)

print("=" * 55)
