#!/usr/bin/env python3
"""C3.5: End-to-end ONNX model inference on GPU via ONNX Runtime.

Usage:
    python infer.py --onnx {onnx} --input {input} --output {output} --batch-size 256

Environment:
    Set NVIDIA_TF32_OVERRIDE=0 to disable TF32 for Ampere+ GPUs.
    Set LD_LIBRARY_PATH with CUDA/cuDNN paths for ORT GPU provider.
    Use run_infer.sh wrapper for convenience.
"""
import argparse
import json
import os
import sys

# Disable TF32: Ampere+ defaults cause precision drift in deep nets.
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

import numpy as np
import onnx
import onnxruntime as ort


def load_input_tensors(input_dir: str) -> dict:
    """Load input tensors from manifest.json + .npy files."""
    manifest_path = os.path.join(input_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    tensors = {}
    for entry in manifest["tensors"]:
        npy_path = os.path.join(input_dir, entry["file"])
        tensors[entry["name"]] = np.load(npy_path)
    return tensors


def write_output_tensors(output_dir: str, outputs: dict):
    """Write output tensors in manifest.json + .npy format."""
    os.makedirs(output_dir, exist_ok=True)
    manifest = {"tensors": []}
    for name, data in outputs.items():
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        npy_file = f"{name}.npy"
        np.save(os.path.join(output_dir, npy_file), data)
        manifest["tensors"].append({
            "name": name,
            "file": npy_file,
            "dtype": "float32",
            "shape": list(data.shape),
        })
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="C3.5 ONNX Model Inference")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--input", required=True, help="Input directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU execution (for precision-sensitive deep models)")
    args = parser.parse_args()

    # Determine available providers
    if args.cpu:
        providers = [("CPUExecutionProvider", {})]
    else:
        available = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available:
            # TF32 is disabled via NVIDIA_TF32_OVERRIDE=0 env var
            providers.append(("CUDAExecutionProvider", {"device_id": "0"}))
        providers.append(("CPUExecutionProvider", {}))

    # Create session with graph optimization
    sess_options = ort.SessionOptions()
    if args.cpu:
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    else:
        # GPU: ORT_ENABLE_EXTENDED — better than BASIC, but avoids FULL's
        # aggressive mixed-precision rewrites that break 1e-3 gate.
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.intra_op_num_threads = 8

    session = ort.InferenceSession(
        args.onnx,
        sess_options=sess_options,
        providers=providers,
    )

    # Get input/output metadata
    input_metas = session.get_inputs()
    output_metas = session.get_outputs()
    input_names = [m.name for m in input_metas]
    output_names = [m.name for m in output_metas]

    # Load input tensors
    input_tensors = load_input_tensors(args.input)

    # Map ONNX input names to loaded data
    onnx_to_np = {}
    for onnx_name in input_names:
        if onnx_name in input_tensors:
            onnx_to_np[onnx_name] = input_tensors[onnx_name]
        else:
            for k, v in input_tensors.items():
                if k.lower() == onnx_name.lower():
                    onnx_to_np[onnx_name] = v
                    break

    # Determine total sample count
    first_input = list(onnx_to_np.values())[0]
    N = first_input.shape[0]
    batch_size = min(args.batch_size, N)

    # ── Warm-up: run a small batch to compile CUDA kernels ─────────────────
    if "CUDA" in str(session.get_providers()):
        warm_feed = {}
        for onnx_name, np_data in onnx_to_np.items():
            warm_feed[onnx_name] = np_data[:min(8, N)]
        _ = session.run(output_names, warm_feed)

    # Run inference in batches
    all_outputs = {name: [] for name in output_names}

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        feed = {}
        for onnx_name, np_data in onnx_to_np.items():
            feed[onnx_name] = np_data[start:end]

        results = session.run(output_names, feed)
        for name, arr in zip(output_names, results):
            all_outputs[name].append(arr)

    # Concatenate
    final_outputs = {}
    for name, chunks in all_outputs.items():
        final_outputs[name] = np.concatenate(chunks, axis=0)

    # Write output
    write_output_tensors(args.output, final_outputs)
    active_providers = session.get_providers()
    dev = "cuda" if any("CUDA" in p for p in active_providers) else "cpu"
    print(f"Inference complete: {N} samples, batch_size={batch_size}, "
          f"providers={active_providers}, device={dev}", file=sys.stderr)


if __name__ == "__main__":
    main()
