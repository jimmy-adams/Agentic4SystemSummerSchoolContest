#!/usr/bin/env python3
"""C3.5 v2: Scheduler pipeline inference (replaces ONNX Runtime)."""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scheduler.planner import run_pipeline
import onnx


def load_inputs(d):
    with open(os.path.join(d, "manifest.json")) as f: m = json.load(f)
    return {e["name"]: np.load(os.path.join(d, e["file"])) for e in m["tensors"]}


def write_outputs(d, outputs):
    os.makedirs(d, exist_ok=True)
    m = {"tensors": []}
    for name, data in outputs.items():
        if data.dtype != np.float32: data = data.astype(np.float32)
        np.save(os.path.join(d, f"{name}.npy"), data)
        m["tensors"].append({"name": name, "file": f"{name}.npy", "dtype": "float32", "shape": list(data.shape)})
    with open(os.path.join(d, "manifest.json"), "w") as f: json.dump(m, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=2048)
    args = p.parse_args()

    model = onnx.load(args.onnx)
    init_names = {i.name for i in model.graph.initializer}
    output_names = [o.name for o in model.graph.output]

    all_inp = load_inputs(args.input)
    first = list(all_inp.keys())[0]
    N, B = all_inp[first].shape[0], min(args.batch_size, all_inp[first].shape[0])

    all_out = {n: [] for n in output_names}
    t0 = time.perf_counter()

    for start in range(0, N, B):
        end = min(start + B, N)
        batch = {k: v[start:end] for k, v in all_inp.items()}
        results, stats = run_pipeline(args.onnx, batch, output_names, B)
        for n, arr in results.items():
            all_out[n].append(arr)

    t1 = time.perf_counter()
    final = {n: np.concatenate(chunks, axis=0) for n, chunks in all_out.items()}
    write_outputs(args.output, final)
    print(f"Pipeline: {N} samples, {t1-t0:.2f}s, "
          f"kernels {stats['raw_kernels']}→{stats['opt_kernels']} "
          f"({stats['fusion_reduction']})", file=sys.stderr)


if __name__ == "__main__":
    main()
