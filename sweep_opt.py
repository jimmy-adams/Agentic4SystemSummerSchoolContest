#!/usr/bin/env python3
"""ResNet optimization level sweep."""
import onnxruntime as ort, onnx, numpy as np, time, os
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

MODEL = "/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models/resnet_v1.onnx"
INPUT = "/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35/resnet_v1/input"
GOLDEN = "/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35/resnet_v1/golden/logits.npy"

model = onnx.load(MODEL)
inames = [i.name for i in model.graph.input if i.name not in {init.name for init in model.graph.initializer}]
onames = [o.name for o in model.graph.output]

import json
with open(f"{INPUT}/manifest.json") as f: m = json.load(f)
data = np.load(f"{INPUT}/{m['tensors'][0]['file']}")
golden = np.load(GOLDEN)

for level_name, level in [("BASIC", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
                           ("EXTENDED", ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED),
                           ("ALL", ort.GraphOptimizationLevel.ORT_ENABLE_ALL)]:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = level

    sess = ort.InferenceSession(MODEL, opts,
        providers=[("CUDAExecutionProvider", {"device_id": "0"}), "CPUExecutionProvider"])

    # warmup
    _ = sess.run(onames, {inames[0]: data[:8]})

    t0 = time.perf_counter()
    for start in range(0, len(data), 2048):
        end = min(start + 2048, len(data))
        out = sess.run(onames, {inames[0]: data[start:end]})
    t1 = time.perf_counter()

    # precision
    out_data = np.concatenate([sess.run(onames, {inames[0]: data[start:end]})[0]
                               for start in range(0, len(data), 2048)], axis=0)
    prec_ok = np.allclose(out_data, golden, rtol=1e-3, atol=1e-3)
    max_d = np.max(np.abs(out_data - golden))

    print(f"{level_name:10s}: {t1-t0:.2f}s  precision={'OK' if prec_ok else 'FAIL'}({max_d:.2e})")
