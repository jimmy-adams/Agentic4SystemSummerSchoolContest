#!/usr/bin/env python3
"""ORT tuning for ResNet (96% of C3.5 time)."""
import time, numpy as np, json, os, itertools
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
import onnxruntime as ort

MODEL = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models/resnet_v1.onnx'
INPUT = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35/resnet_v1/input'

with open(f'{INPUT}/manifest.json') as f: m = json.load(f)
data = {e['name']: np.load(f'{INPUT}/{e["file"]}') for e in m['tensors']}
first = list(data.keys())[0]
N = data[first].shape[0]
B = 256

sess = ort.InferenceSession(MODEL, providers=[
    ('CUDAExecutionProvider', {'device_id': '0'}), 'CPUExecutionProvider'])
inames = [i.name for i in sess.get_inputs()]
onames = [o.name for o in sess.get_outputs()]

# Baseline
feed = {n: data[n][:B] for n in inames if n in data}
_ = sess.run(onames, feed)  # warmup
t0 = time.perf_counter()
for s in range(0, N, B):
    e = min(s+B, N)
    _ = sess.run(onames, {n: data[n][s:e] for n in inames if n in data})
t1 = time.perf_counter()
baseline = t1 - t0
print(f"baseline (EXTENDED): {baseline:.2f}s")

# Test variations
configs = [
    ("cuda_graph", "1"),
    ("cudnn_conv_algo_search", "EXHAUSTIVE"),
    ("do_copy_in_default_stream", "1"),
]
for k, v in configs:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    try:
        s = ort.InferenceSession(MODEL, opts, providers=[
            ('CUDAExecutionProvider', {'device_id': '0', k: v}), 'CPUExecutionProvider'])
        _ = s.run(onames, feed)  # warmup
        t0 = time.perf_counter()
        for s2 in range(0, N, B):
            e2 = min(s2+B, N)
            _ = s.run(onames, {n: data[n][s2:e2] for n in inames if n in data})
        t1 = time.perf_counter()
        et = t1 - t0
        chg = (1-et/baseline)*100
        print(f"  {k}={v}: {et:.2f}s ({chg:+.0f}%)")
    except Exception as e:
        print(f"  {k}={v}: ERROR {e}")

# Thread count
for threads in [4, 16, 32]:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    s = ort.InferenceSession(MODEL, opts, providers=[
        ('CUDAExecutionProvider', {'device_id': '0'}), 'CPUExecutionProvider'])
    _ = s.run(onames, feed)
    t0 = time.perf_counter()
    for s2 in range(0, N, B):
        e2 = min(s2+B, N)
        _ = s.run(onames, {n: data[n][s2:e2] for n in inames if n in data})
    t1 = time.perf_counter()
    et = t1 - t0
    print(f"  threads={threads}: {et:.2f}s ({(1-et/baseline)*100:+.0f}%)")
