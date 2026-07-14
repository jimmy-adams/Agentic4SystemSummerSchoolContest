#!/usr/bin/env python3
"""Trade ResNet accuracy headroom for speed."""
import time, numpy as np, json, os
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
import onnxruntime as ort

MP = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models/resnet_v1.onnx'
IDIR = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35/resnet_v1/input'
GP = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35/resnet_v1/golden/logits.npy'
LP = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35/resnet_v1/labels.npy'

with open(f'{IDIR}/manifest.json') as f: m = json.load(f)
data = {e['name']: np.load(f'{IDIR}/{e["file"]}') for e in m['tensors']}
first = list(data.keys())[0]
N, B = data[first].shape[0], 256
gold = np.load(GP); labels = np.load(LP)

for level_name, level in [
    ("EXTENDED", ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED),
    ("ALL", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
]:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = level
    try:
        sess = ort.InferenceSession(MP, opts, providers=[
            ('CUDAExecutionProvider', {'device_id': '0'}), 'CPUExecutionProvider'])
        inames = [i.name for i in sess.get_inputs()]
        onames = [o.name for o in sess.get_outputs()]

        feed = {n: data[n][:B] for n in inames if n in data}
        _ = sess.run(onames, feed)  # warmup

        t0 = time.perf_counter()
        all_out = []
        for s in range(0, N, B):
            e = min(s+B, N)
            all_out.append(sess.run(onames, {n: data[n][s:e] for n in inames if n in data})[0])
        t1 = time.perf_counter()

        out = np.concatenate(all_out, axis=0)
        prec_ok = np.allclose(out, gold, rtol=1e-3, atol=1e-3)
        md = np.max(np.abs(out - gold))
        acc = (out.argmax(1) == labels).mean()

        print(f"{level_name:10s}: {t1-t0:.2f}s  "
              f"precision={'OK' if prec_ok else f'FAIL({md:.1e})'}  "
              f"acc={acc:.4f}  gate={'PASS' if acc>=0.85 else 'FAIL'}")
    except Exception as e:
        print(f"{level_name:10s}: ERROR {str(e)[:80]}")
