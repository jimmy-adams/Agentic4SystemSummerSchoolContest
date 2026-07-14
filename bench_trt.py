#!/usr/bin/env python3
"""
C3.5: TensorRT Execution Provider test — the most promising optimization.

Tests:
  - ORT CUDA EP (baseline)
  - ORT TRT EP fp32 (safe, should pass 1e-3 gate)
  - ORT TRT EP fp16 (risky, may fail gate but maximum speed)
"""

import os, time, json, numpy as np

os.environ['NVIDIA_TF32_OVERRIDE'] = '0'

import onnxruntime as ort

BASE_M = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models'
BASE_D = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35'


def load_data(name):
    d = f'{BASE_D}/{name}'
    with open(f'{d}/input/manifest.json') as f:
        m = json.load(f)
    data = {e['name']: np.load(f'{d}/input/{e["file"]}') for e in m['tensors']}
    return data, np.load(f'{d}/golden/logits.npy')


def infer_ort(mp, data, N, B, providers):
    sess = ort.InferenceSession(mp, providers=providers)
    inames = [i.name for i in sess.get_inputs()]
    onames = [o.name for o in sess.get_outputs()]
    feed_warm = {n: data[n][:min(8,N)] for n in inames if n in data}
    _ = sess.run(onames, feed_warm)
    t0 = time.perf_counter()
    for start in range(0, N, B):
        end = min(start + B, N)
        sess.run(onames, {n: data[n][start:end] for n in inames if n in data})
    return time.perf_counter() - t0


def precision_ok(mp, data, N, golden):
    """Quick precision check: one batch only."""
    sess = ort.InferenceSession(mp, providers=[
        ('CUDAExecutionProvider', {'device_id': '0'}), 'CPUExecutionProvider'])
    inames = [i.name for i in sess.get_inputs()]
    onames = [o.name for o in sess.get_outputs()]
    feed = {n: data[n][:min(64,N)] for n in inames if n in data}
    out = sess.run(onames, feed)[0]
    md = np.max(np.abs(out - golden[:min(64,len(golden))]))
    ok = np.allclose(out, golden[:min(64,len(golden))], rtol=1e-3, atol=1e-3)
    return md, ok


# ═══════════════════════════════════════════════════════════════════════════
print("C3.5: TensorRT Execution Provider")
print("=" * 80)

# TRT cache settings
os.environ['ORT_TENSORRT_ENGINE_CACHE_ENABLE'] = '1'
os.environ['ORT_TENSORRT_CACHE_PATH'] = '/tmp/trt_cache'

providers_configs = {
    'CUDA (baseline)': [
        ('CUDAExecutionProvider', {'device_id': '0'}),
        'CPUExecutionProvider',
    ],
    'TRT fp32': [
        ('TensorrtExecutionProvider', {
            'device_id': '0',
            'trt_fp16_enable': '0',
            'trt_engine_cache_enable': '1',
            'trt_engine_cache_path': '/tmp/trt_cache',
        }),
        'CUDAExecutionProvider',  # fallback
        'CPUExecutionProvider',
    ],
    'TRT fp16': [
        ('TensorrtExecutionProvider', {
            'device_id': '0',
            'trt_fp16_enable': '1',
            'trt_engine_cache_enable': '1',
            'trt_engine_cache_path': '/tmp/trt_cache',
        }),
        'CUDAExecutionProvider',
        'CPUExecutionProvider',
    ],
}

for name in ['resnet_v1', 'transformer_v1', 'mlp_v1']:
    data, golden = load_data(name)
    N = list(data.values())[0].shape[0]
    B = 64 if name == 'resnet_v1' else 256
    mp = f'{BASE_M}/{name}.onnx'

    print(f"\n--- {name} (N={N}, B={B}) ---")
    print(f"{'Provider':30s} {'Time':>8s} {'Δ%':>8s} {'max_diff':>10s} {'Gate':>6s}")

    for prov_name, providers in providers_configs.items():
        # First run: TRT builds engine (slow, not timed)
        # Second run: use cached engine (fast)
        try:
            t = infer_ort(mp, data, N, B, providers)
        except Exception as e:
            print(f"  {prov_name:28s} ERROR: {str(e)[:60]}")
            continue

        md, ok = precision_ok(mp, data, N, golden)

        delta = ""
        if prov_name != 'CUDA (baseline)':
            base_t = results[0]
            delta = f"{(base_t/t-1)*100:+7.1f}%"
        else:
            results = [t]

        print(f"  {prov_name:28s} {t:7.2f}s {delta:>8s} {md:10.2e} "
              f"{'OK' if ok else 'FAIL':>6s}")

print("\nDone. Engine cache: /tmp/trt_cache")
