#!/usr/bin/env python3
"""Explore ORT memory options for C3.5."""
import os; os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
import onnxruntime as ort

# Check available CUDA EP options by trying each
options = [
    "arena_extend_strategy",  # 0=next_power_of_two, 1=same_as_requested
    "gpu_mem_limit",           # bytes limit
    "memory_pattern",          # enable/disable memory pattern optimization
    "enable_cuda_graph",
    "tunable_op_enable",
]

MP = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models/resnet_v1.onnx'

# Baseline
s = ort.InferenceSession(MP, providers=[
    ('CUDAExecutionProvider', {'device_id': '0'}), 'CPUExecutionProvider'])
print(f"Active providers: {s.get_providers()}")

# Test each option
for opt in options:
    for val in ["0", "1"]:
        try:
            opts = {opt: val, 'device_id': '0'}
            s2 = ort.InferenceSession(MP, providers=[
                ('CUDAExecutionProvider', opts), 'CPUExecutionProvider'])
            print(f"  {opt}={val}: OK, active={s2.get_providers()[0]}")
        except Exception as e:
            print(f"  {opt}={val}: {str(e)[:100]}")
        break  # only test first value
