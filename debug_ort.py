#!/usr/bin/env python3
"""Debug ONNX Runtime CUDA EP."""
import os
os.environ["LD_LIBRARY_PATH"] = (
    "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:"
    "/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:"
    "/usr/local/cuda/targets/x86_64-linux/lib:"
    + os.environ.get("LD_LIBRARY_PATH", "")
)
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

import onnxruntime as ort
print("ORT:", ort.__version__)
print("EPs:", ort.get_available_providers())

try:
    s = ort.InferenceSession(
        "/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models/mlp_v1.onnx",
        providers=[("CUDAExecutionProvider", {"device_id": "0"}), "CPUExecutionProvider"]
    )
    print("Active:", s.get_providers())
except Exception as e:
    print("Error:", e)
    print("Falling back to CPU...")
