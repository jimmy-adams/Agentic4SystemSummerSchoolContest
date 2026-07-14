#!/bin/bash
# Wrapper for infer_worker.py — sets CUDA library paths for subprocess.
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
export NVIDIA_TF32_OVERRIDE=0
exec python3 "$(dirname "$0")/infer_worker.py"
