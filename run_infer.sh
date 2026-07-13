#!/bin/bash
# c3_solution/run_infer.sh — wrapper that sets CUDA library paths for ORT GPU
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
export NVIDIA_TF32_OVERRIDE=0
exec python3 /home/mig20/c3_solution/infer.py "$@"
