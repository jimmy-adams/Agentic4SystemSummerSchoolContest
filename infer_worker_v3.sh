#!/bin/bash
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib"
export NVIDIA_TF32_OVERRIDE=0
exec python3 "$(dirname "$0")/infer_worker_v3.py"
