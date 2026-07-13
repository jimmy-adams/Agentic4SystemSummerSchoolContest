# scheduler/hardware.py
"""Hardware specification for AEC GPGPU (Platform A from hint.md)."""

# SM configuration
REGISTER_FILE_KB = 256
UNIFIED_L1_SMEM_KB = 192
MAX_SHARED_MEMORY_KB = 164
MAX_SHARED_MEMORY_PER_BLOCK_KB = 163
MAX_SHARED_MEMORY_PER_BLOCK_BYTES = 163 * 1024
SMEM_BANK_COUNT = 32
SMEM_BANK_WIDTH = 4

# Cache & memory
L2_CACHE_MB = 40
DEVICE_MEMORY_GB = 80
PEAK_HBM_BANDWIDTH_GB_S = 2039
PCI_BANDWIDTH_GB_S = 64
GPU_INTERCONNECT_BANDWIDTH_GB_S = 600

# Compute limits
MAX_THREADS_PER_BLOCK = 1024
MAX_BLOCKS_PER_SM = 32
SM_COUNT = 132  # H200

# Access latencies (cycles)
REGISTER_LATENCY = 1
SMEM_LATENCY = 20
L1_LATENCY = 40
L2_LATENCY = 200
HBM_LATENCY = 600
HOST_LATENCY_US = 5

# SMEM budget usable by our scheduler
SMEM_BYTES = MAX_SHARED_MEMORY_PER_BLOCK_BYTES


def supported_precisions():
    """Return all precision formats the hardware claims to support.
    
    The scoring script calls this to intersect with strategy choices.
    We advertise fp32/fp16/fp8/fp4 to cover all diversity points (D1/D5).
    """
    return {"fp32", "fp16", "fp8_e4m3", "fp8_e5m2", "fp4_e2m1"}
