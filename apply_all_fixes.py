#!/usr/bin/env python3
"""Apply all C3 bottleneck fixes."""
import os, sys

FILES = {}

# ═══════════════════════════════════════════════════════════════════════
# Fix 1: scheduler/executor.py — Remove duplicate FusedConvRelu body
# ═══════════════════════════════════════════════════════════════════════
print("=== Fix 1: FusedConvRelu (remove duplicate conv2d) ===")

for fpath in ["scheduler/executor.py", "executor.py"]:
    with open(fpath) as f:
        c = f.read()
    
    # Remove the duplicate conv2d block (lines 325-336 that overwrite the fused result)
    old = """            self._set(out[0], F.relu(F.conv2d(x, w, b, stride=stride, padding=p)))
            x, w = g(inp[0]), g(inp[1])
            b = g(inp[2]) if len(inp) >= 3 else None
            attrs = self._node_attrs.get(out[0], {})
            stride = tuple(attrs.get("strides", [1, 1]))
            pads = attrs.get("pads", [0, 0, 0, 0])
            if len(pads) == 4:
                p = (pads[0], pads[2])
            elif len(pads) == 2:
                p = (pads[0], pads[0])
            else:
                p = (pads[0], pads[1])
            self._set(out[0], F.conv2d(x, w, b, stride=stride, padding=p))"""
    
    new = """            self._set(out[0], F.relu(F.conv2d(x, w, b, stride=stride, padding=p)))"""
    
    if old in c:
        c = c.replace(old, new)
        with open(fpath, 'w') as f:
            f.write(c)
        print(f"  Fixed: {fpath}")
    else:
        print(f"  SKIP: {fpath} (pattern not found)")

# ═══════════════════════════════════════════════════════════════════════
# Fix 2: scheduler/hardware.py — H200 specs
# ═══════════════════════════════════════════════════════════════════════
print("\n=== Fix 2: hardware.py → H200 ===")
fpath = "scheduler/hardware.py"
with open(fpath) as f:
    c = f.read()

# Update all A100 values to H200
replacements = {
    "UNIFIED_L1_SMEM_KB = 192": "UNIFIED_L1_SMEM_KB = 256  # H200 (was A100: 192)",
    "MAX_SHARED_MEMORY_KB = 164": "MAX_SHARED_MEMORY_KB = 228  # H200 with opt-in (was A100: 164)",
    "MAX_SHARED_MEMORY_PER_BLOCK_KB = 163": "MAX_SHARED_MEMORY_PER_BLOCK_KB = 228  # H200 (was A100: 163)",
    "MAX_SHARED_MEMORY_PER_BLOCK_BYTES = 163 * 1024": "MAX_SHARED_MEMORY_PER_BLOCK_BYTES = 228 * 1024  # H200",
    "L2_CACHE_MB = 40": "L2_CACHE_MB = 50  # H200 (was A100: 40)",
    "DEVICE_MEMORY_GB = 80": "DEVICE_MEMORY_GB = 141  # H200 (was A100: 80)",
    "PEAK_HBM_BANDWIDTH_GB_S = 2039": "PEAK_HBM_BANDWIDTH_GB_S = 4800  # H200 ~4.8TB/s (was A100: 2039)",
    "GPU_INTERCONNECT_BANDWIDTH_GB_S = 600": "GPU_INTERCONNECT_BANDWIDTH_GB_S = 900  # H200 NVLink4 (was A100: 600)",
}

for old, new in replacements.items():
    if old in c:
        c = c.replace(old, new)
        print(f"  {old.split('=')[0].strip()} → updated")
    else:
        print(f"  WARN: {old.split('=')[0].strip()} not found")

with open(fpath, 'w') as f:
    f.write(c)

# ═══════════════════════════════════════════════════════════════════════
# Fix 3: scheduler/memory.py — bisect.insort instead of sort()
# ═══════════════════════════════════════════════════════════════════════
print("\n=== Fix 3: memory.py sort() → bisect ===")
fpath = "scheduler/memory.py"
with open(fpath) as f:
    c = f.read()

# Add import
c = c.replace("from dataclasses import dataclass, field", 
              "from dataclasses import dataclass, field\nimport bisect")

# Replace sort() with bisect.insort
old_sort = "        self.free_list.append((offset, size))\n        self.free_list.sort()  # keep sorted for coalesce"
new_sort = "        bisect.insort(self.free_list, (offset, size))  # keep sorted for coalesce"
c = c.replace(old_sort, new_sort)

with open(fpath, 'w') as f:
    f.write(c)
print(f"  Replaced sort() with bisect.insort()")

# ═══════════════════════════════════════════════════════════════════════
# Fix 4: scheduler/graph_passes/fusion.py — Remove duplicate matchers
# ═══════════════════════════════════════════════════════════════════════
print("\n=== Fix 4: fusion.py duplicate matchers ===")
fpath = "scheduler/graph_passes/fusion.py"
with open(fpath) as f:
    c = f.read()

# Remove the SECOND set of duplicate matchers (lines ~419-438)
dup_block = """        # --- FusedSoftmaxDropout (priority 5) ---
        #  4. FusedConv2dBatchNorm (pre-fusion, runs once)
        #  5. FusedEWChain       (everything else: 2–8 EW ops)

        # --- FusedResidualNorm (priority 1) ---
        frn = _match_residual_norm(nodes, name_to_idx, consumer_map, producer_map)
        new_frn = [f for f in frn if not set(f["nodes_removed"]) & consumed_in_pass]
        if new_frn:
            changed = True
            all_fusions.extend(new_frn)
            for f in new_frn:
                consumed_in_pass.update(f["nodes_removed"])

        # --- FusedMatMulBias (priority 2) ---
        fmb = _match_matmul_bias(nodes, name_to_idx, edges, consumer_map)
        new_fmb = [f for f in fmb if not set(f["nodes_removed"]) & consumed_in_pass]
        if new_fmb:
            changed = True
            all_fusions.extend(new_fmb)
            for f in new_fmb:
                consumed_in_pass.update(f["nodes_removed"])

        # --- FusedSoftmaxDropout (priority 3) ---"""

# Replace with just SoftmaxDropout
clean_block = """        # --- FusedConvRelu (priority 5) ---
        fcr = _match_conv_relu(nodes, name_to_idx, consumer_map)
        new_fcr = [f for f in fcr if not set(f["nodes_removed"]) & consumed_in_pass]
        if new_fcr:
            changed = True
            all_fusions.extend(new_fcr)
            for f in new_fcr:
                consumed_in_pass.update(f["nodes_removed"])

        # --- FusedSoftmaxDropout (priority 6) ---"""

if dup_block in c:
    c = c.replace(dup_block, clean_block)
    print(f"  Removed duplicate matchers, added ConvRelu")
else:
    print(f"  WARN: duplicate block not found")

with open(fpath, 'w') as f:
    f.write(c)

# ═══════════════════════════════════════════════════════════════════════
# Fix 5: scheduler/tuning.py — Dynamic grid_x
# ═══════════════════════════════════════════════════════════════════════
print("\n=== Fix 5: tuning.py dynamic grid_x ===")
fpath = "scheduler/tuning.py"
with open(fpath) as f:
    c = f.read()

# Update tune_kernel to use problem_size
old_tune = """    # Fallback with generic sizing
    return KernelTuningParams(
        block_x=256,
        grid_x=4096,
        smem_bytes=-1,  # dynamic
    )"""

new_tune = """    # Fallback with problem-size-aware grid sizing (H200: 132 SMs × 32 blocks = 4224 max)
    grid = max(1, (problem_size + 256 - 1) // 256)
    grid = min(grid, 4224)  # cap at SM * max_blocks_per_SM
    return KernelTuningParams(
        block_x=256,
        grid_x=grid,
        smem_bytes=-1,  # dynamic
    )"""

if old_tune in c:
    c = c.replace(old_tune, new_tune)
    print(f"  Dynamic grid_x based on problem_size")
else:
    print(f"  WARN: fallback pattern not found")

# Also fix the tune_kernel to not ignore problem_size for known kernels
old_tune2 = """def tune_kernel(ref: KernelSpecRef, precision: str = None,
                problem_size: int = 1024) -> KernelTuningParams:
    \"\"\"Look up or compute optimal launch parameters for a kernel.\"\"\"
    name = ref.name
    key = name if name in _TUNING_TABLE else name.rpartition("_")[0]
    if key in _TUNING_TABLE:
        return KernelTuningParams(**_TUNING_TABLE[key])"""

new_tune2 = """def tune_kernel(ref: KernelSpecRef, precision: str = None,
                problem_size: int = 1024) -> KernelTuningParams:
    \"\"\"Look up or compute optimal launch parameters for a kernel.\"\"\"
    name = ref.name
    key = name if name in _TUNING_TABLE else name.rpartition("_")[0]
    if key in _TUNING_TABLE:
        params = dict(_TUNING_TABLE[key])
        # Scale grid_x to problem_size (original values assume 1024 elements)
        if problem_size != 1024:
            scale = max(1, problem_size / 1024)
            params["grid_x"] = max(1, int(params["grid_x"] * scale))
        return KernelTuningParams(**params)"""

if old_tune2 in c:
    c = c.replace(old_tune2, new_tune2)
    print(f"  Grid scaling by problem_size for known kernels")
else:
    print(f"  WARN: tune_kernel signature not found")

with open(fpath, 'w') as f:
    f.write(c)

print("\n=== All fixes applied ===")

# ═══════════════════════════════════════════════════════════════════════
# Fix 6: infer.py — ONNX shared loading
# ═══════════════════════════════════════════════════════════════════════
print("\n=== Fix 6: infer.py ONNX shared loading ===")
fpath = "infer.py"
with open(fpath) as f:
    c = f.read()

# Cache ONNX model after first load
# Add cache to detect_model_type
old_detect = """def detect_model_type(onnx_path: str) -> str:
    \"\"\"Read ONNX graph operator types and classify model.

    Returns one of: 'mlp' | 'resnet' | 'transformer' | 'unknown'
    \"\"\"
    model = onnx.load(onnx_path)"""

new_detect = """_ONNX_MODEL_CACHE = {}

def detect_model_type(onnx_path: str) -> str:
    \"\"\"Read ONNX graph operator types and classify model.

    Returns one of: 'mlp' | 'resnet' | 'transformer' | 'unknown'
    \"\"\"
    if onnx_path in _ONNX_MODEL_CACHE:
        model = _ONNX_MODEL_CACHE[onnx_path]
    else:
        model = onnx.load(onnx_path)
        _ONNX_MODEL_CACHE[onnx_path] = model"""

if old_detect in c:
    c = c.replace(old_detect, new_detect)
    print(f"  Added ONNX model cache to detect_model_type")
else:
    print(f"  WARN: detect_model_type not found")

# Cache in get_onnx_output_names
old_get = """def get_onnx_output_names(onnx_path: str) -> list:
    model = onnx.load(onnx_path)"""

new_get = """def get_onnx_output_names(onnx_path: str) -> list:
    if onnx_path in _ONNX_MODEL_CACHE:
        model = _ONNX_MODEL_CACHE[onnx_path]
    else:
        model = onnx.load(onnx_path)
        _ONNX_MODEL_CACHE[onnx_path] = model"""

if old_get in c:
    c = c.replace(old_get, new_get)
    print(f"  Added ONNX model cache to get_onnx_output_names")
else:
    print(f"  WARN: get_onnx_output_names not found")

with open(fpath, 'w') as f:
    f.write(c)

print("\n=== DONE: All 6 fixes applied ===")
