"""GEMM analysis: current vs theoretical peak, tiling opportunities."""
import torch, time, sys

device = torch.device("cuda")
B, S, D = 16, 32, 4096
N_WARM, N_ITER = 5, 50

# H200 peak: 989 TFLOPS FP16 Tensor Core, ~500 TFLOPS FP32 (with TF32)
# HBM bandwidth: 4.8 TB/s

print("=== BigFormer MatMul Profile (B=16,S=32,D=4096) ===")
torch.cuda.synchronize()

# ── QKV: [B,S,D] @ [D,3D] = [512,4096] @ [4096,12288] ──
M, K, N_qkv = B*S, D, D*3
flops_qkv = 2 * M * K * N_qkv  # 2*512*4096*12288 = 51.5 GFLOPs
bytes_qkv = (M*K + K*N_qkv + M*N_qkv) * 4  # input + weight + output
print(f"\n  QKV: M={M} K={K} N={N_qkv}")
print(f"    Compute: {flops_qkv/1e9:.1f} GFLOPs")
print(f"    Data: {bytes_qkv/1e6:.1f} MB")
print(f"    Arithmetic intensity: {flops_qkv/bytes_qkv:.1f} FLOPs/byte")

# Benchmark
x = torch.randn(M, K, device=device, dtype=torch.float32)
w = torch.randn(K, N_qkv, device=device, dtype=torch.float32)
for _ in range(N_WARM): _ = x @ w
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_ITER): _ = x @ w
torch.cuda.synchronize()
t_qkv = (time.perf_counter() - t0) / N_ITER * 1000
tflops_qkv = flops_qkv / (t_qkv/1000) / 1e12
bw_qkv = bytes_qkv / (t_qkv/1000) / 1e12
print(f"    Time: {t_qkv:.2f}ms  ({tflops_qkv:.1f} TFLOPS, {bw_qkv:.1f} TB/s)")

# ── FF1: [M,D] @ [D,4D] ──
N_ff1 = D*4
flops_ff1 = 2 * M * K * N_ff1
bytes_ff1 = (M*K + K*N_ff1 + M*N_ff1) * 4
print(f"\n  FF1: M={M} K={K} N={N_ff1}")
print(f"    Compute: {flops_ff1/1e9:.1f} GFLOPs")
print(f"    Arithmetic intensity: {flops_ff1/bytes_ff1:.1f} FLOPs/byte")

w2 = torch.randn(K, N_ff1, device=device, dtype=torch.float32)
for _ in range(N_WARM): _ = x @ w2
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_ITER): _ = x @ w2
torch.cuda.synchronize()
t_ff1 = (time.perf_counter() - t0) / N_ITER * 1000
tflops_ff1 = flops_ff1 / (t_ff1/1000) / 1e12
print(f"    Time: {t_ff1:.2f}ms  ({tflops_ff1:.1f} TFLOPS)")

# ── Fused FF1+GELU+FF2 (tiling opportunity) ──
print(f"\n── Fused FF1+GELU+FF2 (ideal tiling target) ──")
# Currently: 3 separate kernel launches (FF1, GELU, FF2)
# With tiling: single kernel, intermediate stays in registers/SMEM
w3 = torch.randn(N_ff1, D, device=device, dtype=torch.float32)
for _ in range(N_WARM): 
    _ = torch.nn.functional.gelu(x @ w2) @ w3
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_ITER):
    _ = torch.nn.functional.gelu(x @ w2) @ w3
torch.cuda.synchronize()
t_fused = (time.perf_counter() - t0) / N_ITER * 1000
print(f"  Separate launches: {t_ff1 + (t_ff1*0.25):.2f}ms (est FF1+GELU+FF2)")
print(f"  Actual time: {t_fused:.2f}ms")
print(f"  GELU+FF2 overhead: {t_fused - t_ff1:.2f}ms")

# ── cuBLASLt check ──
print(f"\n── cuBLAS version ──")
print(f"  cuBLAS available: {torch.backends.cuda.is_built()}")
print(f"  cuBLASLt: check torch.backends.cuda.matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32}")

# ── Tile size analysis ──
print(f"\n── Tiling analysis ──")
print(f"  H200 SMEM/block: 228KB (configured)")
print(f"  QKV weight tile: 4096 x TILE_K @ 4 bytes")
print(f"  For 128KB SMEM: TILE_K ~= 128K/(4096*4) = 8 rows")
print(f"  Optimal TILE_K for L2 cache (50MB): {50*1024*1024/(4096*4):.0f} rows")
print(f"  FF1 weight tile @ 128KB: K_dim=4096, N_dim=128K/(4096*4) = 8 cols")
print(f"  For 32x32 tile: SMEM = 32*4096*4 + 32*4096*4 = 1MB (too large)")

# ── Potential improvement: batched stride ──
print(f"\n── Batched matmul check ──")
# Current: [512,4096] @ [4096,N] — treats 512 rows independently
# Batched: [16,32,4096] @ [4096,N] — PyTorch uses same cuBLAS path
xb = torch.randn(B, S, D, device=device, dtype=torch.float32)
for _ in range(N_WARM): _ = xb @ w2
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_ITER): _ = xb @ w2
torch.cuda.synchronize()
t_batch = (time.perf_counter() - t0) / N_ITER * 1000
print(f"  2D input [512,4096]: {t_ff1:.2f}ms")
print(f"  3D input [16,32,4096]: {t_batch:.2f}ms")

# ── FP16 tensor core utilization ──
print(f"\n── FP16 Tensor Core potential ──")
xh = torch.randn(M, K, device=device, dtype=torch.float16)
wh = torch.randn(K, N_ff1, device=device, dtype=torch.float16)
for _ in range(N_WARM): _ = xh @ wh
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_ITER): _ = xh @ wh
torch.cuda.synchronize()
t_fp16 = (time.perf_counter() - t0) / N_ITER * 1000
flops_fp16 = flops_ff1  # same FLOPs
tflops_fp16 = flops_fp16 / (t_fp16/1000) / 1e12
print(f"  FP32: {t_ff1:.2f}ms  ({tflops_ff1:.1f} TFLOPS)")
print(f"  FP16: {t_fp16:.2f}ms  ({tflops_fp16:.1f} TFLOPS)")
print(f"  Speedup: {t_ff1/t_fp16:.1f}x")
print(f"  H200 peak FP16: 989 TFLOPS → utilization: {tflops_fp16/989*100:.1f}%")
