"""Profile BigFormer pipeline bottlenecks."""
import time, sys, torch

# Quick profile of key operations at scale
device = torch.device("cuda")
B, S, D = 16, 32, 4096
N_WARMUP, N_ITER = 3, 20

print("=== Profiling key ops (B=16, S=32, D=4096) ===", file=sys.stderr)

# --- Weight transfer (CPU pinned -> GPU) ---
w = torch.randn(4096, 12288, dtype=torch.float32).pin_memory()
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_ITER):
    w_gpu = w.to(device, non_blocking=True)
torch.cuda.synchronize()
t_xfer_qkv = (time.perf_counter() - t0) / N_ITER * 1000
print(f"  Weight xfer (4096x12288): {t_xfer_qkv:.2f}ms", file=sys.stderr)

# Total weight per block: qkv(4096x12288) + proj(4096x4096) + ff1(4096x16384) + ff2(16384x4096)
total_elems = 4096*12288 + 4096*4096 + 4096*16384 + 16384*4096
print(f"  Total weight per block: {total_elems*4/1e6:.1f} MB", file=sys.stderr)
# ~804 MB per block as calculated before. At ~50GB/s PCIe: ~16ms per block

# --- QKV MatMul ---
x = torch.randn(B, S, D, device=device)
w_qkv = torch.randn(D, D*3, device=device)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_ITER):
    _ = x @ w_qkv
torch.cuda.synchronize()
t_mm = (time.perf_counter() - t0) / N_ITER * 1000
print(f"  QKV MatMul [{B},{S},{D}] @ [{D},{D*3}]: {t_mm:.2f}ms", file=sys.stderr)

# --- Multi-head attention ---
qkv = x @ w_qkv
q, k, v = qkv.chunk(3, dim=-1)
q = q.view(B, S, 32, 128).permute(0, 2, 1, 3)
k = k.view(B, S, 32, 128).permute(0, 2, 1, 3)
v = v.view(B, S, 32, 128).permute(0, 2, 1, 3)

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_ITER):
    attn = (q @ k.transpose(-2, -1)) * (128 ** -0.5)
    attn = torch.nn.functional.softmax(attn.float(), dim=-1).to(torch.float32)
    attn_out = attn @ v
    attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, 4096)
torch.cuda.synchronize()
t_attn = (time.perf_counter() - t0) / N_ITER * 1000
print(f"  Multi-head attention: {t_attn:.2f}ms", file=sys.stderr)

# --- Flash Attention (F.scaled_dot_product_attention) ---
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_ITER):
    _ = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=128**-0.5)
torch.cuda.synchronize()
t_flash = (time.perf_counter() - t0) / N_ITER * 1000
print(f"  Flash Attention: {t_flash:.2f}ms  (speedup: {t_attn/t_flash:.1f}x)", file=sys.stderr)

# --- FFN MatMuls ---
w_ff1 = torch.randn(D, D*4, device=device)
w_ff2 = torch.randn(D*4, D, device=device)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_ITER):
    _ = x @ w_ff1
    _ = torch.nn.functional.gelu(_)
    _ = _ @ w_ff2
torch.cuda.synchronize()
t_ffn = (time.perf_counter() - t0) / N_ITER * 1000
print(f"  FFN (MatMul+GELU+MatMul): {t_ffn:.2f}ms", file=sys.stderr)

# --- Total per-block estimate ---
print(f"\n=== Per-block estimate (24 blocks) ===", file=sys.stderr)
per_block = t_mm + t_attn + t_ffn + 16  # ~16ms weight xfer (pipelined away after first)
print(f"  Compute: {per_block:.1f}ms/block × 24 = {per_block*24/1000:.1f}s", file=sys.stderr)
print(f"  With pipeline: weight transfer overlaps with compute", file=sys.stderr)
print(f"  First block penalty: {16:.0f}ms (no overlap)", file=sys.stderr)

# --- FP16 test ---
print(f"\n=== FP16 potential ===", file=sys.stderr)
w_fp16 = torch.randn(4096, 12288, dtype=torch.float16, device=device)
x_fp16 = torch.randn(B, S, D, dtype=torch.float16, device=device)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_ITER):
    _ = x_fp16 @ w_fp16
torch.cuda.synchronize()
t_mm_fp16 = (time.perf_counter() - t0) / N_ITER * 1000
print(f"  FP16 MatMul: {t_mm_fp16:.2f}ms vs FP32: {t_mm:.2f}ms ({t_mm/t_mm_fp16:.1f}x)", file=sys.stderr)
print(f"  FP16 weight size: {total_elems*2/1e6:.1f} MB/block (half bandwidth)", file=sys.stderr)
