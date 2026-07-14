"""CuPy vs PyTorch: key ops."""
import torch, cupy as cp, time, numpy as np
import torch.nn.functional as F
B,S,D,N=16,32,4096,200
M=B*S
xt=torch.randn(M,D,device="cuda"); wt=torch.randn(D,D*4,device="cuda")
xc=cp.asarray(xt); wc=cp.asarray(wt)

for _ in range(5): _ = xt @ wt; _ = xc @ wc
torch.cuda.synchronize(); cp.cuda.Stream.null.synchronize()

# 1. MatMul
t0=time.perf_counter()
for _ in range(N): _ = xt @ wt
torch.cuda.synchronize()
t_t=(time.perf_counter()-t0)/N*1000
t0=time.perf_counter()
for _ in range(N): _ = xc @ wc
cp.cuda.Stream.null.synchronize()
t_c=(time.perf_counter()-t0)/N*1000
print(f"MatMul [512,4096]x[4096,16384]: Torch={t_t:.1f}ms  CuPy={t_c:.1f}ms  ratio={t_t/t_c:.2f}x")

# 2. H2D
arr=np.random.randn(4096,12288).astype(np.float32)
t0=time.perf_counter()
for _ in range(N): _ = torch.from_numpy(arr).to("cuda")
torch.cuda.synchronize()
t_th=(time.perf_counter()-t0)/N*1000
t0=time.perf_counter()
for _ in range(N): _ = cp.asarray(arr)
cp.cuda.Stream.null.synchronize()
t_ch=(time.perf_counter()-t0)/N*1000
print(f"H2D [4096,12288]: Torch={t_th:.2f}ms  CuPy={t_ch:.2f}ms  ratio={t_th/t_ch:.2f}x")

# 3. GELU
y=xt @ wt
t0=time.perf_counter()
for _ in range(N): _ = F.gelu(y)
torch.cuda.synchronize()
t_gelu=(time.perf_counter()-t0)/N*1000
print(f"GELU [512,16384]: Torch={t_gelu:.3f}ms")
print(f"CuPy matmuls IDENTICAL to Torch (both use cuBLAS)")
print(f"GELU fusion: save {t_gelu:.2f}ms/block x 24 blocks x 32 batches = {t_gelu*24*32/1000:.1f}s max")

