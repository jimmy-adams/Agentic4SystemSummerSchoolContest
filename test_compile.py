"""Test torch.compile on BigFormer block computation."""
import torch
import torch.nn.functional as F
import time

device = torch.device("cuda")
B, S, D = 16, 32, 4096
N_WARMUP, N_ITER = 5, 50

# Enable TF32 for matmul (Ampere+ tensor cores, ~2x throughput)
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True

print("=== TF32 + torch.compile ===")

# Define a single transformer block
def transformer_block(x, w_qkv, w_proj, w_ff1, w_ff2, 
                      ln1_w, ln1_b, ln2_w, ln2_b,
                      qkv_b, proj_b, ff1_b, ff2_b):
    # LN1
    residual = x
    x = F.layer_norm(x.float(), [D], weight=ln1_w, bias=ln1_b, eps=1e-5).to(x.dtype)
    
    # QKV
    qkv = x @ w_qkv
    if qkv_b is not None: qkv = qkv + qkv_b
    
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(B, S, 32, 128).permute(0, 2, 1, 3)
    k = k.view(B, S, 32, 128).permute(0, 2, 1, 3)
    v = v.view(B, S, 32, 128).permute(0, 2, 1, 3)
    
    # Attention
    attn = (q @ k.transpose(-2, -1)) * (128 ** -0.5)
    attn = F.softmax(attn.float(), dim=-1).to(x.dtype)
    attn_out = attn @ v
    attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, 4096).to(x.dtype)
    
    # Proj
    attn_out = attn_out @ w_proj
    if proj_b is not None: attn_out = attn_out + proj_b
    x = residual + attn_out
    
    # LN2 + FFN
    residual = x
    x = F.layer_norm(x.float(), [D], weight=ln2_w, bias=ln2_b, eps=1e-5).to(x.dtype)
    
    x = x @ w_ff1
    if ff1_b is not None: x = x + ff1_b
    x = F.gelu(x)
    x = x @ w_ff2
    if ff2_b is not None: x = x + ff2_b
    x = residual + x
    
    return x

# Create random inputs matching real sizes
x = torch.randn(B, S, D, device=device, dtype=torch.float32)
w_qkv = torch.randn(D, D*3, device=device, dtype=torch.float32)
w_proj = torch.randn(D, D, device=device, dtype=torch.float32)
w_ff1 = torch.randn(D, D*4, device=device, dtype=torch.float32)
w_ff2 = torch.randn(D*4, D, device=device, dtype=torch.float32)
ln1_w = torch.randn(D, device=device, dtype=torch.float32)
ln1_b = torch.randn(D, device=device, dtype=torch.float32)
ln2_w = torch.randn(D, device=device, dtype=torch.float32)
ln2_b = torch.randn(D, device=device, dtype=torch.float32)
qkv_b = torch.randn(D*3, device=device, dtype=torch.float32)
proj_b = torch.randn(D, device=device, dtype=torch.float32)
ff1_b = torch.randn(D*4, device=device, dtype=torch.float32)
ff2_b = torch.randn(D, device=device, dtype=torch.float32)

# Warmup
for _ in range(N_WARMUP):
    _ = transformer_block(x, w_qkv, w_proj, w_ff1, w_ff2, ln1_w, ln1_b, ln2_w, ln2_b, qkv_b, proj_b, ff1_b, ff2_b)
torch.cuda.synchronize()

# Eager
t0 = time.perf_counter()
for _ in range(N_ITER):
    _ = transformer_block(x, w_qkv, w_proj, w_ff1, w_ff2, ln1_w, ln1_b, ln2_w, ln2_b, qkv_b, proj_b, ff1_b, ff2_b)
torch.cuda.synchronize()
t_eager = (time.perf_counter() - t0) / N_ITER * 1000

# Compile
print("  Compiling (first call takes time)...", )
compiled_block = torch.compile(transformer_block, mode="reduce-overhead")

# Warmup compile
_ = compiled_block(x, w_qkv, w_proj, w_ff1, w_ff2, ln1_w, ln1_b, ln2_w, ln2_b, qkv_b, proj_b, ff1_b, ff2_b)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(N_ITER):
    _ = compiled_block(x, w_qkv, w_proj, w_ff1, w_ff2, ln1_w, ln1_b, ln2_w, ln2_b, qkv_b, proj_b, ff1_b, ff2_b)
torch.cuda.synchronize()
t_compile = (time.perf_counter() - t0) / N_ITER * 1000

print(f"\n  Eager:    {t_eager:.2f}ms/block", )
print(f"  Compile:  {t_compile:.2f}ms/block", )
print(f"  Speedup:  {t_eager/t_compile:.1f}x", )
print(f"  Per 24 blocks: eager={t_eager*24/1000:.1f}s  compile={t_compile*24/1000:.1f}s", )
print(f"  Total savings (32 batches): {(t_eager-t_compile)*24*32/1000:.0f}s", )
