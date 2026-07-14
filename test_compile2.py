import torch, torch.nn.functional as F, time
device=torch.device("cuda"); B,S,D,N=16,32,4096,100

@torch.compile(mode="max-autotune", dynamic=False)
def block(x, w_qkv, w_proj, w_ff1, w_ff2, ln1_w, ln1_b, ln2_w, ln2_b):
    r=x; x=F.layer_norm(x,[D],weight=ln1_w,bias=ln1_b,eps=1e-5)
    qkv=x@w_qkv; q,k,v=qkv.chunk(3,dim=-1)
    q=q.view(B,S,32,128).permute(0,2,1,3); k=k.view(B,S,32,128).permute(0,2,1,3); v=v.view(B,S,32,128).permute(0,2,1,3)
    ao=(F.softmax((q@k.transpose(-2,-1))*(128**-0.5),dim=-1)@v).permute(0,2,1,3).reshape(B,S,4096)
    ao=ao@w_proj; x=r+ao
    r=x; x=F.layer_norm(x,[D],weight=ln2_w,bias=ln2_b,eps=1e-5)
    x=x@w_ff1; x=F.gelu(x); x=x@w_ff2; x=r+x
    return x

def block_eager(x, w_qkv, w_proj, w_ff1, w_ff2, ln1_w, ln1_b, ln2_w, ln2_b):
    r=x; x=F.layer_norm(x,[D],weight=ln1_w,bias=ln1_b,eps=1e-5)
    qkv=x@w_qkv; q,k,v=qkv.chunk(3,dim=-1)
    q=q.view(B,S,32,128).permute(0,2,1,3); k=k.view(B,S,32,128).permute(0,2,1,3); v=v.view(B,S,32,128).permute(0,2,1,3)
    ao=(F.softmax((q@k.transpose(-2,-1))*(128**-0.5),dim=-1)@v).permute(0,2,1,3).reshape(B,S,4096)
    ao=ao@w_proj; x=r+ao
    r=x; x=F.layer_norm(x,[D],weight=ln2_w,bias=ln2_b,eps=1e-5)
    x=x@w_ff1; x=F.gelu(x); x=x@w_ff2; x=r+x
    return x

w_qkv=torch.randn(4096,12288,device=device);w_proj=torch.randn(4096,4096,device=device)
w_ff1=torch.randn(4096,16384,device=device);w_ff2=torch.randn(16384,4096,device=device)
ln1_w=torch.randn(4096,device=device);ln1_b=torch.randn(4096,device=device)
ln2_w=torch.randn(4096,device=device);ln2_b=torch.randn(4096,device=device)
x=torch.randn(B,S,D,device=device)

_=block(x,w_qkv,w_proj,w_ff1,w_ff2,ln1_w,ln1_b,ln2_w,ln2_b); torch.cuda.synchronize()
for _ in range(5): _=block_eager(x,w_qkv,w_proj,w_ff1,w_ff2,ln1_w,ln1_b,ln2_w,ln2_b)
torch.cuda.synchronize()

t0=time.perf_counter()
for _ in range(N): _=block_eager(x,w_qkv,w_proj,w_ff1,w_ff2,ln1_w,ln1_b,ln2_w,ln2_b)
torch.cuda.synchronize(); t_e=(time.perf_counter()-t0)/N*1000

t0=time.perf_counter()
for _ in range(N): _=block(x,w_qkv,w_proj,w_ff1,w_ff2,ln1_w,ln1_b,ln2_w,ln2_b)
torch.cuda.synchronize(); t_c=(time.perf_counter()-t0)/N*1000

print(f"Eager: {t_e:.1f}ms  Compile: {t_c:.1f}ms  ratio: {t_e/t_c:.2f}x")
