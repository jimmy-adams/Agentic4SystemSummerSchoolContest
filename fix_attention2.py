"""Fix multi-head attention - v2 with correct pattern matching."""
with open('bigformer_gpu3.py', 'r') as f:
    content = f.read()

# The exact pattern from the remote file
old = """        qkv = x @ w_qkv  # [B, S, 12288]
        q, k, v = qkv.chunk(3, dim=-1)
        
        scale = D ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        
        # Causal mask: prevent attending to future tokens
        causal_mask = torch.triu(torch.ones(S, S, device=device), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask, float('-inf'))
        
        attn = F.softmax(attn.float(), dim=-1).to(x.dtype)
        attn_out = attn @ v
        attn_out = attn_out @ w_proj
        x = residual + attn_out"""

new = """        qkv = x @ w_qkv  # [B, S, 12288]
        q, k, v = qkv.chunk(3, dim=-1)  # each [B,S,4096]
        
        # Multi-head: 32 heads, head_dim=128
        # [B,S,4096] -> [B,S,32,128] -> [B,32,S,128]
        q = q.view(B, S, 32, 128).permute(0, 2, 1, 3)
        k = k.view(B, S, 32, 128).permute(0, 2, 1, 3)
        v = v.view(B, S, 32, 128).permute(0, 2, 1, 3)
        
        # Scaled dot-product: scale = 1/sqrt(head_dim=128)
        attn = (q @ k.transpose(-2, -1)) * (128 ** -0.5)
        attn = F.softmax(attn.float(), dim=-1).to(x.dtype)
        attn_out = attn @ v  # [B,32,S,128]
        
        # Merge heads: [B,32,S,128] -> [B,S,4096]
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, 4096)
        attn_out = attn_out @ w_proj
        x = residual + attn_out"""

if old in content:
    content = content.replace(old, new)
    with open('bigformer_gpu3.py', 'w') as f:
        f.write(content)
    print("FIXED!")
else:
    print("NOT FOUND. Lines around 'chunk':")
    for i, line in enumerate(content.split('\n')):
        if 'chunk' in line.lower():
            for j in range(max(0,i-2), min(len(content.split('\n')), i+15)):
                print(f"  {j}: {content.split(chr(10))[j]}")
            break
