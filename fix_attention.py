"""Fix multi-head attention in bigformer_gpu3.py."""
with open('bigformer_gpu3.py', 'r') as f:
    content = f.read()

old = """        # Self-attention
        residual = x
        if ln1_w is not None:
            x = layer_norm(x, ln1_w, ln1_b)
        
        qkv = x @ w_qkv  # [B, S, 12288]
        q, k, v = qkv.chunk(3, dim=-1)
        
        scale = D ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn.float(), dim=-1).to(x.dtype)
        attn_out = attn @ v
        attn_out = attn_out @ w_proj
        x = residual + attn_out"""

new = """        # Self-attention (32-head, head_dim=128)
        residual = x
        if ln1_w is not None:
            x = layer_norm(x, ln1_w, ln1_b)
        
        # QKV projection: [B,S,4096] @ [4096,12288] -> [B,S,12288]
        qkv = x @ w_qkv
        q, k, v = qkv.chunk(3, dim=-1)  # each [B,S,4096]
        
        # Multi-head reshape: [B,S,4096] -> [B,S,32,128] -> [B,32,S,128]
        q = q.view(B, S, 32, 128).permute(0, 2, 1, 3)
        k = k.view(B, S, 32, 128).permute(0, 2, 1, 3)
        v = v.view(B, S, 32, 128).permute(0, 2, 1, 3)
        
        # Scaled dot-product: scale = 1/sqrt(head_dim=128)
        attn = (q @ k.transpose(-2, -1)) * (128 ** -0.5)
        attn = F.softmax(attn.float(), dim=-1).to(x.dtype)
        attn_out = attn @ v  # [B,32,S,128]
        
        # Merge heads: [B,32,S,128] -> [B,S,4096]
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, 4096)
        attn_out = attn_out @ w_proj  # [B,S,4096] @ [4096,4096]
        x = residual + attn_out"""

if old in content:
    content = content.replace(old, new)
    with open('bigformer_gpu3.py', 'w') as f:
        f.write(content)
    print("FIXED: multi-head attention applied")
else:
    print("PATTERN NOT FOUND. Searching...")
    for i, line in enumerate(content.split('\n')):
        if 'Self-attention' in line or 'qkv' in line.lower() or 'scale =' in line:
            print(f"  {i}: {line[:80]}")
