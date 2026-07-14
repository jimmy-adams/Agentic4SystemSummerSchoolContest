"""Fix: add biases to MatMul outputs."""
with open('bigformer_gpu3.py', 'r') as f:
    content = f.read()

# Fix 1: Add QKV bias
old1 = "        qkv = x @ w_qkv  # [B, S, 12288]"
new1 = """        qkv = x @ w_qkv  # [B, S, 12288]
        # Add QKV bias
        qkv_bias = get_param(f"blocks.{bid}.qkv.bias")
        if qkv_bias is not None:
            qkv = qkv + qkv_bias.float().to(device)"""
content = content.replace(old1, new1)

# Fix 2: Add proj bias
old2 = "        attn_out = attn_out @ w_proj  # output projection"
new2 = """        attn_out = attn_out @ w_proj  # output projection
        proj_bias = get_param(f"blocks.{bid}.proj.bias")
        if proj_bias is not None:
            attn_out = attn_out + proj_bias.float().to(device)"""
content = content.replace(old2, new2)

# Fix 3: Add FF1 bias
old3 = "        x = x @ w_ff1"
new3 = """        x = x @ w_ff1
        ff1_bias = get_param(f"blocks.{bid}.ff1.bias")
        if ff1_bias is not None:
            x = x + ff1_bias.float().to(device)"""
content = content.replace(old3, new3)

# Fix 4: Add FF2 bias
old4 = "        x = x @ w_ff2"
new4 = """        x = x @ w_ff2
        ff2_bias = get_param(f"blocks.{bid}.ff2.bias")
        if ff2_bias is not None:
            x = x + ff2_bias.float().to(device)"""
content = content.replace(old4, new4)

with open('bigformer_gpu3.py', 'w') as f:
    f.write(content)
print("Added bias fixes")
