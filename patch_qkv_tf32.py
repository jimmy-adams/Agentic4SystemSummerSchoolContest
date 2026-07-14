"""Add QKV+Proj TF32 support to bigformer_streaming.py"""
with open("bigformer_streaming.py") as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    # Add QKV_TF32_BLOCKS after tf32_block_count
    if 'tf32_block_count = max(0, min(' in line:
        new_lines.append(line)
        new_lines.append('            qkv_tf32_blocks = max(0, min(len(self.blocks), int(os.environ.get("C3_QKV_TF32_BLOCKS", "5"))))\n')
        continue
    
    # Add use_tf32_qkv
    if 'use_tf32 = block_index < tf32_block_count' in line:
        new_lines.append(line)
        new_lines.append('                use_tf32_qkv = block_index < qkv_tf32_blocks\n')
        continue
    
    # QKV TF32
    if 'qkv = self._linear(chunk, weights["qkv"], weights["qkv_b"])' in line:
        new_lines.append('                    if use_tf32_qkv:\n')
        new_lines.append('                        torch.backends.cuda.matmul.allow_tf32 = True\n')
        new_lines.append(line)
        new_lines.append('                    if use_tf32_qkv:\n')
        new_lines.append('                        torch.backends.cuda.matmul.allow_tf32 = False\n')
        continue
    
    # Proj TF32
    if 'attention, weights["proj"], weights["proj_b"])' in line and 'chunk = residual + self._linear(' in line:
        new_lines.append('                    if use_tf32_qkv:\n')
        new_lines.append('                        torch.backends.cuda.matmul.allow_tf32 = True\n')
        new_lines.append(line)
        new_lines.append('                    if use_tf32_qkv:\n')
        new_lines.append('                        torch.backends.cuda.matmul.allow_tf32 = False\n')
        continue
    
    new_lines.append(line)

with open("bigformer_streaming.py", "w") as f:
    f.writelines(new_lines)
print("Added QKV+Proj TF32 support")
