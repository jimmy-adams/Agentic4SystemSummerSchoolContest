# C3.3 score calculation — two interpretations
mlp_raw, mlp_opt = 6, 4
rn_raw,  rn_opt  = 48, 14

mlp_f2 = min((mlp_raw-mlp_opt)/mlp_raw*5, 3)
rn_f2  = min((rn_raw-rn_opt)/rn_raw*5, 3)
per_model_avg = (mlp_f2 + rn_f2)/2

combined_raw = mlp_raw + rn_raw
combined_opt = mlp_opt + rn_opt
combined_f2 = min((combined_raw-combined_opt)/combined_raw*5, 3)

print('=== F2 Kernel Launch Reduction ===')
print(f'MLP:     {mlp_raw}->{mlp_opt}  ({mlp_raw-mlp_opt}/{mlp_raw} = {(mlp_raw-mlp_opt)/mlp_raw:.0%})')
print(f'ResNet:  {rn_raw}->{rn_opt}  ({rn_raw-rn_opt}/{rn_raw} = {(rn_raw-rn_opt)/rn_raw:.0%})')
print()
print(f'方法A (分别算再平均): ({mlp_f2:.2f}+{rn_f2:.2f})/2 = {per_model_avg:.2f}')
print(f'方法B (合并去掉再算): min(({combined_raw}-{combined_opt})/{combined_raw}*5, 3) = {combined_f2:.2f}')
print()
print(f'F1 (命名pattern): 2/5  ← 瓶颈(MLP/ResNet无Dropout/LN/MatMulBias)')
print(f'F2 (启动减少):    {max(per_model_avg, combined_f2):.2f}/3')
print(f'F3 (缓冲减少):    同上')
print(f'F4 (融合正确性):  4/4')
print()
total_a = 2 + per_model_avg*2 + 4
total_b = 2 + combined_f2*2 + 4
print(f'C3.3 方案A: {total_a:.1f}/15')
print(f'C3.3 方案B: {total_b:.1f}/15')
