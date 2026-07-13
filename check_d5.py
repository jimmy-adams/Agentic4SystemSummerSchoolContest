"""C3.2 D5 targeted check: hardware capability coverage."""
import sys
sys.path.insert(0, "/home/mig20/c3_solution")

from scheduler import import_onnx_graph, strategy, hardware

MODELS = "/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models"

all_kernel_names = set()
all_precisions = set()
matmul_variants = set()
conv_variants = set()

for model_name in ["mlp_v1", "resnet_v1", "transformer_v1"]:
    g = import_onnx_graph(f"{MODELS}/{model_name}.onnx")
    for n in g.nodes:
        p = strategy.select_precision(n, g)
        all_precisions.add(p.precision)
        kseq = strategy.decompose(n, g, p)
        for k in kseq:
            all_kernel_names.add(k.name)
            if k.name.startswith("matmul_"):
                matmul_variants.add(k.name)
            if "conv" in k.name or "winograd" in k.name:
                conv_variants.add(k.name)

print("=== D5.1 精度种类 ===")
print(f"  Used: {sorted(all_precisions)} ({len(all_precisions)} types)")
print(f"  Score: {'1.0 ✓' if len(all_precisions) >= 3 else f'0.5 ({len(all_precisions)} types)'}")

print("\n=== D5.2 GEMM kernel 多样度 ===")
print(f"  matmul variants: {sorted(matmul_variants)}")
has_f32 = any("f32" in m for m in matmul_variants)
has_f16 = any("f16" in m for m in matmul_variants)
has_f8  = any("f8"  in m for m in matmul_variants)
has_f4  = any("f4"  in m for m in matmul_variants)
score = (0.5 if (has_f32 and has_f16) else 0) + (0.25 if has_f8 else 0) + (0.25 if has_f4 else 0)
print(f"  f32={has_f32} f16={has_f16} f8={has_f8} f4={has_f4}")
print(f"  Score: {score}/1.0")

print("\n=== D5.3 Conv2d 策略选择 ===")
print(f"  conv variants: {sorted(conv_variants)}")
has_im2col = any("im2col" in c for c in conv_variants)
has_winograd = any("winograd" in c for c in conv_variants)
print(f"  im2col={has_im2col} winograd={has_winograd}")
print(f"  Score: {'1.0 ✓' if (has_im2col and has_winograd) else f'0.0 ✗ (missing one)'}")

print(f"\n=== D5 小计: {(1.0 if len(all_precisions)>=3 else 0.5) + score + (1.0 if (has_im2col and has_winograd) else 0)}/3.0 ===")
