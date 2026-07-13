#!/usr/bin/env python3
import sys
import numpy as np

output_dir = sys.argv[1]
golden_dir = sys.argv[2]
labels_path = sys.argv[3] if len(sys.argv) > 3 else None
acc_threshold = float(sys.argv[4]) if len(sys.argv) > 4 else None

out = np.load(f"{output_dir}/logits.npy")
gold = np.load(f"{golden_dir}/logits.npy")

print(f"Output shape: {out.shape}")
print(f"Golden shape: {gold.shape}")
print(f"Dtype: out={out.dtype}, gold={gold.dtype}")

prec_pass = np.allclose(out, gold, rtol=1e-3, atol=1e-3)
max_diff = np.max(np.abs(out - gold))
mean_diff = np.mean(np.abs(out - gold))
print(f"Precision pass (1e-3): {prec_pass}")
print(f"Max abs diff: {max_diff}")
print(f"Mean abs diff: {mean_diff}")

if labels_path is not None:
    lab = np.load(labels_path)
    acc = (out.argmax(1) == lab).mean()
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    if acc_threshold is not None:
        print(f"Accuracy threshold ({acc_threshold}): {'PASS' if acc >= acc_threshold else 'FAIL'}")
