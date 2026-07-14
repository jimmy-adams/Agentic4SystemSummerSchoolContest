#!/bin/bash
# Edge case tests for C3.5 inference
set -e
SOL="/home/mig20/c3_solution"
MODEL="$HOME/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models/mlp_v1.onnx"
INPUT="$HOME/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35/mlp_v1/input"
GOLDEN="$HOME/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35/mlp_v1/golden"
LABELS="$HOME/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35/mlp_v1/labels.npy"

echo "=== Edge Case 1: batch_size=1 ==="
bash "$SOL/run_infer.sh" --onnx "$MODEL" --input "$INPUT" --output /tmp/edge_b1 --batch-size 1
python3 "$SOL/verify.py" /tmp/edge_b1 "$GOLDEN" "$LABELS" 0.98
echo "PASS"

echo "=== Edge Case 2: batch_size > total samples (50000) ==="
bash "$SOL/run_infer.sh" --onnx "$MODEL" --input "$INPUT" --output /tmp/edge_big --batch-size 50000
python3 "$SOL/verify.py" /tmp/edge_big "$GOLDEN" "$LABELS" 0.98
echo "PASS"

echo "=== Edge Case 3: batch_size=1000 (non-power-of-2) ==="
bash "$SOL/run_infer.sh" --onnx "$MODEL" --input "$INPUT" --output /tmp/edge_b1000 --batch-size 1000
python3 "$SOL/verify.py" /tmp/edge_b1000 "$GOLDEN" "$LABELS" 0.98
echo "PASS"

echo "=== Edge Case 4: C3.1 with missing file ==="
python3 "$SOL/export_dag.py" --onnx /nonexistent/model.onnx --output /tmp/bad.json 2>&1 && echo "UNEXPECTED SUCCESS" || echo "PASS (expected error)"

echo "=== ALL EDGE CASES PASSED ==="
