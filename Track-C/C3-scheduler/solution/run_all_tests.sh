#!/bin/bash
# Full C3 self-test suite
set -e
MODELS="/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models"
TESTDATA="/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35"
SOL="/home/mig20/c3_solution"

echo '═══════════════════════════════════'
echo '  C3.1: ONNX -> DAG JSON'
echo '═══════════════════════════════════'
for m in mlp_v1 resnet_v1 transformer_v1; do
    python3 "$SOL/export_dag.py" --onnx "$MODELS/$m.onnx" --output "/tmp/dag_$m.json"
    python3 -c "import json; d=json.load(open('/tmp/dag_${m}.json')); print('  $m:', len(d['nodes']), 'nodes,', len(d['edges']), 'edges')"
done

echo ''
echo '═══════════════════════════════════'
echo '  C3.2 + C3.3: 算子分解/融合'
echo '═══════════════════════════════════'
python3 "$SOL/test_c32_c33.py"

echo ''
echo '═══════════════════════════════════'
echo '  C3.4: 内存规划'
echo '═══════════════════════════════════'
python3 "$SOL/test_c34.py"

echo ''
echo '═══════════════════════════════════'
echo '  C3.5: 端到端推理'
echo '═══════════════════════════════════'

echo '--- MLP (GPU) ---'
bash "$SOL/run_infer.sh" --onnx "$MODELS/mlp_v1.onnx" --input "$TESTDATA/mlp_v1/input" --output /tmp/test_mlp --batch-size 256
python3 "$SOL/verify.py" /tmp/test_mlp "$TESTDATA/mlp_v1/golden" "$TESTDATA/mlp_v1/labels.npy" 0.98

echo ''
echo '--- ResNet (GPU) ---'
bash "$SOL/run_infer.sh" --onnx "$MODELS/resnet_v1.onnx" --input "$TESTDATA/resnet_v1/input" --output /tmp/test_resnet --batch-size 256
python3 "$SOL/verify.py" /tmp/test_resnet "$TESTDATA/resnet_v1/golden" "$TESTDATA/resnet_v1/labels.npy" 0.85

echo ''
echo '--- Transformer (GPU) ---'
bash "$SOL/run_infer.sh" --onnx "$MODELS/transformer_v1.onnx" --input "$TESTDATA/transformer_v1/input" --output /tmp/test_tf --batch-size 256
python3 "$SOL/verify.py" /tmp/test_tf "$TESTDATA/transformer_v1/golden"

echo ''
echo '═══════════════════════════════════'
echo '  ALL TESTS COMPLETE'
echo '═══════════════════════════════════'
