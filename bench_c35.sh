#!/bin/bash
# C3.5 performance benchmark — runtime + peak memory
set -e

SOL="/home/mig20/c3_solution"
MODEL_BASE="/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models"
DATA_BASE="/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35"

bench_one() {
    local name=$1
    local model="$MODEL_BASE/${name}.onnx"
    local input="$DATA_BASE/${name}/input"
    local output="/tmp/bench_${name}"
    local bs=$2

    echo "--- $name (batch=$bs) ---"

    # Clear GPU memory before each run
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

    # Start memory monitor in background
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 0.2 > /tmp/mem_$$.log 2>/dev/null &
    local monitor_pid=$!

    # Time the run
    local start=$(date +%s.%N)
    bash "$SOL/run_infer.sh" --onnx "$model" --input "$input" --output "$output" --batch-size "$bs" 2>/dev/null
    local end=$(date +%s.%N)

    # Stop monitor
    kill $monitor_pid 2>/dev/null; wait $monitor_pid 2>/dev/null

    local elapsed=$(echo "$end - $start" | bc)
    local peak_mem=$(sort -n /tmp/mem_$$.log | tail -1 2>/dev/null || echo "N/A")
    rm -f /tmp/mem_$$.log

    # Verify precision
    local golden="$DATA_BASE/${name}/golden"
    local labels="$DATA_BASE/${name}/labels.npy"
    local threshold=""
    case $name in
        mlp_v1) threshold="0.98" ;;
        resnet_v1) threshold="0.85" ;;
        *) threshold="" ;;
    esac

    if [ -n "$threshold" ]; then
        python3 "$SOL/verify.py" "$output" "$golden" "$labels" "$threshold" 2>&1 | grep -E "Precision|Accuracy|Max"
    else
        python3 "$SOL/verify.py" "$output" "$golden" 2>&1 | grep -E "Precision|Max"
    fi

    printf "  Time: %.3fs  Peak GPU mem: %s MB\n" "$elapsed" "$peak_mem"
    echo ""
}

echo "=============================================="
echo " C3.5 Performance Benchmark"
echo "=============================================="
echo ""

bench_one "mlp_v1" 256
bench_one "resnet_v1" 256
bench_one "transformer_v1" 256

echo "=============================================="
echo " Done"
echo "=============================================="
