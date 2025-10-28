#!/bin/bash
#
# Get hardware performance counters using rocprofv3
#

set -e

echo "=========================================="
echo "rocprofv3 Hardware Counters - Version 3"
echo "=========================================="
echo ""

OUTPUT_DIR="./counters/counter_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Run with kernel trace to collect counter data
# rocprofv3 automatically collects available counters with --kernel-trace
echo "Running: rocprofv3 --kernel-trace -- python tiny_llama_v3.py --batch-size 8 --seq-len 128 --num-steps 10"
echo ""

cd "$OUTPUT_DIR"
rocprofv3 --kernel-trace -- python ../../tiny_llama_v3.py --batch-size 8 --seq-len 128 --num-steps 10
ROCPROF_EXIT=$?

echo ""
if [ $ROCPROF_EXIT -eq 0 ]; then
    echo "[SUCCESS] Counter collection completed"
else
    echo "[FAILED] Counter collection failed with exit code $ROCPROF_EXIT"
    exit 1
fi
echo ""

echo "Generated files:"
find . -type f -ls
echo ""

# Find the kernel trace CSV file
KERNEL_TRACE=$(find . -name "*kernel_trace.csv" -type f | head -1)

if [ -n "$KERNEL_TRACE" ]; then
    echo "Found kernel trace: $KERNEL_TRACE"
    echo ""
    echo "Analyzing kernel trace data..."
    echo ""

    cd ../..
    python analyze_kernel_trace.py "$OUTPUT_DIR/$KERNEL_TRACE"

    echo ""
else
    echo "[WARNING] No kernel_trace.csv file found"
    echo ""
    echo "Looking for other counter data:"
    find . \( -name "*.csv" -o -name "*.json" -o -name "*.txt" \) -exec echo "Found: {}" \;
    echo ""
fi

echo "Hardware counters provide detailed GPU performance metrics:"
echo "  - Memory bandwidth utilization"
echo "  - Cache hit rates"
echo "  - Compute unit occupancy"
echo "  - VGPR/SGPR usage"
echo ""
