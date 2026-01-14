#!/bin/bash
#
# Get detailed GPU metrics using rocprof-compute
# Compatible with ROCm 6.x and 7.x
#
# Note: rocprof-compute requires data center GPUs (MI100, MI200, MI300 series)
# for full hardware counter support. Consumer GPUs may have limited counter availability.

set -e

echo "=========================================="
echo "rocprof-compute Profiling - TinyTransformer V4"
echo "=========================================="
echo ""

OUTPUT_DIR="./rocprof_compute/profile_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Run with rocprof-compute to collect detailed GPU metrics
WORKLOAD_NAME="tiny_llama_v4_$(date +%Y%m%d_%H%M%S)"
echo "Running: rocprof-compute profile --name $WORKLOAD_NAME -d $OUTPUT_DIR -- python tiny_llama_v4.py --batch-size 8 --seq-len 128 --num-steps 10"
echo ""

rocprof-compute profile --name "$WORKLOAD_NAME" -d "$OUTPUT_DIR" -- python tiny_llama_v4.py --batch-size 8 --seq-len 128 --num-steps 10
ROCPROF_EXIT=$?

echo ""
if [ $ROCPROF_EXIT -eq 0 ]; then
    echo "[SUCCESS] rocprof-compute profiling completed"
else
    echo "[FAILED] rocprof-compute profiling failed with exit code $ROCPROF_EXIT"
    exit 1
fi
echo ""

echo "Generated files:"
find "$OUTPUT_DIR" -type f -ls | head -20
echo ""

echo "To analyze results:"
echo "  rocprof-compute analyze -p $OUTPUT_DIR/workloads/$WORKLOAD_NAME/rocprof --dispatch <N> -n tiny_llama_dispatch"
echo ""
echo "For available analysis options:"
echo "  rocprof-compute analyze --help"
echo ""
