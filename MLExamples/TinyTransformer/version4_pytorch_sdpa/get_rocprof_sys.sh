#!/bin/bash
#
# Get system-level profiling using rocprof-sys
# Compatible with ROCm 6.x and 7.x
#
# NOTE: rocprof-sys may produce memory map dumps in some configurations.
# Issue reference: TBD
#

set -e

echo "=========================================="
echo "rocprof-sys Profiling - TinyTransformer V4"
echo "=========================================="
echo ""

OUTPUT_DIR="./rocprof_sys/profile_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Run with rocprof-sys to collect system-level traces
echo "Running: rocprof-sys-run --profile --trace -- python tiny_llama_v4.py --batch-size 8 --seq-len 128 --num-steps 10"
echo ""

cd "$OUTPUT_DIR"
rocprof-sys-run --profile --trace -- python ../../tiny_llama_v4.py --batch-size 8 --seq-len 128 --num-steps 10
ROCPROF_EXIT=$?

echo ""
if [ $ROCPROF_EXIT -eq 0 ]; then
    echo "[SUCCESS] rocprof-sys profiling completed"
else
    echo "[FAILED] rocprof-sys profiling failed with exit code $ROCPROF_EXIT"
    exit 1
fi
echo ""

echo "Generated files:"
find . -type f -ls | head -20
echo ""

echo "To analyze results:"
echo "  Open the .proto file in Perfetto UI: https://ui.perfetto.dev/"
echo ""
