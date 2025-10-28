#!/bin/bash
#
# Get a trace using rocprofv3 with runtime tracing
#

set -e

echo "=========================================="
echo "rocprofv3 Runtime Trace - Version 1"
echo "=========================================="
echo ""

OUTPUT_DIR="./traces/trace_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Running: rocprofv3 --runtime-trace --output-format pftrace -- python tiny_llama_v1.py --batch-size 8 --seq-len 128 --num-steps 10"
echo ""

cd "$OUTPUT_DIR"
rocprofv3 --runtime-trace --output-format pftrace -- python ../../tiny_llama_v1.py --batch-size 8 --seq-len 128 --num-steps 10
ROCPROF_EXIT=$?

echo ""
if [ $ROCPROF_EXIT -eq 0 ]; then
    echo "[SUCCESS] Trace generation completed"
else
    echo "[FAILED] Trace generation failed with exit code $ROCPROF_EXIT"
    exit 1
fi
echo ""

echo "Generated files:"
find . -type f -ls
echo ""

echo "Perfetto trace files:"
find . -name "*.pftrace" -exec ls -lh {} \;
echo ""

echo "To view trace:"
echo "  Visit: https://ui.perfetto.dev/"
echo "  Open the largest .pftrace file"
echo ""
