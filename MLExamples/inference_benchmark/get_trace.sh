#!/bin/bash
# Script to profile inference_benchmark with rocprofv3 runtime trace
# This captures GPU API calls, kernel launches, and memory operations

set -e

# Create output directory with timestamp
OUTPUT_DIR="profiling_results/trace_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Starting rocprofv3 runtime trace profiling for inference_benchmark..."
echo "Output directory: $OUTPUT_DIR"

# Run with rocprofv3 to collect runtime trace
# Using resnet50 as the default network with standard batch size
rocprofv3 \
    --hip-trace \
    --hsa-trace \
    --marker-trace \
    --output-directory "$OUTPUT_DIR" \
    -- python micro_benchmarking_pytorch.py \
    --network resnet50 \
    --batch-size 64 \
    --iterations 10

echo ""
echo "Profiling complete! Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR"
echo ""
echo "To view the trace, open the .pftrace file in Perfetto UI:"
echo "https://ui.perfetto.dev/"
echo ""

# Find and highlight the pftrace file
PFTRACE_FILE=$(find "$OUTPUT_DIR" -name "*.pftrace" | head -1)
if [ -n "$PFTRACE_FILE" ]; then
    echo "Trace file: $PFTRACE_FILE"
    echo "Size: $(du -h "$PFTRACE_FILE" | cut -f1)"
fi
