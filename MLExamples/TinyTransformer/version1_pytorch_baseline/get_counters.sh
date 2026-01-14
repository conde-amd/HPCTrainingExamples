#!/bin/bash
# Script to profile TinyTransformer with rocprofv3 kernel trace
# This captures kernel execution metrics for performance analysis
#
# Supports both ROCm 6.x (CSV output) and ROCm 7.x (SQLite database output)

set -e

# Detect ROCm version
ROCM_VERSION=""
ROCM_MAJOR=""

# Method 1: Check rocminfo
if command -v rocminfo &> /dev/null; then
    ROCM_VERSION=$(rocminfo | grep -i "ROCm Version" | head -1 | awk '{print $3}')
fi

# Method 2: Check ROCM_PATH
if [ -z "$ROCM_VERSION" ] && [ -n "$ROCM_PATH" ]; then
    if [ -f "$ROCM_PATH/.info/version" ]; then
        ROCM_VERSION=$(cat "$ROCM_PATH/.info/version")
    fi
fi

# Method 3: Check hipcc version (more reliable for module-loaded ROCm)
if [ -z "$ROCM_VERSION" ] && command -v hipcc &> /dev/null; then
    HIP_VERSION=$(hipcc --version 2>/dev/null | grep -i "HIP version" | head -1 | awk '{print $3}')
    if [ -n "$HIP_VERSION" ]; then
        ROCM_VERSION="$HIP_VERSION"
    fi
fi

# Extract major version
if [ -n "$ROCM_VERSION" ]; then
    ROCM_MAJOR=$(echo "$ROCM_VERSION" | cut -d. -f1)
    echo "Detected ROCm version: $ROCM_VERSION"
else
    echo "Warning: Could not detect ROCm version, assuming ROCm 7.x"
    ROCM_MAJOR="7"
fi

# Create output directory with timestamp
OUTPUT_DIR="./counters/counter_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Starting rocprofv3 kernel trace collection for TinyTransformer..."
echo "Output directory: $OUTPUT_DIR"

# Run with rocprofv3 to collect kernel trace
rocprofv3 \
    --kernel-trace \
    --output-directory "$OUTPUT_DIR" \
    -- python tiny_llama_v1.py \
    --batch-size 8 \
    --seq-len 128 \
    --num-steps 10

echo ""
echo "Profiling complete! Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR"/*/ 2>/dev/null || ls -lh "$OUTPUT_DIR"
echo ""

# Analyze results based on ROCm version
echo "To analyze results:"
DB_FILE=$(find "$OUTPUT_DIR" -name "*_results.db" 2>/dev/null | head -1)
if [ -n "$DB_FILE" ]; then
    echo "  Database file: $DB_FILE"
    echo ""
    echo "  Export to CSV:"
    echo "    rocpd2csv -i $DB_FILE -o kernel_stats.csv"
    echo ""
    echo "  Get kernel summary:"
    echo "    rocpd summary -i $DB_FILE --region-categories KERNEL"
else
    echo "  Check $OUTPUT_DIR for output files"
fi
