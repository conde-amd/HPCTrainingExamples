#!/bin/bash
# Script to profile TinyTransformer V4 with rocprofv3 runtime trace
# This captures GPU API calls, kernel launches, and memory operations
#
# Compatible with ROCm 6.x and 7.x

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
OUTPUT_DIR="./traces/trace_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Starting rocprofv3 runtime trace profiling for TinyTransformer V4..."
echo "Output directory: $OUTPUT_DIR"

# Build rocprofv3 command with appropriate flags for ROCm version
# ROCm 6.4+ and 7.x require explicit --output-format pftrace to generate Perfetto traces
if [ "$ROCM_MAJOR" = "7" ] || [ "$ROCM_MAJOR" = "6" ]; then
    echo "Using ROCm 6.x/7.x: --output-format pftrace (generates Perfetto trace)"
    OUTPUT_FORMAT="--output-format pftrace"
else
    echo "Using ROCm 5.x or older: default format"
    OUTPUT_FORMAT=""
fi

echo ""
echo "Collecting full runtime trace (HIP/HSA API calls, kernels, memory operations)"
echo ""

# Run with rocprofv3 to collect full runtime trace
cd "$OUTPUT_DIR"
rocprofv3 \
    --runtime-trace \
    $OUTPUT_FORMAT \
    -- python ../../tiny_llama_v4.py --batch-size 8 --seq-len 128 --num-steps 10

echo ""
echo "Profiling complete! Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
ls -lh ./*/ 2>/dev/null || ls -lh .
echo ""

# Find and report pftrace files
PFTRACE=$(find . -name "*.pftrace" -size +1k 2>/dev/null | head -1)
if [ -n "$PFTRACE" ]; then
    echo "Perfetto trace file: $PFTRACE"
    echo "Size: $(ls -lh "$PFTRACE" | awk '{print $5}')"
    echo ""
    echo "To view the trace:"
    echo "  1. Visit: https://ui.perfetto.dev/"
    echo "  2. Open: $PFTRACE"
fi
