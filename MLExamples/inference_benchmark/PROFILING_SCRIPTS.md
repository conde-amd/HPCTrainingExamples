# Profiling Scripts for inference_benchmark

This directory contains profiling scripts for analyzing the performance of PyTorch inference benchmarks using various ROCm profiling tools.

## Overview

All scripts are configured to profile **ResNet50** with:
- Batch size: 64
- Iterations: 10

The scripts use the standard command:
```bash
python micro_benchmarking_pytorch.py --network resnet50 --batch-size 64 --iterations 10
```

## Available Profiling Scripts

### 1. get_counters.sh - rocprofv3 Kernel Trace with Hardware Counters

**Purpose:** Captures detailed GPU hardware metrics and kernel execution statistics

**Features:**
- Collects hardware counter data for all GPU kernels
- Includes `analyze_kernel_trace.py` for automatic analysis
- Shows kernel execution statistics and performance hotspots
- Identifies top time-consuming kernels

**Output:**
- `profiling_results/counters_<timestamp>/` directory
- `kernel_trace.csv` with detailed kernel metrics
- Automated analysis summary showing:
  - Kernel execution counts
  - Total/average/min/max durations
  - Percentage of total GPU time

**Usage:**
```bash
./get_counters.sh
```

**When to use:**
- Identify performance bottlenecks at the kernel level
- Understand which GPU operations consume the most time
- Analyze kernel execution patterns and frequencies

---

### 2. get_trace.sh - rocprofv3 Runtime Trace

**Purpose:** Captures GPU API calls, kernel launches, and memory operations

**Features:**
- Records HIP/HSA API calls
- Traces kernel launches and execution
- Captures memory operations (allocations, transfers)
- Generates Perfetto trace format (.pftrace) for visualization

**Output:**
- `profiling_results/trace_<timestamp>/` directory
- `.pftrace` file for interactive timeline visualization

**Visualization:**
Open the `.pftrace` file at [https://ui.perfetto.dev/](https://ui.perfetto.dev/)

**Usage:**
```bash
./get_trace.sh
```

**When to use:**
- Visualize timeline of GPU operations
- Analyze CPU-GPU synchronization
- Identify memory transfer bottlenecks
- Understand overall execution flow

---

### 3. get_rocprof_sys.sh - System-Level Profiling

**Purpose:** System-level profiling with call stack sampling

**Features:**
- Call stack sampling for CPU and GPU code
- System-level performance analysis
- Captures both application and runtime behavior

**Output:**
- `profiling_results/rocprof_sys_<timestamp>/` directory
- System-level profiling data

**Known Issues:**
⚠️ **Note:** rocprof-sys may produce memory map dumps in some configurations. This is a known issue tracked in GitHub issue #1406. If profiling fails or produces excessive output, consider using `get_trace.sh` (rocprofv3) or `get_rocprof_compute.sh` instead.

**Usage:**
```bash
./get_rocprof_sys.sh
```

**Analysis:**
```bash
rocprof-sys-avail --help
rocprof-sys-analyze --help
```

**When to use:**
- System-level performance analysis
- Call stack profiling
- When kernel-level profiling is insufficient

---

### 4. get_rocprof_compute.sh - Detailed GPU Metrics

**Purpose:** Comprehensive compute performance analysis with detailed hardware metrics

**Features:**
- Detailed GPU hardware counter collection
- Compute performance analysis
- Unique workload names with timestamps
- Comprehensive metric coverage

**Output:**
- `profiling_results/rocprof_compute_<timestamp>/` directory
- Workload-specific performance data

**Usage:**
```bash
./get_rocprof_compute.sh
```

**Analysis:**
```bash
rocprof-compute analyze --help
rocprof-compute analyze --workload-dir profiling_results/rocprof_compute_<timestamp>
```

**When to use:**
- Detailed hardware performance analysis
- Compute utilization metrics
- Memory bandwidth and cache analysis
- Advanced performance tuning

---

## Workflow Recommendations

### Quick Performance Check
1. Start with `get_counters.sh` to identify top kernels
2. Review the automated analysis for hotspots

### Detailed Analysis
1. Run `get_trace.sh` to visualize execution timeline
2. Open `.pftrace` in Perfetto UI to analyze CPU-GPU interaction
3. Run `get_rocprof_compute.sh` for detailed hardware metrics

### Advanced Tuning
1. Use `get_rocprof_compute.sh` for comprehensive metrics
2. Analyze specific hardware counters
3. Iterate on optimizations and re-profile

---

## Output Directory Structure

All scripts create timestamped output directories:
```
profiling_results/
├── counters_YYYYMMDD_HHMMSS/
├── trace_YYYYMMDD_HHMMSS/
├── rocprof_sys_YYYYMMDD_HHMMSS/
└── rocprof_compute_YYYYMMDD_HHMMSS/
```

---

## Customizing Profiling Runs

To profile different networks or configurations, modify the scripts to use different arguments:

```bash
# Example: Profile VGG16 with larger batch size
python micro_benchmarking_pytorch.py --network vgg16 --batch-size 128 --iterations 10

# Example: Profile with FP16
python micro_benchmarking_pytorch.py --network resnet50 --batch-size 64 --iterations 10 --fp16 1

# Example: Profile with PyTorch 2.0 compile
python micro_benchmarking_pytorch.py --network resnet50 --batch-size 64 --iterations 10 --compile
```

Available networks include: `alexnet`, `densenet121`, `inception_v3`, `resnet50`, `resnet101`, `SqueezeNet`, `vgg16`, and more.

---

## Requirements

- ROCm 6.4.4 or later
- AMD GPU (tested on RX 7900 XTX / gfx1100)
- Profiling tools installed:
  - `rocprofv3`
  - `rocprof-compute`
  - `rocprof-sys`
- Python 3 with PyTorch (ROCm build)

---

## Troubleshooting

### Locale Errors (rocprof-compute)
If you see: `ERROR Please ensure that the 'en_US.UTF-8' locale is available`

**Solution:** Rebuild the devcontainer (Dockerfiles already updated) or set locale manually:
```bash
export LANG=en_US.UTF-8
export LANGUAGE=en_US:en
export LC_ALL=en_US.UTF-8
```

### Memory Map Dumps (rocprof-sys)
If `get_rocprof_sys.sh` produces excessive memory map output instead of clean profiles, this is a known issue. Use alternative profilers: `get_trace.sh` or `get_rocprof_compute.sh`.

### Permission Errors
Ensure scripts are executable:
```bash
chmod +x get_*.sh
```

---

## Additional Resources

- [ROCm Profiling Documentation](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/)
- [Perfetto UI](https://ui.perfetto.dev/)
- [MIOpen Performance Database](https://rocm.github.io/MIOpen/doc/html/perfdatabase.html)

---

## Related Files

- `README.md` - Main documentation for inference_benchmark
- `analyze_kernel_trace.py` - Kernel trace analysis script (auto-created by `get_counters.sh`)
- `micro_benchmarking_pytorch.py` - Main benchmark script
