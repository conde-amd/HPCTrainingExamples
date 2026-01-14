# ML Example: TinyTransformer Triton with ROCm Profiling

README.md from `HPCTrainingExamples/MLExamples/TinyTransformer/version3_triton` from the Training Examples repository.

In this example we provide a Triton-optimized implementation of Tiny LLaMA with custom GPU kernels for profiling transformer workloads on AMD GPUs. This version builds on version2 with custom Triton kernels for RMSNorm, Flash Attention, and a hybrid SwiGLU approach. Several profiling scripts are provided to capture different aspects of GPU performance.

## Features of the profiling scripts

The version3_triton example contains several profiling scripts that capture different aspects of GPU performance:

- **get_trace.sh**: Runtime trace collection using rocprofv3. Captures HIP/HSA API calls, kernel execution timeline, memory operations (H2D, D2H, D2D transfers), and synchronization events. Output is a Perfetto trace file for timeline visualization.
- **get_counters.sh**: Kernel trace collection using rocprofv3. Captures kernel execution statistics including timing and call counts. Useful for identifying hotspot kernels and their execution patterns.
- **get_rocprof_compute.sh**: Detailed GPU hardware metrics using rocprof-compute. Provides comprehensive performance analysis including compute utilization, memory bandwidth, and hardware counter data.
- **get_rocprof_sys.sh**: System-level profiling using rocprof-sys. Captures call stack sampling and system-level performance data for end-to-end analysis.
- **get_hotspots.sh**: GPU hotspot analysis using rocprofv3 stats mode. Identifies kernels with highest time consumption.

## Key Optimizations

This version implements custom Triton GPU kernels:

- **RMSNorm Triton Kernel**: Fused variance computation and normalization (3 kernels → 1)
- **Flash Attention Triton Kernel**: Memory-efficient attention with O(S) complexity instead of O(S²)
- **Hybrid SwiGLU**: PyTorch for matrix multiplications + Triton for activation fusion
- **Automatic Tuning**: Triton compiler optimizations for target hardware

## Overview of the model

The model is controlled with the following arguments:

- `--batch-size <N>`: batch size for training (default: 8)
- `--seq-len <N>`: sequence length (default: 256)
- `--num-steps <N>`: number of training steps (default: 50)
- `--hidden-dim <N>`: hidden dimension (default: 512)
- `--num-layers <N>`: number of transformer layers (default: 8)
- `--num-heads <N>`: number of attention heads (default: 8)
- `--learning-rate <float>`: learning rate (default: 3e-4)
- `--use-amp`: enable automatic mixed precision

## Running the Triton model

Load the required modules:

```
module load pytorch rocm triton
```

Run a basic training run:

```
echo "Running TinyTransformer V3 Triton"
python tiny_llama_v3.py --batch-size 8 --seq-len 128 --num-steps 10
```

## Runtime Trace Profiling with get_trace.sh

This script captures GPU API calls, kernel launches, and memory operations for timeline analysis.

Run the profiling script:

```
echo "Collecting runtime trace with rocprofv3"
./get_trace.sh
```

The script will output results to `traces/trace_<timestamp>/`. To analyze the results:

```
echo "Opening trace in Perfetto UI"
echo "Visit https://ui.perfetto.dev/ and open the .pftrace file"
```

## Kernel Trace Profiling with get_counters.sh

This script collects kernel execution statistics including timing and call counts.

Run the profiling script:

```
echo "Collecting kernel trace with rocprofv3"
./get_counters.sh
```

The script will output results to `counters/counter_<timestamp>/`.

ROCm 6.x outputs CSV files directly, while ROCm 7.x outputs SQLite databases. For ROCm 7.x database files, use rocpd tools:

```
echo "Exporting kernel statistics to CSV"
rocpd2csv -i <db_file> -o kernel_stats.csv
```

```
echo "Getting kernel summary"
rocpd summary -i <db_file> --region-categories KERNEL
```

Documentation for rocpd tools: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-rocpd-output-format.html

## GPU Hardware Metrics with get_rocprof_compute.sh

This script collects detailed GPU performance metrics for hardware utilization analysis.

Run the profiling script:

```
echo "Collecting GPU hardware metrics with rocprof-compute"
./get_rocprof_compute.sh
```

The script will output results to `rocprof_compute/profile_<timestamp>/`. To analyze the results:

```
echo "Generating performance analysis report"
rocprof-compute analyze -p <output_dir>/workloads/<workload_name>/rocprof --dispatch <N> -n tiny_llama_dispatch
```

For available analysis options:

```
rocprof-compute analyze --help
```

Note: rocprof-compute requires data center GPUs (MI100, MI200, MI300 series) for full hardware counter support. Consumer GPUs may have limited counter availability.

## System-Level Profiling with get_rocprof_sys.sh

This script captures system-level performance with call stack sampling.

Run the profiling script:

```
echo "Collecting system-level profile with rocprof-sys"
./get_rocprof_sys.sh
```

The script will output results to `rocprof_sys/profile_<timestamp>/`. To analyze the results:

```
echo "Opening trace in Perfetto UI"
echo "Visit https://ui.perfetto.dev/ and open the .proto file"
```

Note: rocprof-sys may produce memory map dumps in some configurations. If profiling fails or produces excessive output, consider using rocprofv3 (get_trace.sh) instead.

## GPU Hotspot Analysis with get_hotspots.sh

This script identifies kernels with the highest execution time using rocprofv3 stats mode.

Run the profiling script:

```
echo "Collecting GPU hotspots"
./get_hotspots.sh
```

The script will output kernel statistics to `hotspots/hotspot_<timestamp>/`.

## Expected Performance Improvements

Results from AMD MI325X with ROCm 6.4.4:

| Version | Throughput | Memory | Improvement |
|---------|-----------|--------|-------------|
| V1 Baseline | 372.9 samples/sec | 522.3 MB | - |
| V3 Triton | 2065.0 samples/sec | 281.8 MB | 5.5x faster, 46% less memory |

Key optimizations impact:
- Flash Attention (Triton): 46% memory reduction
- RMSNorm (Triton): 3 kernels → 1
- Hybrid SwiGLU: PyTorch matmul + Triton activation

## Additional Resources

- rocprofv3 documentation: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-rocprofv3.html
- rocpd output format: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-rocpd-output-format.html
- Perfetto UI: https://ui.perfetto.dev/
- Triton Language Tutorial: https://triton-lang.org/main/getting-started/tutorials/index.html
