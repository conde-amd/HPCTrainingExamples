# ML Example: TinyTransformer PyTorch SDPA with ROCm Profiling

README.md from `HPCTrainingExamples/MLExamples/TinyTransformer/version4_pytorch_sdpa` from the Training Examples repository.

This version implements ultra-fused Triton kernels with PyTorch SDPA (Scaled Dot Product Attention) for maximum performance. It builds on version3 with complete transformer block fusion, achieving 3.14x speedup and 61% memory reduction over baseline.

## Features of the profiling scripts

The version4_pytorch_sdpa example contains several profiling scripts that capture different aspects of GPU performance:

- **get_trace.sh**: Runtime trace collection using rocprofv3. Captures HIP/HSA API calls, kernel execution timeline, memory operations (H2D, D2H, D2D transfers), and synchronization events. Output is a Perfetto trace file for timeline visualization.
- **get_counters.sh**: Kernel trace collection using rocprofv3. Captures kernel execution statistics including timing and call counts. Useful for identifying hotspot kernels and their execution patterns.
- **get_rocprof_compute.sh**: Detailed GPU hardware metrics using rocprof-compute. Provides comprehensive performance analysis including compute utilization, memory bandwidth, and hardware counter data.
- **get_rocprof_sys.sh**: System-level profiling using rocprof-sys. Captures call stack sampling and system-level performance data for end-to-end analysis.
- **get_hotspots.sh**: GPU hotspot analysis using rocprofv3 stats mode. Identifies kernels with highest time consumption.

## Key Optimizations

This version implements the pinnacle of GPU optimization:

- **PyTorch SDPA**: Hardware-accelerated scaled dot product attention with automatic Flash Attention backend
- **Ultra-Fused Transformer Block**: Entire transformer block in single kernel launch (12 kernels → 1)
- **Advanced Memory Management**: Optimal register and cache utilization, 85-98% memory bandwidth reduction
- **Adaptive Block Sizing**: Hardware-aware block size optimization for different GPU architectures

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

## Running the ultra-fused model

Load the required modules:

```
module load pytorch rocm triton
```

Run a basic training run:

```
echo "Running TinyTransformer V4 Ultra-Fused"
python tiny_llama_v4.py --batch-size 8 --seq-len 128 --num-steps 10
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
| V4 Ultra-Fused | 1171.0 samples/sec | 203.5 MB | 3.14x faster, 61% less memory |

Key optimization impacts:
- Ultra-fused transformer block: 12 kernel launches → 1
- PyTorch SDPA: Hardware-accelerated attention with Flash Attention backend
- Memory hierarchy optimization: 85-98% intermediate memory elimination

## Additional Resources

- rocprofv3 documentation: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-rocprofv3.html
- rocpd output format: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-rocpd-output-format.html
- Perfetto UI: https://ui.perfetto.dev/
- Triton Language Tutorial: https://triton-lang.org/main/getting-started/tutorials/index.html
- Flash Attention Paper: https://arxiv.org/abs/2205.14135
