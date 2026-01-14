# ML Example: TinyTransformer Baseline with ROCm Profiling

README.md from `HPCTrainingExamples/MLExamples/TinyTransformer/version1_pytorch_baseline` from the Training Examples repository.

In this example we provide a baseline PyTorch implementation of Tiny LLaMA for profiling transformer workloads on AMD GPUs. The model runs forward and backward passes with configurable batch size and sequence length, measuring training throughput. This workload is useful for understanding transformer performance characteristics and for learning ROCm profiling tools. Several profiling scripts are provided to capture different aspects of GPU performance, from high-level API traces to detailed hardware metrics.

## Features of the profiling scripts

The version1_pytorch_baseline example contains several profiling scripts that capture different aspects of GPU performance:

- **get_trace.sh**: Runtime trace collection using rocprofv3. Captures HIP/HSA API calls, kernel execution timeline, memory operations (H2D, D2H, D2D transfers), and synchronization events. Output is a Perfetto trace file for timeline visualization.
- **get_counters.sh**: Kernel trace collection using rocprofv3. Captures kernel execution statistics including timing and call counts. Useful for identifying hotspot kernels and their execution patterns.
- **get_rocprof_compute.sh**: Detailed GPU hardware metrics using rocprof-compute. Provides comprehensive performance analysis including compute utilization, memory bandwidth, and hardware counter data.
- **get_rocprof_sys.sh**: System-level profiling using rocprof-sys. Captures call stack sampling and system-level performance data for end-to-end analysis.

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
- `--enable-pytorch-profiler`: enable PyTorch profiler
- `--enable-deepspeed-flops`: enable DeepSpeed FLOPS profiler

## Running the baseline

Load the required modules:

```
module load pytorch rocm
```

Run a basic training run:

```
echo "Running TinyTransformer baseline"
python tiny_llama_v1.py --batch-size 8 --seq-len 128 --num-steps 10
```

For mixed precision training:

```
echo "Running with automatic mixed precision"
python tiny_llama_v1.py --batch-size 16 --seq-len 128 --num-steps 10 --use-amp
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

If a `.db` file is generated instead (ROCm 7.x without --output-format):

```
echo "Converting database to Perfetto format"
rocpd2pftrace -i <db_file> -o trace.pftrace
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

## Additional Resources

- rocprofv3 documentation: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-rocprofv3.html
- rocpd output format: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-rocpd-output-format.html
- Perfetto UI: https://ui.perfetto.dev/
