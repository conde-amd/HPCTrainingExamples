# ML Example: PyTorch Micro-Benchmarking with ROCm Profiling

README.md from `HPCTrainingExamples/MLExamples/pytorch_microbench` from the Training Examples repository.

In this example we provide a PyTorch micro-benchmarking tool for measuring GPU throughput on AMD GPUs. The benchmark runs forward and backward passes on various CNN architectures, measuring images processed per second. This workload is useful for establishing baseline GPU performance and for learning ROCm profiling tools. Several profiling scripts are provided to capture different aspects of GPU performance, from high-level API traces to detailed hardware metrics.

## Features of the profiling scripts

The pytorch_microbench example contains several profiling scripts that capture different aspects of GPU performance:

- **get_trace.sh**: Runtime trace collection using rocprofv3. Captures HIP/HSA API calls, kernel execution timeline, memory operations (H2D, D2H, D2D transfers), and synchronization events. Output is a Perfetto trace file for timeline visualization.
- **get_counters.sh**: Kernel trace collection using rocprofv3. Captures kernel execution statistics including timing and call counts. Useful for identifying hotspot kernels and their execution patterns.
- **get_rocprof_compute.sh**: Detailed GPU hardware metrics using rocprof-compute. Provides comprehensive performance analysis including compute utilization, memory bandwidth, and hardware counter data.
- **get_rocprof_sys.sh**: System-level profiling using rocprof-sys. Captures call stack sampling and system-level performance data for end-to-end analysis.

## Overview of the benchmark

The benchmark is controlled with the following arguments:

- `--network <name>`: neural network architecture to benchmark (alexnet, densenet121, inception_v3, resnet50, resnet101, SqueezeNet, vgg16, etc.)
- `--batch-size <N>`: batch size for forward/backward passes (default: 64)
- `--iterations <N>`: number of iterations to run (default: 10)
- `--fp16 <0|1>`: enable FP16 precision (default: 0, disabled)
- `--compile`: enable PyTorch 2.0 torch.compile optimizations
- `--compileContext <dict>`: compilation options as Python dict string
- `--distributed_dataparallel`: use DistributedDataParallel for multi-GPU
- `--device_ids <ids>`: comma-separated GPU indices for distributed runs

## Running the micro-benchmark

Load the required modules:

```
module load pytorch rocm
```

Run a basic micro-benchmark with ResNet50:

```
echo "Running ResNet50 micro-benchmark"
python micro_benchmarking_pytorch.py --network resnet50 --batch-size 64 --iterations 10
```

Example output (Radeon RX 7900 XTX, ROCm 6.4):

```
INFO: running forward and backward for warmup.
INFO: running the benchmark..
OK: finished running benchmark..
--------------------SUMMARY--------------------------
Microbenchmark for network : resnet50
Num devices: 1
Dtype: FP32
Mini batch size [img] : 64
Time per mini-batch : 0.177
Throughput [img/sec] : 360.74
```

Note the throughput reported in images/second. This measures the combined forward and backward pass performance.

For multi-GPU runs using torchrun (recommended):

```
echo "Running 2-GPU micro-benchmark with torchrun"
torchrun --nproc-per-node 2 micro_benchmarking_pytorch.py --network resnet50 --batch-size 128
```

For PyTorch 2.0 compilation:

```
echo "Running with torch.compile max-autotune"
python micro_benchmarking_pytorch.py --network resnet50 --compile --compileContext "{'mode': 'max-autotune'}"
```

## Runtime Trace Profiling with get_trace.sh

This script captures GPU API calls, kernel launches, and memory operations for timeline analysis.

Run the profiling script:

```
echo "Collecting runtime trace with rocprofv3"
./get_trace.sh
```

The script will output results to `profiling_results/trace_<timestamp>/`. To analyze the results:

```
echo "Opening trace in Perfetto UI"
echo "Visit https://ui.perfetto.dev/ and open the .pftrace file"
```

Example output (ROCm 6.4):

```
Detected ROCm version: 6.4.4-129
Starting rocprofv3 runtime trace profiling for pytorch_microbench...
Output directory: profiling_results/trace_20260114_151142
Using ROCm 6.x/7.x: --output-format pftrace (generates Perfetto trace)

Collecting full runtime trace (HIP/HSA API calls, kernels, memory operations)

INFO: running forward and backward for warmup.
INFO: running the benchmark..
OK: finished running benchmark..
...
Profiling complete! Results saved to: profiling_results/trace_20260114_151142

Generated files:
total 25M
-rw-r--r-- 1 root root 25M Jan 14 15:11 5712_results.pftrace

Perfetto trace file found: profiling_results/trace_20260114_151142/.../5712_results.pftrace
Size: 25M

To view the trace:
  1. Visit: https://ui.perfetto.dev/
  2. Open: profiling_results/trace_20260114_151142/.../5712_results.pftrace
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

The script will output results to `profiling_results/counters_<timestamp>/`.

Example output (ROCm 6.4):

```
Detected ROCm version: 6.4.4-129
Starting rocprofv3 kernel trace collection for pytorch_microbench...
Output directory: profiling_results/counters_20260114_151213
...
Profiling complete! Results saved to: profiling_results/counters_20260114_151213

Generated files:
total 8.6M
-rw-r--r-- 1 root root 1.6K Jan 14 15:12 5864_agent_info.csv
-rw-r--r-- 1 root root 8.5M Jan 14 15:12 5864_kernel_trace.csv

To analyze results:
  Check profiling_results/counters_20260114_151213 for output files
```

ROCm 6.x outputs CSV files directly, while ROCm 7.x outputs SQLite databases. For ROCm 7.x database files, use rocpd tools:

```
echo "Exporting kernel statistics to CSV"
rocpd2csv -i <db_file> -o kernel_stats.csv
```

```
echo "Getting kernel summary"
rocpd summary -i <db_file> --region-categories KERNEL
```

Example kernel analysis (ResNet50, 10 iterations):

```
Total kernels: 21175
Unique kernels: 68
Total GPU time: 2080.62 ms

Kernel Name                                                     Count    Total(ms)      Avg(us)    %Time
--------------------------------------------------------------------------------------------------------
miopenSp3AsmConv_v30_3_1_gfx11_fp32_f2x3_stride1                  732      760.707     1039.217    36.6%
MIOpenBatchNormBwdSpatial                                         636      168.497      264.932     8.1%
void at::native::vectorized_elementwise_kernel<4, at::nati...     384      120.959      314.997     5.8%
void at::native::vectorized_elementwise_kernel<4, at::nati...     588       96.744      164.530     4.6%
Cijk_Alik_Bljk_SB_MT64x64x8_SN_1LDSB0_APM1_ABV0_ACED0_AF0E...    2304       88.475       38.401     4.3%
MIOpenBatchNormFwdTrainSpatial                                    480       73.505      153.136     3.5%
Cijk_Alik_Bljk_SB_MT16x16x16_SN_1LDSB0_APM1_ABV0_ACED0_AF0...     768       70.635       91.973     3.4%
miopenSp3AsmConv_v30_3_1_gfx11_fp32_f3x2_stride1                  108       48.377      447.933     2.3%
...
```

The top kernels show MIOpen convolutions (`miopenSp3AsmConv`) and batch normalization (`MIOpenBatchNorm`) dominate execution time, which is expected for ResNet50.

Documentation for rocpd tools: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-rocpd-output-format.html

## GPU Hardware Metrics with get_rocprof_compute.sh

This script collects detailed GPU performance metrics for hardware utilization analysis.

Run the profiling script:

```
echo "Collecting GPU hardware metrics with rocprof-compute"
./get_rocprof_compute.sh
```

The script will output results to `profiling_results/rocprof_compute_<timestamp>/`. To analyze the results:

```
echo "Generating performance analysis report"
rocprof-compute analyze -p <output_dir>/workloads/<workload_name>/rocprof --dispatch <N> -n microbench_dispatch
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

The script will output results to `profiling_results/rocprof_sys_<timestamp>/`. To analyze the results:

```
echo "Opening trace in Perfetto UI"
echo "Visit https://ui.perfetto.dev/ and open the .proto file"
```

Note: rocprof-sys may produce memory map dumps in some configurations. If profiling fails or produces excessive output, consider using rocprofv3 (get_trace.sh) instead.

## Performance Tuning

For optimal performance on specific hardware, tune MIOpen by setting the environment variable before running:

```
export MIOPEN_FIND_ENFORCE=3
python micro_benchmarking_pytorch.py --network resnet50
```

This writes to a local performance database. See [MIOpen documentation](https://rocm.github.io/MIOpen/doc/html/perfdatabase.html) for details.

## Additional Resources

- rocprofv3 documentation: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-rocprofv3.html
- rocpd output format: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-rocpd-output-format.html
- Perfetto UI: https://ui.perfetto.dev/
