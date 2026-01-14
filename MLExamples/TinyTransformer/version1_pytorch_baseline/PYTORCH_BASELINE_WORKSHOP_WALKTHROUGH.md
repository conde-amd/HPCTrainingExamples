# Tiny LLaMA PyTorch Baseline - Workshop Walkthrough

PYTORCH_BASELINE_WORKSHOP_WALKTHROUGH.md from `HPCTrainingExamples/MLExamples/TinyTransformer/version1_pytorch_baseline` in the Training Examples repository.

This walkthrough demonstrates profiling techniques for transformer training workloads using Tiny LLaMA V1 as the baseline model.

## Prerequisites

- ROCm installation with rocprofv3
- PyTorch with ROCm support
- DeepSpeed (optional, for FLOPS profiling)

## Environment Verification

Check ROCm installation:

```
rocminfo | grep "Name:"
```

Check GPU status:

```
rocm-smi
```

Verify PyTorch with ROCm:

```
python3 -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
"
```

## Model Overview

Tiny LLaMA is a scaled-down transformer decoder with configurable parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| hidden_dim | 256 | Model dimension |
| n_layers | 4 | Transformer layers |
| n_heads | 8 | Attention heads |
| intermediate_dim | 512 | FFN intermediate dimension |
| vocab_size | 1000 | Vocabulary size |

Default model size: ~2.9M parameters (~11 MB FP32)

## Running the Baseline

Quick validation:

```
python3 tiny_llama_v1.py --batch-size 4 --seq-len 64 --num-steps 5
```

Standard training run:

```
python3 tiny_llama_v1.py --batch-size 8 --seq-len 128 --num-steps 20
```

Expected output:

```
==========================================
Tiny LLaMA V1 - PyTorch Baseline
==========================================
Configuration:
  Batch Size: 8
  Sequence Length: 128
  Number of Steps: 20
  ...

Starting training...
Step 1/20: Loss = 6.9088, Time = 0.234 seconds
Step 2/20: Loss = 6.9076, Time = 0.046 seconds
...
Step 20/20: Loss = 6.8821, Time = 0.044 seconds

==========================================
Performance Summary:
==========================================
Average time per step: 0.045 seconds
Training speed: 177.8 samples/sec
Peak memory usage: 2847 MB
==========================================
```

## Profiling with PyTorch Profiler

Enable PyTorch profiler for detailed operator-level analysis:

```
python3 tiny_llama_v1.py \
    --batch-size 8 \
    --seq-len 128 \
    --num-steps 20 \
    --enable-pytorch-profiler \
    --profile-dir ./pytorch_profiles \
    --profile-steps 5
```

View results with TensorBoard:

```
tensorboard --logdir ./pytorch_profiles --port 6006
```

## Memory Analysis

Test memory scaling with different batch sizes:

```
python3 tiny_llama_v1.py --batch-size 4 --seq-len 128 --num-steps 15
python3 tiny_llama_v1.py --batch-size 8 --seq-len 128 --num-steps 15
python3 tiny_llama_v1.py --batch-size 16 --seq-len 128 --num-steps 15
```

Test sequence length scaling:

```
python3 tiny_llama_v1.py --batch-size 8 --seq-len 64 --num-steps 10
python3 tiny_llama_v1.py --batch-size 8 --seq-len 128 --num-steps 10
python3 tiny_llama_v1.py --batch-size 8 --seq-len 256 --num-steps 10
```

Memory scales linearly with batch size and quadratically with sequence length (due to attention matrices).

## Performance Study

Use the performance study launcher for pre-configured problem sizes:

```
./launch_performance_study.sh tiny
./launch_performance_study.sh small
./launch_performance_study.sh medium --enable-profilers
```

Available problem sizes:

| Size | Hidden Dim | Layers | Seq Len | Batch | Est. Parameters |
|------|-----------|--------|---------|-------|-----------------|
| tiny | 256 | 4 | 128 | 8 | ~2.9M |
| small | 512 | 8 | 256 | 8 | ~20.9M |
| medium | 1024 | 12 | 512 | 16 | ~167M |
| large | 2048 | 16 | 1024 | 8 | ~1.3B |

## Key Performance Metrics

- **Training Speed**: samples/sec processed
- **FLOPS**: Floating point operations per second
- **MFU**: Model FLOPS Utilization (% of theoretical peak)
- **Memory Usage**: Peak GPU memory consumed

Baseline performance characteristics:
- Training speed: 50-200 samples/sec (varies by hardware)
- GPU utilization: 60-75% (typical for baseline PyTorch)
- Attention operations: ~35-45% of compute time
- FFN operations: ~30-40% of compute time

## Optimization Opportunities

Based on profiling analysis, the baseline model shows opportunities for:

1. **Kernel Fusion**: Combine separate QKV projections into single GEMM
2. **Flash Attention**: Reduce attention memory from O(S^2) to O(S)
3. **SwiGLU Fusion**: Combine gate and up projections
4. **Mixed Precision**: FP16/BF16 for 2x memory reduction

## Troubleshooting

CUDA/ROCm memory errors:

```
python3 tiny_llama_v1.py --batch-size 4 --seq-len 64 --num-steps 10
```

Check GPU utilization:

```
rocm-smi
```

Memory fragmentation:

```
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## Additional Resources

- [PyTorch Profiler Documentation](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [DeepSpeed FLOPS Profiler](https://www.deepspeed.ai/tutorials/flops-profiler/)
