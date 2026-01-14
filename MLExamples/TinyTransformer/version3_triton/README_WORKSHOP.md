# Version 3: Triton Kernel Integration - Workshop Edition

README_WORKSHOP.md from `HPCTrainingExamples/MLExamples/TinyTransformer/version3_triton` in the Training Examples repository.

## Quick Start

```
cd version3_triton/
python tiny_llama_v3.py --batch-size 8 --seq-len 128 --num-steps 20
```

Expected output: Loss ~7.0, Speed ~2065 samples/sec, Memory ~282 MB

## Performance Results (AMD MI325X, ROCm 6.4.4)

| Metric | V1 Baseline | V3 Optimized | Improvement |
|--------|-------------|--------------|-------------|
| Training Speed | 372.9 samples/sec | 2065.0 samples/sec | 5.5x faster |
| Memory Usage | 522.3 MB | 281.8 MB | 46% reduction |

## Optimizations Applied

### 1. Flash Attention (Triton Kernel)
Memory-efficient attention using online softmax. Reduces memory from O(S²) to O(S).

### 2. RMSNorm (Triton Kernel)
Fused variance computation + normalization (3 kernels → 1).

### 3. Hybrid SwiGLU Strategy
Use PyTorch/rocBLAS for matrix multiplies, PyTorch for activation. Custom Triton kernel was 2.4x slower.

### 4. Tensor Contiguity
Always `.contiguous()` before Triton kernels. Non-contiguous tensors caused 20x slowdown.

### 5. Weight Initialization
Proper initialization (std=0.02) prevents exploding loss.

## Key Learnings

1. **Correctness First**: Validate before optimizing
2. **Memory Layout Matters**: Non-contiguous tensors kill performance
3. **Hybrid Wins**: Use best tool for each operation
4. **Measure Accurately**: Always `torch.cuda.synchronize()` for timing
5. **Iterate**: Fix one issue at a time, re-measure

## Performance Debugging Exercise

```
cd exercises/performance_debugging/
./run_all_stages.sh
```

Shows the complete optimization journey through 5 stages:
- Stage 1: Broken (loss=942) - missing weight init
- Stage 2: Slow (15 samp/s) - non-contiguous tensors
- Stage 3: Better (311 samp/s) - added .contiguous()
- Stage 4: Same (306 samp/s) - accurate timing revealed issue
- Stage 5: Optimal (2065 samp/s) - hybrid kernel strategy

## Common Issues

**ImportError: No module named 'triton'**
```
pip install triton
```

**Performance slower than expected**
- Ensure tensors are contiguous
- Use CUDA synchronization for accurate timing
- Use hybrid SwiGLU (not custom Triton matmul)

## Additional Resources

- Triton Documentation: https://triton-lang.org/
- Flash Attention Paper: https://arxiv.org/abs/2205.14135
- Performance Debugging Guide: exercises/performance_debugging/README.md
