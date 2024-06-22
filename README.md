# Unified MoE Compression

This is the official implementation of the [paper](https://arxiv.org/abs/2406.02500):  

```
Demystifying the Compression of Mixture-of-Experts Through a Unified Framework
Shwai He*, Daize Dong*, Liang Ding, Ang Li
```

## Installation

```bash
conda create -n moe_compression python=3.10
conda activate moe_compression
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Running Compression

### Expert Slimming

#### Pruning

```bash
bash scripts/compress/pruning/mixtral_prune.sh
bash scripts/compress/pruning/deepseek_prune.sh
bash scripts/compress/pruning/deepseek_prune_noshared.sh
```

#### Quantization

```bash
bash scripts/compress/quantization/awq.sh
bash scripts/compress/quantization/gptq.sh
```

### Expert Trimming

#### Expert Drop

For normal `bfloat16` or `float32` models, run:

```bash
bash scripts/compress/expert_drop/mixtral_expert_drop.sh
bash scripts/compress/expert_drop/deepseek_expert_drop.sh
```

For quantized models, run:

```bash
bash scripts/compress/expert_drop/mixtral_expert_drop_quantized.sh
bash scripts/compress/expert_drop/deepseek_expert_drop_quantized.sh
```

#### Layer Drop

```bash
bash scripts/compress/layer_drop/mixtral_layer_drop.sh
bash scripts/compress/layer_drop/deepseek_layer_drop.sh
bash scripts/compress/layer_drop/mixtral_layer_drop_quantized.sh
bash scripts/compress/layer_drop/deepseek_layer_drop_quantized.sh
```

#### Block Drop

```bash
bash scripts/compress/block_drop/mixtral_block_drop.sh
bash scripts/compress/block_drop/deepseek_block_drop.sh
bash scripts/compress/block_drop/mixtral_block_drop_quantized.sh
bash scripts/compress/block_drop/deepseek_block_drop_quantized.sh
```

## Measuring Speedup

```bash
bash scripts/benchmark/benchmark_speedup.sh
bash scripts/benchmark/measure_flops.sh
```

## Evaluation

### Loss & PPL

```bash
bash scripts/evaluate/mixtral_evaluate.sh
bash scripts/evaluate/deepseek_evaluate.sh
```

### Benchmarks

Please refer to [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
