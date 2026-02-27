# Qwen2.5 Overview

## What Is Qwen2.5?

Qwen2.5 is a family of language models developed by Alibaba's Qwen team, released in 2024. It uses a modern LLaMA-like architecture with several improvements over older models like GPT-2. The dashboard includes **Qwen2.5-0.5B** as a representative of this modern architecture.

## Architecture Details

| Property | Qwen2.5-0.5B |
|----------|-------------|
| Parameters | 494M |
| Layers | 24 |
| Attention Heads | 14 per layer |
| Hidden Dimension | 896 |
| Vocabulary Size | 151,936 |
| Positional Encoding | Rotary (RoPE) |
| Normalization | RMSNorm |
| Activation Function | SiLU |

## Key Architectural Differences from GPT-2

### RoPE (Rotary Position Embeddings)
Instead of GPT-2's learned absolute position embeddings, Qwen uses **Rotary Position Embeddings (RoPE)**. RoPE encodes position by rotating query and key vectors in attention. This means:
- Better generalization to different sequence lengths
- Position information baked into the attention computation
- Attention patterns may look different from GPT-2

### RMSNorm Instead of LayerNorm
Qwen uses **RMSNorm** (Root Mean Square Normalization) instead of standard LayerNorm. RMSNorm only normalizes vector magnitude without centering, making it simpler and slightly faster.

### SiLU Activation
Where GPT-2 uses GELU in the MLP, Qwen uses **SiLU** (Sigmoid Linear Unit, also called "Swish"), a smooth activation function common in modern architectures.

### Grouped-Query Attention (GQA)
Qwen uses **Grouped-Query Attention**, where multiple query heads share key and value heads. This reduces memory usage while maintaining quality. In the attention visualization, you may notice this affects how attention patterns are structured.

## Why Include Qwen2.5?

- **Modern architecture**: Represents the latest generation of LLM design (post-LLaMA innovations)
- **Contrast with GPT-2**: Every major architectural choice is different — PE type, normalization, activation, attention variant
- **Good size**: 494M parameters with 24 layers provides enough depth for interesting head specialization while staying fast
- **Open access**: No gated access or license agreement required (unlike LLaMA or Gemma)

## What to Expect in the Dashboard

When using Qwen2.5-0.5B:
- **More layers**: 24 layers vs GPT-2's 12 means more attention patterns to explore
- **Different tokenizer**: Qwen has a much larger vocabulary (152K vs 50K), so the same text tokenizes differently
- **Different attention patterns**: RoPE-based attention creates different positional patterns from GPT-2
- **Comparing with GPT-2**: Running the same prompt on both is the best way to see how architecture affects predictions and attention

## HuggingFace Model IDs

- `Qwen/Qwen2.5-0.5B` (in dropdown)
- `Qwen/Qwen2.5-1.5B`, `Qwen/Qwen2.5-3B` (larger, enter manually)
