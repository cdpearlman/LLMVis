# LLaMA Overview

## What Is LLaMA?

LLaMA (Large Language Model Meta AI) is a family of open-weight language models developed by Meta. First released in 2023, LLaMA models introduced several architectural improvements over GPT-2 and became the foundation for many other models (Mistral, Qwen, etc.). In the dashboard, models labeled "LLaMA-like" share this architecture.

## Architectural Differences from GPT-2

LLaMA models use several key innovations:

### RoPE (Rotary Position Embeddings)
Instead of GPT-2's learned absolute position embeddings, LLaMA uses **Rotary Position Embeddings (RoPE)**. RoPE encodes position information by rotating the query and key vectors in attention. This means:
- The model can generalize better to different sequence lengths
- Position information is baked into the attention computation itself
- Attention patterns may look different from GPT-2 because of how positions are encoded

### RMSNorm Instead of LayerNorm
LLaMA uses **RMSNorm** (Root Mean Square Normalization) instead of the standard LayerNorm used in GPT-2. RMSNorm is simpler and slightly faster -- it only normalizes the magnitude of the vectors without centering them first.

### SiLU Activation
Where GPT-2 uses GELU activation in the MLP, LLaMA uses **SiLU** (Sigmoid Linear Unit, also called "Swish"). This is a smooth activation function that tends to produce slightly different MLP behavior.

### Grouped-Query Attention (GQA)
Larger LLaMA variants use **Grouped-Query Attention**, where multiple query heads share the same key and value heads. This reduces memory usage and speeds up inference without significantly hurting quality. This means the number of key/value heads may be smaller than the number of query heads.

## Models Using LLaMA Architecture

The dashboard's "llama_like" family includes:
- **Meta LLaMA**: LLaMA 2 (7B, 13B, 70B), LLaMA 3 (1B, 3B, 8B, 70B)
- **Qwen**: Qwen2, Qwen2.5 (0.5B to 72B) -- available in the dashboard dropdown as "Qwen2.5-0.5B"
- **Mistral**: Mistral-7B, Mixtral-8x7B

## What to Expect in the Dashboard

When using a LLaMA-like model (such as Qwen2.5-0.5B):

- **More layers and heads**: Even the small Qwen2.5-0.5B has 24 layers and 14 heads, compared to GPT-2's 12 layers and 12 heads
- **Different attention patterns**: RoPE-based attention may show different positional patterns compared to GPT-2
- **Different tokenizer**: LLaMA-family models use a different BPE vocabulary, so the same text may tokenize differently
- **Comparing with GPT-2**: Running the same prompt on both GPT-2 and a LLaMA-like model is a great way to see how architecture affects predictions

## HuggingFace Model IDs

- `Qwen/Qwen2.5-0.5B` (in default dropdown)
- `meta-llama/Llama-3.2-1B`, `meta-llama/Llama-3.1-8B`
- `mistralai/Mistral-7B-v0.3`
