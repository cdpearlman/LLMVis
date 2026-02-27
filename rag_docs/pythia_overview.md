# Pythia Overview

## What Is Pythia?

Pythia is a suite of language models created by EleutherAI specifically to support interpretability research. Released in 2023, the Pythia models were trained on the deduplicated Pile dataset with full public training data and checkpoints, making them uniquely transparent.

## Architecture Details

| Property | Pythia-160M | Pythia-410M |
|----------|-------------|-------------|
| Parameters | 85M (actual) | 302M (actual) |
| Layers | 12 | 24 |
| Attention Heads | 12 per layer | 16 per layer |
| Hidden Dimension | 768 | 1024 |
| Vocabulary Size | 50,304 | 50,304 |
| Positional Encoding | Rotary (RoPE) |  Rotary (RoPE) |
| Normalization | LayerNorm | LayerNorm |
| Activation Function | GELU | GELU |
| Parallel Attn+MLP | Yes | Yes |

## Key Architectural Features

### Rotary Position Embeddings (RoPE)
Unlike GPT-2's learned absolute positions, Pythia uses **Rotary Position Embeddings**. RoPE encodes position by rotating the query and key vectors, which means:
- Position information is baked into the attention computation
- The model can potentially generalize to longer sequences
- Attention patterns may look different from GPT-2

### Parallel Attention + MLP
Pythia computes attention and MLP in **parallel** within each layer (rather than sequentially). This is a design choice that slightly changes how information flows through the network.

### GPT-NeoX Architecture
Pythia uses the GPT-NeoX architecture (`GPTNeoXForCausalLM`), which is EleutherAI's evolution of GPT-Neo. The module paths use `gpt_neox.layers.{N}.attention` instead of GPT-2's `transformer.h.{N}.attn`.

## Why Pythia Matters for Interpretability

- **Open training data**: The Pile dataset is fully public, so you can trace what the model learned
- **Training checkpoints**: Hundreds of checkpoints are available, letting researchers study how features develop during training
- **Designed for research**: EleutherAI specifically built Pythia to be a good subject for mechanistic interpretability
- **Comparison with GPT-Neo**: Same creator but different architecture (rotary vs absolute PE), enabling clean ablation studies of positional encoding

## Comparing Pythia and GPT-2

Running the same prompt on both is highly educational:
- **Same size, different PE**: Pythia-160M and GPT-2 both have 12 layers and 12 heads, but use different positional encodings
- **Parallel vs sequential**: Pythia processes attention and MLP in parallel; GPT-2 does them sequentially
- **Different training data**: GPT-2 was trained on WebText; Pythia on The Pile

## HuggingFace Model IDs

- `EleutherAI/pythia-160m` (in dropdown)
- `EleutherAI/pythia-410m` (in dropdown)
- `EleutherAI/pythia-1b`, `EleutherAI/pythia-1.4b` (larger, enter manually)
