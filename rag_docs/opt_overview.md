# OPT Overview

## What Is OPT?

OPT (Open Pre-trained Transformer) is a family of language models released by Meta in 2022. OPT was designed to replicate GPT-3's architecture and performance while being openly available to researchers. It uses a decoder-only transformer architecture similar to GPT-2 but with options for much larger sizes.

## Architecture Details

OPT's architecture is close to GPT-2 but has some differences:

| Property | OPT-125M | OPT-350M | OPT-1.3B |
|----------|----------|----------|----------|
| Parameters | 125M | 350M | 1.3B |
| Layers | 12 | 24 | 24 |
| Attention Heads | 12 | 16 | 32 |
| Hidden Dimension | 768 | 1024 | 2048 |
| Vocabulary Size | 50,272 | 50,272 | 50,272 |

### Key Differences from GPT-2

- **Learned positional embeddings**: Like GPT-2, OPT uses learned absolute position embeddings (unlike LLaMA's RoPE)
- **LayerNorm placement**: OPT uses pre-norm LayerNorm (applied before each sublayer), which is slightly different from GPT-2's original arrangement
- **Larger variants available**: OPT scales up to 175 billion parameters, though only smaller variants are practical for interactive use

### Similarities to GPT-2

- Same general decoder-only architecture
- Same tokenizer style (BPE with ~50K vocabulary)
- Same attention mechanism (standard multi-head self-attention)
- Similar training objective (next-token prediction)

## What to Expect in the Dashboard

When using OPT models:

- **OPT-125M is very similar to GPT-2**: Same number of layers (12), heads (12), and hidden dimension (768). You'll see similar attention patterns and predictions.
- **Different module paths**: The dashboard auto-detects OPT's internal structure (e.g., `model.decoder.layers.N.self_attn`), so hooking works automatically.
- **Tokenization**: OPT's tokenizer is very similar to GPT-2's, so the same text usually produces similar (but not identical) token sequences.
- **Good for comparison**: Running the same prompt on GPT-2 and OPT-125M can show how similar architectures with different training data produce different predictions.

## HuggingFace Model IDs

- `facebook/opt-125m`
- `facebook/opt-350m`
- `facebook/opt-1.3b`
- `facebook/opt-2.7b`

Note: OPT models are not in the default dropdown but can be loaded by typing the model ID directly.
