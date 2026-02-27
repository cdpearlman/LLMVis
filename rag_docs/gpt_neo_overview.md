# GPT-Neo Overview

## What Is GPT-Neo?

GPT-Neo is a family of open-source language models created by EleutherAI in 2021. It was one of the first serious open-source alternatives to OpenAI's GPT-3, designed as an open replication of the GPT architecture. GPT-Neo is the predecessor to EleutherAI's later Pythia suite.

## Architecture Details

| Property | GPT-Neo 125M |
|----------|-------------|
| Parameters | 85M (actual) |
| Layers | 12 |
| Attention Heads | 12 per layer |
| Hidden Dimension | 768 |
| Vocabulary Size | 50,257 |
| Positional Encoding | Learned absolute |
| Normalization | LayerNorm |
| Activation Function | GELU |

## Key Architectural Feature: Local Attention

GPT-Neo's most distinctive feature is its **alternating local and global attention** pattern:
- **Even layers**: Use local attention with a window of 256 tokens (each token only attends to nearby tokens)
- **Odd layers**: Use standard global attention (each token can attend to all previous tokens)

This alternating pattern is a significant architectural difference from GPT-2 (which uses global attention in every layer) and creates interesting attention visualization patterns in the dashboard.

## Why GPT-Neo Is Useful for Learning

- **Same size as GPT-2**: At 125M parameters with 12 layers and 12 heads, it's directly comparable to GPT-2
- **Same positional encoding**: Uses learned absolute positions like GPT-2, so attention pattern differences are due to architecture, not PE
- **Different attention pattern**: The local attention in even layers creates visually distinct attention maps — great for understanding how attention scope affects behavior
- **Bridge to Pythia**: GPT-Neo → GPT-NeoX → Pythia is an evolutionary chain. Comparing GPT-Neo (absolute PE) with Pythia (rotary PE) shows the effect of positional encoding

## Comparing GPT-Neo and GPT-2

| Feature | GPT-2 (124M) | GPT-Neo 125M |
|---------|-------------|--------------|
| Layers × Heads | 12 × 12 | 12 × 12 |
| Hidden Dim | 768 | 768 |
| Attention | All global | Alternating local/global |
| PE Type | Learned absolute | Learned absolute |
| Training Data | WebText | The Pile |
| Creator | OpenAI | EleutherAI |

## HuggingFace Model IDs

- `EleutherAI/gpt-neo-125M` (in dropdown)
- `EleutherAI/gpt-neo-1.3B`, `EleutherAI/gpt-neo-2.7B` (larger, enter manually)
