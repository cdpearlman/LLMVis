# GPT-2 Overview

## What Is GPT-2?

GPT-2 (Generative Pre-trained Transformer 2) is a language model created by OpenAI in 2019. It was one of the first models to demonstrate that scaling up transformers could produce impressively fluent text. It remains one of the most well-studied and accessible models for learning about transformer internals.

## Architecture Details

| Property | Value |
|----------|-------|
| Parameters | ~124 million (small variant) |
| Layers | 12 |
| Attention Heads | 12 per layer (144 total) |
| Hidden Dimension | 768 |
| MLP Dimension | 3072 (4x hidden) |
| Vocabulary Size | 50,257 tokens |
| Positional Encoding | Learned absolute positions |
| Max Sequence Length | 1024 tokens |
| Normalization | LayerNorm |
| Activation Function | GELU |

## Why Start with GPT-2?

GPT-2 small is the **recommended starting model** for learning with this dashboard:

- **Fast**: Small enough to load quickly and run interactively
- **Well-studied**: More research papers have analyzed GPT-2's internals than almost any other model. Many examples and references use GPT-2.
- **Clear patterns**: With 12 heads per layer, the attention patterns are easy to visualize and categorize
- **Manageable size**: 144 total attention heads is small enough to explore systematically

## What to Expect in the Dashboard

When analyzing GPT-2, you'll typically see:

- **Tokenization**: GPT-2 uses BPE (Byte-Pair Encoding). Common words are single tokens; rare words get split. Spaces are typically attached to the beginning of the following token.
- **Embeddings**: 768-dimensional vectors, which capture rich semantic information despite being relatively compact.
- **Attention patterns**: You'll see a good mix of head categories. Expect several Previous-Token heads (especially in early layers), some First/Positional heads, and a variety of other patterns.
- **Output**: GPT-2 can produce reasonably coherent text for simple prompts. For factual prompts, it sometimes produces outdated or incorrect facts (it was trained on data from before 2019).

## GPT-2 Variants

The dashboard includes GPT-2 Small and Medium in the dropdown. Larger variants can be loaded by typing the model ID:

- **GPT-2 Small** (124M params, 12 layers) -- in dropdown as "GPT-2 (124M)"
- **GPT-2 Medium** (355M params, 24 layers) -- in dropdown as "GPT-2 Medium (355M)"
- **GPT-2 Large** (774M params, 36 layers) -- enter `gpt2-large`
- **GPT-2 XL** (1.5B params, 48 layers) -- enter `gpt2-xl`

Comparing GPT-2 Small and Medium is a great way to see how attention heads specialize as models scale up within the same architecture.

## HuggingFace Model IDs

- `gpt2` or `openai-community/gpt2`
- `gpt2-medium` or `openai-community/gpt2-medium`
- `gpt2-large` or `openai-community/gpt2-large`
- `gpt2-xl` or `openai-community/gpt2-xl`
