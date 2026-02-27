# Model Selector Guide

## How to Choose a Model

The dashboard supports seven transformer models from five architecture families. Select a model from the dropdown menu in the generator section at the top of the page.

### Available Models

| Model | Family | Params | Layers × Heads | Key Feature |
|-------|--------|--------|----------------|-------------|
| **GPT-2 (124M)** | GPT-2 | 85M | 12 × 12 | The MI classic — start here |
| **GPT-2 Medium (355M)** | GPT-2 | 302M | 24 × 16 | Scale comparison within GPT-2 |
| **GPT-Neo 125M** | GPT-Neo | 85M | 12 × 12 | Local attention in alternating layers |
| **Pythia-160M** | Pythia | 85M | 12 × 12 | Rotary PE, parallel attn+MLP |
| **Pythia-410M** | Pythia | 302M | 24 × 16 | Larger Pythia for scale comparison |
| **OPT-125M** | OPT | 85M | 12 × 12 | ReLU activation (unique contrast) |
| **Qwen2.5-0.5B (494M)** | Qwen2 | 391M | 24 × 14 | Modern: RMSNorm, SiLU, rotary PE |

### Architecture Comparisons

These models were chosen to highlight specific architectural differences:

- **Positional encoding**: GPT-2, GPT-Neo, and OPT use absolute positions. Pythia and Qwen use rotary (RoPE). Comparing the same prompt across both types shows how PE affects attention patterns.
- **Activation function**: Most models use GELU, but OPT uses ReLU and Qwen uses SiLU. This affects MLP behavior.
- **Normalization**: GPT-2, Neo, Pythia, and OPT use LayerNorm. Qwen uses RMSNorm.
- **Attention scope**: GPT-Neo alternates between local (256-token window) and global attention, unlike all other models.

### What Happens When You Load a Model

1. The model is downloaded from HuggingFace (this may take a moment the first time)
2. The dashboard **auto-detects** the model's architecture family
3. Internal hooks are automatically configured to capture attention patterns, MLP activations, and other data
4. The layer and head dropdowns are populated based on the model's structure

### Tips for Choosing

- **Start with GPT-2**: It's small, fast, and the most widely studied. Most educational resources reference GPT-2.
- **Compare same-size models**: GPT-2, GPT-Neo 125M, Pythia-160M, and OPT-125M all have 12 layers and 12 heads — differences in their attention patterns come from architecture, not scale.
- **Compare scale**: GPT-2 vs GPT-2 Medium (or Pythia-160M vs Pythia-410M) shows how more layers and heads change behavior.
- **Try Qwen2.5 for a modern perspective**: It uses an entirely different design philosophy from GPT-2.
- **Memory matters**: All dropdown models are small enough for interactive exploration. Larger models can be entered manually but may be slow.

### Generation Settings

After selecting a model and entering a prompt, you can configure:
- **Number of Generation Choices (Beams)**: 1-5 beams. More beams explore more paths but take longer.
- **Number of New Tokens**: 1-20 tokens to generate. Shorter is faster.

Click **Analyze** to run the model and see results in the pipeline and generation sections.
