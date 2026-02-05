# Model Selector Guide

## How to Choose a Model

The dashboard supports several transformer model families. You can select a model from the dropdown menu in the generator section at the top of the page.

### Available Models

Currently, the dashboard offers:

- **GPT-2 (124M)**: OpenAI's GPT-2 small model. 12 layers, 12 attention heads, 768-dimensional embeddings. This is the best model to start with -- it's small, fast, and well-studied.
- **Qwen2.5-0.5B**: A LLaMA-like model from Alibaba's Qwen family. 24 layers, 14 attention heads, 896-dimensional embeddings. Slightly larger and uses different architectural features (RoPE, SiLU activation).

You can also enter a custom **HuggingFace model ID** in the dropdown (type it in). The dashboard supports GPT-2, LLaMA, OPT, GPT-NeoX, BLOOM, Falcon, and MPT model families.

### What Happens When You Load a Model

1. The model is downloaded from HuggingFace (this may take a moment the first time)
2. The dashboard **auto-detects** the model's architecture family
3. Internal hooks are automatically configured to capture attention patterns, MLP activations, and other data
4. The layer and head dropdowns in the sidebar and ablation panel are populated based on the model's structure

### Auto-Detection

The dashboard has a registry that maps model names to their architecture family. When it recognizes a model, it automatically configures:
- Which internal modules to hook for attention capture
- Which normalization parameters to track
- The correct patterns for extracting layer outputs

If you enter an unknown model, the sidebar's configuration dropdowns may need manual adjustment.

### Tips for Choosing

- **Start with GPT-2**: It's small, fast, and the most widely studied. Most educational resources reference GPT-2.
- **Try Qwen2.5-0.5B for comparison**: It uses a different architecture (LLaMA-style). Comparing results between GPT-2 and Qwen can highlight how architectural differences affect attention patterns.
- **Larger models are slower**: Models with more parameters take longer to load and analyze. Stick to small models for interactive exploration.
- **Memory matters**: Larger models require more RAM. If the dashboard becomes unresponsive, try a smaller model.

### Generation Settings

After selecting a model and entering a prompt, you can configure:
- **Number of Generation Choices (Beams)**: 1-5 beams. More beams explore more paths but take longer.
- **Number of New Tokens**: 1-20 tokens to generate. Shorter is faster.

Click **Analyze** to run the model and see results in the pipeline and generation sections.
