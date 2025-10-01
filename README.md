# Transformer Activation Capture and Visualization

This project provides tools for capturing activations from transformer models and visualizing attention patterns using bertviz and an interactive Dash web application.

## Overview

The project consists of multiple components:
1. **Interactive Dashboard** (`app.py`) - Web-based visualization with automatic model family detection
2. **Model Configuration** (`utils/model_config.py`) - Hard-coded mappings for common model families
3. **Activation Capture** (`utils/model_patterns.py`) - PyVene-based activation capture utilities
4. **Legacy Tools** (`agnostic_capture.py`, `bertviz_head_model_view.py`) - Command-line tools

## New Feature: Automatic Model Family Detection

The dashboard now automatically detects model families and pre-fills dropdown selections with appropriate modules and parameters. This eliminates manual selection for common architectures.

### Supported Model Families

- **LLaMA-like**: LLaMA 2/3, Mistral, Mixtral, Qwen2/2.5
- **GPT-2**: GPT-2, GPT-2 Medium/Large/XL
- **OPT**: Facebook OPT models (125M - 30B)
- **GPT-NeoX**: EleutherAI Pythia, GPT-NeoX-20B
- **BLOOM**: BigScience BLOOM models
- **Falcon**: TII Falcon models
- **MPT**: MosaicML MPT models

### How It Works

1. Select a model from the dropdown
2. The app detects the model family (e.g., "gpt2", "llama_like")
3. Dropdowns auto-fill with family-specific patterns:
   - **Attention modules**: e.g., `transformer.h.{N}.attn` for GPT-2
   - **MLP modules**: e.g., `model.layers.{N}.mlp` for LLaMA
   - **Normalization parameters**: e.g., `model.norm.weight` for LLaMA
   - **Logit lens parameter**: e.g., `lm_head.weight`
4. You can still manually adjust selections if needed

### Adding New Models

Edit `utils/model_config.py` and add entries to `MODEL_TO_FAMILY`:

```python
MODEL_TO_FAMILY = {
    "your-org/your-model": "llama_like",  # or gpt2, opt, etc.
    # ...
}
```

No code changes needed if the model follows an existing family's architecture!

## Files

### `agnostic_capture.py`
A model-agnostic activation capture tool that hooks into transformer modules and saves their outputs.

**Key Features:**
- Automatically categorizes modules into attention, MLP, and other types
- Interactive module selection by pattern
- Supports both PyTorch hooks and PyVene integration
- Saves data in organized JSON structure for easy retrieval

**Usage:**
```bash
# Basic usage
python agnostic_capture.py --model "Qwen/Qwen2.5-0.5B" --prompt "Once upon a time"

# Capture attention weights for bertviz
python agnostic_capture.py --model "Qwen/Qwen2.5-0.5B" --prompt "Once upon a time" --output my_activations.json

# Auto-select patterns (attention:0, mlp:0 selects first pattern from each)
python agnostic_capture.py --auto-select "attn:0;mlp:0;other:" --model "Qwen/Qwen2.5-0.5B"
```

**Interactive Selection:**
When run without `--auto-select`, the script will:
1. Show available module patterns grouped by type
2. Allow you to select patterns by index, name, or suffix
3. Selected patterns apply to ALL layers that contain them

### `bertviz_head_model_view.py`
Creates interactive HTML visualizations of attention patterns using the bertviz library.

**Features:**
- Generates head view (attention patterns per head)
- Generates model view (attention patterns across layers)
- Automatically extracts attention weights from captured data
- Saves HTML files that can be opened in any browser

**Usage:**
```bash
python bertviz_head_model_view.py
```

**Output:**
- `bertviz/attention_head_view_{model_name}.html` - Head-level attention patterns
- `bertviz/attention_model_view_{model_name}.html` - Model-level attention patterns

## Data Structure

The captured data is organized in the following JSON structure:

```json
{
  "model": "model_name",
  "prompt": "input_text",
  "input_ids": [[token_ids]],
  "selected_patterns": {
    "attention": ["pattern1", "pattern2"],
    "mlp": ["pattern1"],
    "other": []
  },
  "selected_modules": {
    "attention": ["model.layers.0.self_attn", "model.layers.1.self_attn", ...],
    "mlp": ["model.layers.0.mlp", "model.layers.1.mlp", ...],
    "other": []
  },
  "captured": {
    "attention_outputs": {
      "model.layers.0.self_attn": {
        "output": [
          [[...]], // Attention output (processed values)
          [[...]]  // Attention weights (used by bertviz)
        ]
      }
    },
    "mlp_outputs": { ... },
    "other_outputs": { ... }
  }
}
```

## Workflow

1. **Capture Activations:**
   ```bash
   python agnostic_capture.py --model "Qwen/Qwen2.5-0.5B" --prompt "Your text here"
   ```
   - Select attention patterns (e.g., `model.{layer}.self_attn`)
   - This creates `agnostic_activations.json`

2. **Generate Visualizations:**
   ```bash
   python bertviz_head_model_view.py
   ```
   - Reads from `agnostic_activations.json`
   - Creates HTML visualization files in `bertviz/` directory

3. **View Results:**
   - Open the generated HTML files in your browser
   - Explore attention patterns across heads and layers

## Requirements

```bash
pip install torch transformers bertviz
```

Optional:
```bash
pip install pyvene  # For enhanced hooking capabilities
```

## Important Notes

### For Attention Visualization:
- **Must capture `self_attn` modules** (not `self_attn.o_proj`) for bertviz to work
- Attention modules return tuples: `(output, attention_weights)`
- bertviz uses the attention weights (element 1) for visualization

### Module Selection:
- Patterns use `{layer}` placeholder (e.g., `model.{layer}.self_attn`)
- Selected patterns apply to ALL layers automatically
- Use indices, exact names, or unique suffixes for selection

### File Outputs:
- `agnostic_activations.json` - Captured activation data
- `bertviz/attention_head_view_{model}.html` - Per-head attention visualization
- `bertviz/attention_model_view_{model}.html` - Cross-layer attention visualization

## Troubleshooting

**"Attention tensor does not have correct dimensions"**
- Ensure you captured `self_attn` modules, not output projections
- Check that attention weights have shape `(batch, heads, seq_len, seq_len)`

**"Module not found"**
- Verify module patterns match your model architecture
- Use the interactive selection to see available patterns

**"No data captured"**
- Check hook registration succeeded
- Ensure selected modules exist in the model
- Verify the model actually runs forward pass

## Example Session

```bash
# 1. Capture attention data
python agnostic_capture.py --model "Qwen/Qwen2.5-0.5B" --prompt "The cat sat on the mat"
# Select attention patterns: 0 (for model.{layer}.self_attn)
# Select MLP patterns: (press enter to skip)
# Select other patterns: (press enter to skip)

# 2. Generate visualizations
python bertviz_head_model_view.py

# 3. Open bertviz/attention_head_view_Qwen_Qwen2.5-0.5B.html in browser
```

This will show you how the model attends to different tokens when processing "The cat sat on the mat".
