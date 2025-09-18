# Transformer Activation Capture and Visualization

This project provides tools for capturing activations from transformer models and visualizing attention patterns using bertviz.

## Overview

The project consists of two main components:
1. **Activation Capture** (`agnostic_capture.py`) - Captures activations from any transformer model
2. **Attention Visualization** (`bertviz_head_model_view.py`) - Creates interactive attention visualizations

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
