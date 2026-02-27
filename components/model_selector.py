"""
Model selector component with model dropdown and prompt input.

This component provides the interface for:
- Selecting transformer models from available options
- Entering text prompts for analysis
"""

from dash import html, dcc

# Available models organized by family
AVAILABLE_MODELS = [
    # GPT-2 family (OpenAI) — absolute positional encoding, LayerNorm, GELU
    {"label": "GPT-2 (124M)", "value": "gpt2"},
    {"label": "GPT-2 Medium (355M)", "value": "gpt2-medium"},

    # GPT-Neo (EleutherAI) — absolute PE, LayerNorm, GELU
    {"label": "GPT-Neo 125M", "value": "EleutherAI/gpt-neo-125M"},

    # Pythia (EleutherAI) — rotary PE, LayerNorm, GELU, parallel attn+MLP
    {"label": "Pythia-160M", "value": "EleutherAI/pythia-160m"},
    {"label": "Pythia-410M", "value": "EleutherAI/pythia-410m"},

    # OPT (Meta) — absolute PE, LayerNorm, ReLU activation
    {"label": "OPT-125M", "value": "facebook/opt-125m"},

    # Qwen2.5 (Alibaba) — rotary PE, RMSNorm, SiLU activation
    {"label": "Qwen2.5-0.5B (494M)", "value": "Qwen/Qwen2.5-0.5B"},
]

def create_model_selector():
    """Create the model selection and prompt input interface."""
    return html.Div([
        # Model selection
        html.Div([
            html.Label("Select Model:", className="input-label"),
            dcc.Dropdown(
                id='model-dropdown',
                options=AVAILABLE_MODELS,
                value=None,
                placeholder="Choose a transformer model...",
                className="model-dropdown",
                style={"minWidth": "300px"}
            )
        ], className="input-container"),
        
        # Prompt input
        html.Div([
            html.Label("Enter Prompt:", className="input-label"),
            dcc.Textarea(
                id='prompt-input',
                placeholder="Enter text prompt for analysis...",
                value="",
                style={
                    "width": "100%", 
                    "height": "100px",
                    "resize": "vertical"
                },
                className="prompt-input"
            )
        ], className="input-container"),
        
        # Status indicator
        html.Div([
            html.Div(id="model-status", className="status-indicator")
        ], className="status-container")
        
    ], className="model-selector-content")
