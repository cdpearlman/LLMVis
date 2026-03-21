"""
Model selector component with model dropdown and prompt input.

This component provides the interface for:
- Selecting transformer models from available options
- Entering text prompts for analysis
"""

from dash import html, dcc

# Example prompts for guiding users toward meaningful experiments
EXAMPLE_PROMPTS = [
    {
        "label": "Track indirect objects",
        "prompt": "The chef gave the waiter a generous tip because",
        "tooltip": "Tests whether the model tracks who gave and who received across clauses"
    },
    {
        "label": "Resolve ambiguity",
        "prompt": "The bat flew over the",
        "tooltip": "Tests how earlier context shapes predictions when a word has multiple meanings (bat)"
    },
    {
        "label": "Understand repetition",
        "prompt": "The cat sat on the mat. The cat sat on the",
        "tooltip": "Activates duplicate-token and induction detectors that complete repeated sequences"
    },
    {
        "label": "Link pronouns",
        "prompt": "The nurse said that she",
        "tooltip": "Tests coreference resolution and reveals gender bias in attention detectors"
    }
]

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
            html.Div("Not sure what to prompt? See how models:", className="example-prompts-label"),
            html.Div([
                html.Button(
                    p["label"],
                    id={"type": "example-prompt-btn", "index": i},
                    className="example-prompt-chip",
                    title=p["tooltip"],
                    n_clicks=0
                )
                for i, p in enumerate(EXAMPLE_PROMPTS)
            ], className="example-prompts-container"),
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
