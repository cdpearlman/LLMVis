"""
Model selector component with model dropdown and prompt input.

This component provides the interface for:
- Selecting transformer models from available options
- Entering text prompts for analysis
"""

from dash import html, dcc

# Available models organized by family
AVAILABLE_MODELS = [
    # LLaMA-like models (Qwen)
    {"label": "Qwen2.5-0.5B", "value": "Qwen/Qwen2.5-0.5B"},
    # {"label": "Qwen2.5-1.5B", "value": "Qwen/Qwen2.5-1.5B"},
    
    # GPT-2 family
    {"label": "GPT-2 (124M)", "value": "gpt2"}
    # {"label": "GPT-2 Medium (355M)", "value": "gpt2-medium"},
    # {"label": "GPT-2 Large (774M)", "value": "gpt2-large"},
    
    # # OPT family
    # {"label": "OPT-125M", "value": "facebook/opt-125m"},
    # {"label": "OPT-350M", "value": "facebook/opt-350m"},
    
    # # GPT-NeoX family (Pythia)
    # {"label": "Pythia-70M", "value": "EleutherAI/pythia-70m"},
    # {"label": "Pythia-160M", "value": "EleutherAI/pythia-160m"},
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
                style={"minWidth": "300px"}  # Ensure dropdown is wide enough
            )
        ], className="input-container"),
        
        # Prompt input
        html.Div([
            html.Div([
                html.Label("Enter Prompt:", className="input-label", style={"display": "inline-block", "marginRight": "10px"}),
                html.Button(
                    [html.I(className="fas fa-plus", style={"marginRight": "5px"}), "Compare"],
                    id="compare-prompts-btn",
                    className="compare-button",
                    title="Add a second prompt for comparison"
                )
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "0.5rem"}),
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
        
        # Second prompt input (initially hidden)
        html.Div([
            html.Label("Second Prompt (for comparison):", className="input-label"),
            dcc.Textarea(
                id='prompt-input-2',
                placeholder="Enter second text prompt for comparison...",
                value="",
                style={
                    "width": "100%", 
                    "height": "100px",
                    "resize": "vertical"
                },
                className="prompt-input"
            )
        ], id="second-prompt-container", className="input-container", style={"display": "none"}),
        
        # Status indicator
        html.Div([
            html.Div(id="model-status", className="status-indicator")
        ], className="status-container")
        
    ], className="model-selector-content")
