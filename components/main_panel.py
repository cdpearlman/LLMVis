"""
Main panel component for the dashboard.

This component contains:
- Model selector and prompt input
- Layer accordion visualization
- Status and results display
"""

from dash import html, dcc
from .model_selector import create_model_selector

def create_main_panel():
    """Create the main content panel."""
    return html.Div([
        # Model selection section
        html.Div([
            html.H3("Model Configuration", className="section-title"),
            create_model_selector()
        ], className="config-section"),
        
        # Analysis loading indicator
        html.Div(id="analysis-loading-indicator", className="loading-container"),
        
        # Check Token input with probability graph
        html.Div([
            html.Div([
                html.Label("Check Token (optional):", className="input-label"),
                dcc.Input(
                    id='check-token-input',
                    type='text',
                    placeholder="Enter a token to track its probability...",
                    value="",
                    style={"width": "300px"},
                    className="prompt-input"
                )
            ], style={"flex": "0 0 auto"}),
            html.Div([
                dcc.Graph(
                    id='check-token-graph',
                    figure={},
                    style={'height': '450px', 'width': '100%'},
                    config={'displayModeBar': False}
                )
            ], id='check-token-graph-container', style={'flex': '1', 'minWidth': '300px', 'display': 'none'})
        ], className="input-container", style={"marginBottom": "1.5rem", "display": "flex", "gap": "1.5rem", "alignItems": "flex-start"}),
        
        # Layer-based visualization section with loading spinner
        html.Div([
            html.H3("Layer-by-Layer Predictions", className="section-title"),
            dcc.Loading(
                id="layer-accordions-loading",
                type="default",
                children=html.Div(id='layer-accordions-container', className="layer-accordions"),
                overlay_style={"visibility":"visible", "opacity": .7, "backgroundColor": "white"},
                custom_spinner=html.Div([
                    html.I(className="fas fa-spinner fa-spin", style={'fontSize': '24px', 'color': '#667eea', 'marginRight': '10px'}),
                    html.Span("Loading visuals...", style={'fontSize': '16px', 'color': '#495057'})
                ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'padding': '2rem'})
            )
        ], className="visualization-section"),
        
        # Two-Prompt Comparison section (shown when comparing)
        html.Div([
            html.H3("Two-Prompt Comparison Analysis", className="section-title"),
            html.Div([
                html.P(
                    "Comparison analysis will appear here when two prompts are provided.",
                    className="placeholder-text"
                )
            ], id="comparison-container", className="results-area")
        ], id="comparison-section", className="results-section", style={'display': 'none'}),
        
        # Experiments section (shown after analysis)
        html.Div([
            html.Details([
                html.Summary("Experiments", className="experiments-summary"),
                html.Div([
                    # Ablation experiment subsection
                    html.Div([
                        html.H4("Ablation Study", style={'marginTop': '0.5rem', 'marginBottom': '0.5rem'}),
                        html.P([
                            "Ablation experiments test the influence of individual layers by replacing their activations ",
                            "with the mean activation across all tokens. This reveals how much each layer contributes ",
                            "to the model's final prediction for the given prompt(s)."
                        ], style={'fontSize': '14px', 'color': '#6c757d', 'marginBottom': '1rem'}),
                        html.P("Select a layer to ablate:", style={'fontSize': '14px', 'fontWeight': '500', 'marginBottom': '0.5rem'}),
                        html.Div(id='ablation-layer-buttons', className='ablation-buttons-grid'),
                        html.Button("Run Experiment", id='run-ablation-btn', className='run-experiment-btn', style={'marginTop': '1rem'})
                    ], className='ablation-section')
                ], className='experiments-content')
            ], open=True)
        ], id='experiments-section', className='experiments-section', style={'display': 'none'})
        
    ], className="main-panel-content")
