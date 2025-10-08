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
        
        # Analysis Results section (layer-specific analysis on node click)
        html.Div([
            html.H3("Analysis Results", className="section-title"),
            html.Div([
                html.P(
                    "Click a layer node to see detailed attention analysis and head categorization.",
                    className="placeholder-text"
                )
            ], id="results-container", className="results-area")
        ], className="results-section")
        
    ], className="main-panel-content")
