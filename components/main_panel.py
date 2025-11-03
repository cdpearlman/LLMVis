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
        ], className="visualization-section")
        
    ], className="main-panel-content")
