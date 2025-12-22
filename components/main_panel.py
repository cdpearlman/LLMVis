"""
Main panel component for the dashboard.

This component contains:
- Generator Interface (Model, Prompt, Settings)
- Result List (for beam search)
- Sequence Analyzer (Scrubber, Visualizations)
"""

from dash import html, dcc
from .model_selector import create_model_selector
from .tokenization_panel import create_tokenization_panel

def create_main_panel():
    """Create the main content panel."""
    return html.Div([
        # 1. Generator Interface
        html.Div([
            html.H3("Generator Interface", className="section-title"),
            create_model_selector(),
            
            # Generation Settings
            html.Div([
                html.H4("Generation Settings", style={'fontSize': '14px', 'marginTop': '15px', 'marginBottom': '10px'}),
                
                # Sliders Row
                html.Div([
                    # Max New Tokens
                    html.Div([
                        html.Label("Max New Tokens:", className="input-label"),
                        dcc.Slider(
                            id='max-new-tokens-slider',
                            min=1, max=20, step=1, value=1,
                            marks={1: '1', 5: '5', 10: '10', 20: '20'},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], style={'flex': '1', 'marginRight': '20px'}),
                    
                    # Beam Width
                    html.Div([
                        html.Label("Beam Width (Parallel Futures):", className="input-label"),
                        dcc.Slider(
                            id='beam-width-slider',
                            min=1, max=5, step=1, value=1,
                            marks={1: '1', 3: '3', 5: '5'},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], style={'flex': '1'})
                ], style={'display': 'flex', 'marginBottom': '20px'}),
                
                # Generate Button
                html.Button(
                    [html.I(className="fas fa-play", style={'marginRight': '8px'}), "Generate"],
                    id="generate-btn",
                    className="action-button primary-button",
                    style={'width': '100%', 'padding': '12px', 'fontSize': '16px'}
                )
            ], className="generation-settings", style={'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px', 'marginTop': '15px'})
            
        ], className="config-section"),
        
        # 2. Results List (Populated via Callback)
        dcc.Loading(
            id="generation-loading",
            type="default",
            children=html.Div(id="generation-results-container", style={'marginTop': '20px'}),
            color='#667eea'
        ),
        
        # Analysis Loading Indicator (Deprecated, keeping ID for safety if needed or remove)
        # html.Div(id="analysis-loading-indicator", className="loading-container"),
        
        # 3. Sequence Analyzer (Visualizations)
        html.Div([
            html.Hr(style={'margin': '30px 0', 'borderTop': '1px solid #dee2e6'}),
            
            html.Div([
                html.H3("Sequence Analyzer", className="section-title"),
                
                # Scrubber
                html.Div([
                    html.Label("Sequence Scrubber (Step):", className="input-label"),
                    html.P("Drag to see how the model processed each step of the sequence.", style={'fontSize': '12px', 'color': '#6c757d'}),
                    dcc.Slider(
                        id='sequence-scrubber',
                        min=0, max=0, step=1, value=0,
                        marks={0: 'Start'},
                        tooltip={"placement": "bottom", "always_visible": True},
                        disabled=True
                    )
                ], id="scrubber-container", style={'marginBottom': '30px', 'padding': '15px', 'backgroundColor': '#e3f2fd', 'borderRadius': '8px'}),
                
                # Tokenization Panel
                create_tokenization_panel(),
                
                # Layer Visualizations
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
            ])
        ], id="analysis-view-container", style={'display': 'none'}) # Hidden by default
        
    ], className="main-panel-content")
