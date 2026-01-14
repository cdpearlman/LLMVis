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
                
                # Tokenization Panel (moved above heatmap)
                create_tokenization_panel(),
                
                # Layer Visualizations - Heatmap
                html.Div([
                    html.H3("Layer-by-Layer Predictions", className="section-title"),
                    
                    # Mode toggle buttons (comparison/ablation)
                    html.Div([
                        # Comparison mode toggle
                        html.Div([
                            html.Button("Prompt 1", id='heatmap-prompt1-btn', n_clicks=0,
                                       className='heatmap-toggle-btn active',
                                       style={'padding': '6px 16px', 'marginRight': '4px', 'border': '1px solid #667eea',
                                              'borderRadius': '4px 0 0 4px', 'backgroundColor': '#667eea', 'color': 'white',
                                              'cursor': 'pointer', 'fontSize': '13px'}),
                            html.Button("Prompt 2", id='heatmap-prompt2-btn', n_clicks=0,
                                       className='heatmap-toggle-btn',
                                       style={'padding': '6px 16px', 'border': '1px solid #667eea',
                                              'borderRadius': '0 4px 4px 0', 'backgroundColor': 'white', 'color': '#667eea',
                                              'cursor': 'pointer', 'fontSize': '13px'})
                        ], id='comparison-toggle-container', style={'display': 'none', 'marginRight': '20px'}),
                        
                        # Ablation mode toggle
                        html.Div([
                            html.Button("Original", id='heatmap-original-btn', n_clicks=0,
                                       className='heatmap-toggle-btn active',
                                       style={'padding': '6px 16px', 'marginRight': '4px', 'border': '1px solid #28a745',
                                              'borderRadius': '4px 0 0 4px', 'backgroundColor': '#28a745', 'color': 'white',
                                              'cursor': 'pointer', 'fontSize': '13px'}),
                            html.Button("Ablated", id='heatmap-ablated-btn', n_clicks=0,
                                       className='heatmap-toggle-btn',
                                       style={'padding': '6px 16px', 'border': '1px solid #28a745',
                                              'borderRadius': '0 4px 4px 0', 'backgroundColor': 'white', 'color': '#28a745',
                                              'cursor': 'pointer', 'fontSize': '13px'})
                        ], id='ablation-toggle-container', style={'display': 'none'})
                    ], id='heatmap-toggles', style={'display': 'flex', 'marginBottom': '15px'}),
                    
                    # Store for active heatmap mode
                    dcc.Store(id='heatmap-mode-store', data={'comparison': 'prompt1', 'ablation': 'original'}),
                    
                    # Heatmap container
                    dcc.Loading(
                        id="heatmap-loading",
                        type="default",
                        children=html.Div(id='heatmap-container', className="heatmap-visualization"),
                        overlay_style={"visibility": "visible", "opacity": .7, "backgroundColor": "white"},
                        custom_spinner=html.Div([
                            html.I(className="fas fa-spinner fa-spin", style={'fontSize': '24px', 'color': '#667eea', 'marginRight': '10px'}),
                            html.Span("Loading heatmap...", style={'fontSize': '16px', 'color': '#495057'})
                        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'padding': '2rem'})
                    ),
                    
                    # Sequence Ablation Results
                    html.Div(id='sequence-ablation-results-container', style={'marginTop': '30px', 'display': 'none'})
                ], className="visualization-section")
            ])
        ], id="analysis-view-container", style={'display': 'none'}),
        
        # Modal for layer details (click on heatmap cell)
        html.Div([
            html.Div([
                # Modal header
                html.Div([
                    html.H4(id='heatmap-modal-title', style={'margin': '0', 'color': '#495057'}),
                    html.Button('×', id='heatmap-modal-close', n_clicks=0,
                               style={'position': 'absolute', 'right': '15px', 'top': '15px',
                                      'background': 'none', 'border': 'none', 'fontSize': '28px',
                                      'cursor': 'pointer', 'color': '#6c757d', 'lineHeight': '1'})
                ], style={'position': 'relative', 'borderBottom': '1px solid #e9ecef', 
                         'paddingBottom': '15px', 'marginBottom': '15px'}),
                
                # Modal content container (populated by callback)
                html.Div(id='heatmap-modal-content', style={'maxHeight': '70vh', 'overflowY': 'auto'})
                
            ], id='heatmap-modal-inner', style={
                'backgroundColor': 'white',
                'padding': '25px',
                'borderRadius': '12px',
                'maxWidth': '900px',
                'width': '90%',
                'maxHeight': '85vh',
                'boxShadow': '0 10px 40px rgba(0,0,0,0.2)',
                'position': 'relative'
            })
        ], id='heatmap-modal-overlay', style={
            'position': 'fixed',
            'top': '0',
            'left': '0',
            'width': '100%',
            'height': '100%',
            'backgroundColor': 'rgba(0,0,0,0.5)',
            'zIndex': '1000',
            'display': 'none',
            'alignItems': 'center',
            'justifyContent': 'center'
        })
        
    ], className="main-panel-content")
