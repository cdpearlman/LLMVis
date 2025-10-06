"""
Main panel component for the dashboard.

This component contains:
- Model selector and prompt input
- Visualization area (placeholder for now)
- Status and results display
"""

from dash import html, dcc
import dash_cytoscape as cyto
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
        
        # Check Token input (for 4th edge visualization)
        html.Div([
            html.Label("Check Token (optional):", className="input-label"),
            html.Div([
                dcc.Input(
                    id='check-token-input',
                    type='text',
                    placeholder="Enter a token to track its probability...",
                    value="",
                    style={"width": "300px", "display": "inline-block", "marginRight": "10px"},
                    className="prompt-input"
                ),
                html.Button(
                    "Submit",
                    id="submit-check-token-btn",
                    className="action-button primary-button",
                    style={"display": "inline-block", "padding": "0.5rem 1rem", "fontSize": "13px"}
                ),
                html.Span(" (Adds a 4th edge showing this token's probability at each layer)", 
                         style={"fontSize": "12px", "color": "#6c757d", "marginLeft": "10px"})
            ], style={"display": "flex", "alignItems": "center"})
        ], className="input-container", style={"marginBottom": "1.5rem"}),
        
        # Visualization section (first prompt)
        html.Div([
            html.H3("Model Flow Visualization", className="section-title"),
            html.Div([
                cyto.Cytoscape(
                    id='model-flow-graph',
                    elements=[],
                    layout={'name': 'preset'},
                    style={'width': '100%', 'height': '400px'},
                    zoom=1.0,
                    pan={'x': 100, 'y': 200},
                stylesheet=[
                    # Node styles
                    {
                        'selector': 'node',
                        'style': {
                            'width': '60px',
                            'height': '60px',
                            'background-color': '#667eea',
                            'border-color': '#5a67d8',
                            'border-width': '2px',
                            'label': 'data(label)',
                            'text-valign': 'center',
                            'color': 'white',
                            'font-size': '10px',
                            'text-wrap': 'wrap'
                        }
                    },
                    # Divergent layer node style (red border for different prompts)
                    {
                        'selector': 'node.divergent-layer',
                        'style': {
                            'border-color': '#e53e3e',
                            'border-width': '4px'
                        }
                    },
                    # Output node style (distinct from layer nodes)
                    {
                        'selector': 'node[id="output_node"]',
                        'style': {
                            'width': '80px',
                            'height': '80px',
                            'background-color': '#48bb78',
                            'border-color': '#38a169',
                            'border-width': '3px',
                            'label': 'data(label)',
                            'text-valign': 'center',
                            'color': 'white',
                            'font-size': '11px',
                            'font-weight': 'bold',
                            'text-wrap': 'wrap',
                            'shape': 'round-rectangle'
                        }
                    },
                    # Edge styles
                    {
                        'selector': 'edge',
                        'style': {
                            'width': 'data(width)',
                            'opacity': 'data(opacity)',
                            'line-color': 'data(color)',
                            'target-arrow-color': 'data(color)',
                            'target-arrow-shape': 'triangle',
                            'curve-style': 'bezier'
                        }
                    }
                ]
                ),
                # Tooltip for edge hover
                html.Div(id='edge-tooltip', style={'display': 'none'})
            ], style={'position': 'relative'})
        ], className="visualization-section"),
        
        # Second visualization (for comparison - initially hidden)
        html.Div([
            html.H3("Model Flow Visualization (Prompt 2)", className="section-title"),
            html.Div([
                cyto.Cytoscape(
                    id='model-flow-graph-2',
                    elements=[],
                    layout={'name': 'preset'},
                    style={'width': '100%', 'height': '400px'},
                    zoom=1.0,
                    pan={'x': 100, 'y': 200},
                stylesheet=[
                    # Node styles
                    {
                        'selector': 'node',
                        'style': {
                            'width': '60px',
                            'height': '60px',
                            'background-color': '#667eea',
                            'border-color': '#5a67d8',
                            'border-width': '2px',
                            'label': 'data(label)',
                            'text-valign': 'center',
                            'color': 'white',
                            'font-size': '10px',
                            'text-wrap': 'wrap'
                        }
                    },
                    # Divergent layer node style (red border for different prompts)
                    {
                        'selector': 'node.divergent-layer',
                        'style': {
                            'border-color': '#e53e3e',
                            'border-width': '4px'
                        }
                    },
                    # Output node style (distinct from layer nodes)
                    {
                        'selector': 'node[id="output_node"]',
                        'style': {
                            'width': '80px',
                            'height': '80px',
                            'background-color': '#48bb78',
                            'border-color': '#38a169',
                            'border-width': '3px',
                            'label': 'data(label)',
                            'text-valign': 'center',
                            'color': 'white',
                            'font-size': '11px',
                            'font-weight': 'bold',
                            'text-wrap': 'wrap',
                            'shape': 'round-rectangle'
                        }
                    },
                    # Edge styles
                    {
                        'selector': 'edge',
                        'style': {
                            'width': 'data(width)',
                            'opacity': 'data(opacity)',
                            'line-color': 'data(color)',
                            'target-arrow-color': 'data(color)',
                            'target-arrow-shape': 'triangle',
                            'curve-style': 'bezier'
                        }
                    }
                ]
                ),
                # Tooltip for edge hover
                html.Div(id='edge-tooltip-2', style={'display': 'none'})
            ], style={'position': 'relative'})
        ], id="second-visualization-section", className="visualization-section", style={'display': 'none'}),
        
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
