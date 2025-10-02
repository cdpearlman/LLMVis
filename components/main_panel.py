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
            )
        ], className="visualization-section"),
        
        # Second visualization (for comparison - initially hidden)
        html.Div([
            html.H3("Model Flow Visualization (Prompt 2)", className="section-title"),
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
            )
        ], id="second-visualization-section", className="visualization-section", style={'display': 'none'}),
        
        # Head Categorization section
        html.Div([
            html.H3("Attention Head Categorization", className="section-title"),
            html.Div([
                html.P(
                    "Head categorization will appear here after running analysis.",
                    className="placeholder-text"
                )
            ], id="head-categorization-container", className="results-area")
        ], className="results-section"),
        
        # Results section (BertViz visualization on node click)
        html.Div([
            html.H3("Layer Analysis (BertViz)", className="section-title"),
            html.Div([
                html.P(
                    "Click a layer node to see detailed attention analysis.",
                    className="placeholder-text"
                )
            ], id="results-container", className="results-area")
        ], className="results-section")
        
    ], className="main-panel-content")
