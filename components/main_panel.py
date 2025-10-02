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
        
        # Results section (placeholder)
        html.Div([
            html.H3("Analysis Results", className="section-title"),
            html.Div([
                html.P(
                    "Analysis results will appear here after running the analysis.",
                    className="placeholder-text"
                )
            ], id="results-container", className="results-area")
        ], className="results-section")
        
    ], className="main-panel-content")
