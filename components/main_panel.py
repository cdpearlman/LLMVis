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
        
        # Visualization section
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
