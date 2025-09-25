"""
Main dashboard component for displaying layer tiles and connections.
"""

import dash
from dash import dcc, html
import plotly.graph_objects as go
import plotly.express as px
import logging
from typing import List, Dict, Any

from ..constants import COMPONENT_IDS, CSS_CLASSES, TILE_WIDTH, TILE_HEIGHT
from ..utils.layout_math import (
    compute_grid_positions, calculate_tile_coordinates, 
    compute_edge_paths, normalize_edge_opacities,
    calculate_dashboard_dimensions, get_scroll_config
)

logger = logging.getLogger(__name__)

def create_dashboard() -> html.Div:
    """
    Create the main dashboard container.
    
    Returns:
        Dash HTML div containing the dashboard
    """
    return html.Div([
        # Dashboard content area
        html.Div([
            # Placeholder content
            html.Div([
                html.H3("Transformer Layer Visualization", className="dashboard-title"),
                html.P("Select a model and click 'Find Module Names', then choose your parameters and click 'Visualize' to see the layer connections.", 
                       className="dashboard-instructions")
            ], className="dashboard-placeholder")
        ], id=COMPONENT_IDS["dashboard_content"], className="dashboard-content"),
        
        # Hidden modal for layer details
        create_layer_modal()
        
    ], id=COMPONENT_IDS["main_dashboard"], className=CSS_CLASSES["dashboard"])

def create_layer_modal() -> html.Div:
    """
    Create modal for displaying detailed layer views.
    
    Returns:
        Modal component
    """
    return html.Div([
        html.Div([
            # Modal header
            html.Div([
                html.H4("Layer Details", id="modal-title"),
                html.Button("×", id="modal-close", className="close-btn")
            ], className="modal-header"),
            
            # Modal tabs
            html.Div([
                dcc.Tabs(
                    id=COMPONENT_IDS["modal_tabs"],
                    value="head-view",
                    children=[
                        dcc.Tab(label="Head View", value="head-view"),
                        dcc.Tab(label="Model View", value="model-view")
                    ]
                )
            ], className="modal-tabs"),
            
            # Modal content
            html.Div(
                id=COMPONENT_IDS["modal_content"],
                className="modal-body"
            )
            
        ], className="modal-content")
    ], id=COMPONENT_IDS["layer_modal"], className="modal", style={"display": "none"})

def create_layer_tile(layer_idx: int, thumbnail_html: str, coordinates: Dict[str, float]) -> Dict[str, Any]:
    """
    Create a single layer tile representation.
    
    Args:
        layer_idx: Layer index
        thumbnail_html: HTML for the thumbnail
        coordinates: Tile coordinates
        
    Returns:
        Dictionary representing the tile
    """
    return {
        "type": "tile",
        "layer_idx": layer_idx,
        "x": coordinates["x"],
        "y": coordinates["y"],
        "width": coordinates["width"],
        "height": coordinates["height"],
        "thumbnail_html": thumbnail_html,
        "clickable": True
    }

def build_dashboard_figure(
    tiles: List[Dict[str, Any]], 
    edges: List[Dict[str, Any]], 
    tokens: List[str]
) -> go.Figure:
    """
    Build the main dashboard Plotly figure with tiles and edges.
    
    Args:
        tiles: List of tile dictionaries
        edges: List of edge dictionaries
        tokens: List of token strings for context
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Add edges first (so they appear behind tiles)
    for edge in edges:
        fig.add_trace(go.Scatter(
            x=[edge["start_x"], edge["end_x"]],
            y=[edge["start_y"], edge["end_y"]],
            mode="lines",
            line=dict(
                color=f"rgba(0, 0, 0, {edge['opacity']})",
                width=3
            ),
            hovertext=edge["hover_text"],
            hoverinfo="text",
            showlegend=False,
            name=f"Edge L{edge['layer_idx']}"
        ))
    
    # Add layer tiles
    for tile in tiles:
        # Create a rectangle for the tile
        fig.add_shape(
            type="rect",
            x0=tile["x"],
            y0=tile["y"],
            x1=tile["x"] + tile["width"],
            y1=tile["y"] + tile["height"],
            fillcolor="lightblue",
            line=dict(color="navy", width=2),
        )
        
        # Add layer label
        fig.add_annotation(
            x=tile["x"] + tile["width"]/2,
            y=tile["y"] + tile["height"]/2,
            text=f"Layer {tile['layer_idx']}",
            showarrow=False,
            font=dict(size=12, color="navy"),
        )
    
    # Calculate figure dimensions
    if tiles:
        max_x = max(tile["x"] + tile["width"] for tile in tiles) + 50
        max_y = max(tile["y"] + tile["height"] for tile in tiles) + 50
    else:
        max_x, max_y = 800, 600
    
    # Update layout
    fig.update_layout(
        width=max_x,
        height=max_y,
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[0, max_x]
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[0, max_y],
            scaleanchor="x",
            scaleratio=1
        ),
        plot_bgcolor="white",
        margin=dict(l=0, r=0, t=30, b=0),
        title="Layer Connections and Token Predictions"
    )
    
    return fig

def render_dashboard_visualization(
    activation_data: Dict[str, Any],
    logit_lens_results: List[List[tuple]],
    bertviz_outputs: Dict[str, Any],
    tokens: List[str]
) -> html.Div:
    """
    Render the complete dashboard visualization.
    
    Args:
        activation_data: Complete activation data
        logit_lens_results: Logit lens results per layer
        bertviz_outputs: BertViz rendering outputs
        tokens: Token strings
        
    Returns:
        Dashboard HTML component
    """
    try:
        num_layers = len(logit_lens_results)
        
        if num_layers == 0:
            return html.Div([
                html.H3("No Data", className="text-center"),
                html.P("No layers found in the results.", className="text-center text-muted")
            ])
        
        # Compute layout
        positions = compute_grid_positions(num_layers)
        coordinates = calculate_tile_coordinates(positions)
        
        # Create tiles
        tiles = []
        for i in range(num_layers):
            # Get thumbnail from bertviz outputs
            thumbnail_html = bertviz_outputs.get('thumbnails', {}).get(i, f"<div>Layer {i}</div>")
            tile = create_layer_tile(i, thumbnail_html, coordinates[i])
            tiles.append(tile)
        
        # Create edges
        edges = compute_edge_paths(coordinates, logit_lens_results)
        edges = normalize_edge_opacities(edges)
        
        # Build figure
        fig = build_dashboard_figure(tiles, edges, tokens)
        
        # Create dashboard component
        dashboard_component = html.Div([
            html.Div([
                html.H3(f"Visualization: {num_layers} Layers"),
                html.P(f"Prompt: \"{' '.join(tokens[:10])}{'...' if len(tokens) > 10 else ''}\"")
            ], className="dashboard-header"),
            
            dcc.Graph(
                figure=fig,
                id="dashboard-graph",
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
                }
            ),
            
            html.Div([
                html.P(f"✓ Generated visualization for {num_layers} layers with {len(edges)} connections")
            ], className="dashboard-status")
        ])
        
        logger.info(f"Rendered dashboard with {num_layers} layers and {len(edges)} edges")
        return dashboard_component
        
    except Exception as e:
        logger.error(f"Failed to render dashboard: {e}")
        return html.Div([
            html.H3("Visualization Error", className="text-center text-danger"),
            html.P(f"Failed to render visualization: {e}", className="text-center")
        ])

def create_loading_dashboard() -> html.Div:
    """
    Create a loading state for the dashboard.
    
    Returns:
        Loading dashboard component
    """
    return html.Div([
        html.Div([
            html.I(className="fas fa-spinner fa-spin fa-2x"),
            html.H4("Generating Visualization...", className="mt-3"),
            html.P("This may take a moment while we capture activations and generate attention patterns.")
        ], className="text-center py-5")
    ])

def create_error_dashboard(error_message: str) -> html.Div:
    """
    Create an error state for the dashboard.
    
    Args:
        error_message: Error message to display
        
    Returns:
        Error dashboard component
    """
    return html.Div([
        html.Div([
            html.I(className="fas fa-exclamation-triangle fa-2x text-danger"),
            html.H4("Visualization Error", className="mt-3 text-danger"),
            html.P(error_message, className="text-muted")
        ], className="text-center py-5")
    ])
