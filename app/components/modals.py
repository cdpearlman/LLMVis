"""
Modal components for detailed layer views.
"""

import dash
from dash import dcc, html
import logging

from ..constants import COMPONENT_IDS

logger = logging.getLogger(__name__)

def create_layer_modal() -> html.Div:
    """
    Create the layer detail modal with tabs for head view and model view.
    
    Returns:
        Modal HTML component
    """
    return html.Div([
        # Modal overlay
        html.Div([
            # Modal dialog
            html.Div([
                # Modal header
                html.Div([
                    html.H4("Layer Attention Details", id="modal-layer-title"),
                    html.Button(
                        "×", 
                        id="modal-close-btn",
                        className="close-button",
                        **{"aria-label": "Close"}
                    )
                ], className="modal-header"),
                
                # Modal tabs
                html.Div([
                    dcc.Tabs(
                        id=COMPONENT_IDS["modal_tabs"],
                        value="head-view",
                        children=[
                            dcc.Tab(
                                label="Head View (Single Layer)",
                                value="head-view",
                                className="modal-tab"
                            ),
                            dcc.Tab(
                                label="Model View (All Layers)",
                                value="model-view", 
                                className="modal-tab"
                            )
                        ],
                        className="modal-tabs"
                    )
                ]),
                
                # Modal content
                html.Div([
                    html.Div(
                        id=COMPONENT_IDS["modal_content"],
                        className="modal-body-content"
                    )
                ], className="modal-body"),
                
                # Modal footer
                html.Div([
                    html.Button(
                        "Close",
                        id="modal-close-footer",
                        className="btn btn-secondary"
                    )
                ], className="modal-footer")
                
            ], className="modal-dialog")
        ], className="modal-overlay")
    ], 
    id=COMPONENT_IDS["layer_modal"], 
    className="modal-container",
    style={"display": "none"}
    )

def render_head_view_content(layer_idx: int, head_view_html: str) -> html.Div:
    """
    Render content for the head view tab.
    
    Args:
        layer_idx: Layer index
        head_view_html: HTML content for head view
        
    Returns:
        Head view content component
    """
    return html.Div([
        html.H5(f"Layer {layer_idx} - Attention Head Patterns"),
        html.P("This view shows attention patterns for each head in the selected layer."),
        html.Div([
            # Embed the BertViz HTML
            html.Iframe(
                srcDoc=head_view_html,
                style={
                    "width": "100%",
                    "height": "600px",
                    "border": "1px solid #ddd",
                    "border-radius": "4px"
                }
            )
        ], className="bertviz-container")
    ])

def render_model_view_content(model_view_html: str) -> html.Div:
    """
    Render content for the model view tab.
    
    Args:
        model_view_html: HTML content for model view
        
    Returns:
        Model view content component
    """
    return html.Div([
        html.H5("Multi-Layer Attention Patterns"),
        html.P("This view shows attention patterns across all layers in the model."),
        html.Div([
            # Embed the BertViz HTML
            html.Iframe(
                srcDoc=model_view_html,
                style={
                    "width": "100%", 
                    "height": "700px",
                    "border": "1px solid #ddd",
                    "border-radius": "4px"
                }
            )
        ], className="bertviz-container")
    ])

def render_loading_content(tab_name: str) -> html.Div:
    """
    Render loading content for modal tabs.
    
    Args:
        tab_name: Name of the tab being loaded
        
    Returns:
        Loading content component
    """
    return html.Div([
        html.Div([
            html.I(className="fas fa-spinner fa-spin fa-2x"),
            html.H5(f"Generating {tab_name}...", className="mt-3"),
            html.P("Please wait while we generate the attention visualization.")
        ], className="text-center py-5")
    ])

def render_error_content(tab_name: str, error_message: str) -> html.Div:
    """
    Render error content for modal tabs.
    
    Args:
        tab_name: Name of the tab with error
        error_message: Error message to display
        
    Returns:
        Error content component
    """
    return html.Div([
        html.Div([
            html.I(className="fas fa-exclamation-triangle fa-2x text-danger"),
            html.H5(f"Error Loading {tab_name}", className="mt-3 text-danger"),
            html.P(error_message, className="text-muted"),
            html.P("Please try again or check the console for more details.")
        ], className="text-center py-5")
    ])

# Modal control callbacks would be defined in the main app.py file
# to avoid circular imports and ensure proper callback registration
