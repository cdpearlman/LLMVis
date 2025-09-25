"""
Sidebar component with pattern and parameter selection dropdowns.
"""

import dash
from dash import dcc, html
import logging

from ..constants import COMPONENT_IDS, CSS_CLASSES

logger = logging.getLogger(__name__)

def create_sidebar() -> html.Div:
    """
    Create the sidebar component with selection dropdowns.
    
    Returns:
        Dash HTML div containing the sidebar components
    """
    return html.Div([
        html.H3("Module & Parameter Selection", className="sidebar-title"),
        
        # Attention Pattern Selection
        html.Div([
            html.Label([
                html.I(className="fas fa-eye"),
                " Attention Output Variable"
            ], className="dropdown-label"),
            dcc.Dropdown(
                id=COMPONENT_IDS["attention_pattern_dropdown"],
                placeholder="Select attention pattern...",
                clearable=False,
                disabled=True,
                className="pattern-dropdown"
            ),
            html.Small("Select the attention module pattern", className="help-text")
        ], className="dropdown-group"),
        
        # MLP Pattern Selection
        html.Div([
            html.Label([
                html.I(className="fas fa-cogs"),
                " MLP Output Variable"
            ], className="dropdown-label"),
            dcc.Dropdown(
                id=COMPONENT_IDS["mlp_pattern_dropdown"],
                placeholder="Select MLP pattern...",
                clearable=False,
                disabled=True,
                className="pattern-dropdown"
            ),
            html.Small("Select the MLP module pattern", className="help-text")
        ], className="dropdown-group"),
        
        # Normalization Parameter Selection
        html.Div([
            html.Label([
                html.I(className="fas fa-balance-scale"),
                " Normalization Parameter"
            ], className="dropdown-label"),
            dcc.Dropdown(
                id=COMPONENT_IDS["norm_param_dropdown"],
                placeholder="Select normalization parameter...",
                clearable=False,
                disabled=True,
                className="param-dropdown"
            ),
            html.Small("Select the final normalization parameter", className="help-text")
        ], className="dropdown-group"),
        
        # Logit Lens Parameter Selection
        html.Div([
            html.Label([
                html.I(className="fas fa-microscope"),
                " Logit Lens Parameter"
            ], className="dropdown-label"),
            dcc.Dropdown(
                id=COMPONENT_IDS["logit_param_dropdown"],
                placeholder="Select logit lens parameter...",
                clearable=False,
                disabled=True,
                className="param-dropdown"
            ),
            html.Small("Select the projection parameter for logit lens", className="help-text")
        ], className="dropdown-group"),
        
        # Visualize Button
        html.Div([
            html.Button(
                [
                    html.I(className="fas fa-chart-line"),
                    " Visualize"
                ],
                id=COMPONENT_IDS["visualize_btn"],
                n_clicks=0,
                className="btn btn-success btn-block",
                disabled=True
            )
        ], className="button-group"),
        
        # Status message area
        html.Div(
            id=COMPONENT_IDS["sidebar_status"],
            className="status-message"
        )
        
    ], className=CSS_CLASSES["sidebar"])

def validate_selections(attention_pattern, mlp_pattern, norm_param, logit_param) -> tuple[bool, str]:
    """
    Validate that all required selections are made.
    
    Args:
        attention_pattern: Selected attention pattern
        mlp_pattern: Selected MLP pattern
        norm_param: Selected normalization parameter
        logit_param: Selected logit lens parameter
        
    Returns:
        Tuple of (is_valid, message)
    """
    missing = []
    
    if not attention_pattern:
        missing.append("Attention Pattern")
    if not mlp_pattern:
        missing.append("MLP Pattern")
    if not norm_param:
        missing.append("Normalization Parameter")
    if not logit_param:
        missing.append("Logit Lens Parameter")
    
    if missing:
        message = f"Missing: {', '.join(missing)}"
        return False, message
    
    return True, "All selections complete"

# Callbacks will be defined in main app.py

def update_sidebar_options(dropdown_options: dict) -> tuple:
    """
    Update sidebar dropdown options with discovered patterns and parameters.
    
    Args:
        dropdown_options: Dictionary with options for each dropdown
        
    Returns:
        Tuple of options for each dropdown component
    """
    # Format options for dropdowns
    attention_options = [
        {"label": pattern, "value": pattern} 
        for pattern in dropdown_options.get("attention_pattern_options", [])
    ]
    
    mlp_options = [
        {"label": pattern, "value": pattern} 
        for pattern in dropdown_options.get("mlp_pattern_options", [])
    ]
    
    norm_options = [
        {"label": param, "value": param} 
        for param in dropdown_options.get("norm_param_options", [])
    ]
    
    logit_options = [
        {"label": param, "value": param} 
        for param in dropdown_options.get("logit_param_options", [])
    ]
    
    logger.info(f"Updated sidebar options: "
                f"{len(attention_options)} attention, "
                f"{len(mlp_options)} MLP, "
                f"{len(norm_options)} norm, "
                f"{len(logit_options)} logit")
    
    # Return options and enabled states
    has_options = bool(attention_options and mlp_options and norm_options and logit_options)
    
    return (
        attention_options,  # attention dropdown options
        mlp_options,       # mlp dropdown options
        norm_options,      # norm dropdown options
        logit_options,     # logit dropdown options
        not has_options,   # attention dropdown disabled
        not has_options,   # mlp dropdown disabled
        not has_options,   # norm dropdown disabled
        not has_options    # logit dropdown disabled
    )

def get_current_selections(attention_pattern, mlp_pattern, norm_param, logit_param) -> dict:
    """
    Get current selections as a dictionary.
    
    Returns:
        Dictionary with current selections
    """
    return {
        "attention_pattern": attention_pattern,
        "mlp_pattern": mlp_pattern,
        "norm_param_name": norm_param,
        "logit_lens_param_name": logit_param
    }
