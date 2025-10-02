"""
Sidebar component with module and parameter selection dropdowns.

This component provides the left sidebar interface for selecting:
- Attention modules
- Layer blocks (residual stream outputs)
- Normalization parameters
- Logit lens parameters
"""

from dash import html, dcc

def create_sidebar():
    """Create the left sidebar with selection dropdowns."""
    return html.Div([
        # Toggle button for collapsing/expanding sidebar
        html.Button(
            html.I(className="fas fa-bars"),
            id="sidebar-toggle-btn",
            className="sidebar-toggle-button",
            title="Toggle sidebar"
        ),
        
        # Sidebar content (hidden when collapsed)
        html.Div([
            html.H3("Module Selection", className="sidebar-title"),
        
        # Loading/status indicator
        html.Div(id="loading-indicator", className="loading-container"),
        
        # Attention modules dropdown
        html.Div([
            html.Label("Attention Modules:", className="dropdown-label"),
            dcc.Dropdown(
                id='attention-modules-dropdown',
                options=[],
                value=None,
                placeholder="Select attention modules...",
                multi=True,
                className="module-dropdown"
            )
        ], className="dropdown-container"),
        
        # Layer blocks dropdown (residual stream outputs)
        html.Div([
            html.Label("Layer Blocks:", className="dropdown-label"),
            dcc.Dropdown(
                id='block-modules-dropdown',
                options=[],
                value=None,
                placeholder="Select layer blocks...",
                multi=True,
                className="module-dropdown"
            )
        ], className="dropdown-container"),
        
        # Normalization parameters dropdown
        html.Div([
            html.Label("Normalization Parameters:", className="dropdown-label"),
            dcc.Dropdown(
                id='norm-params-dropdown',
                options=[],
                value=None,
                placeholder="Select norm parameters...",
                multi=True,
                className="module-dropdown"
            )
        ], className="dropdown-container"),
        
        # Logit lens parameter dropdown (single selection)
        html.Div([
            html.Label("Logit Lens Parameter:", className="dropdown-label"),
            dcc.Dropdown(
                id='logit-lens-dropdown',
                options=[],
                value=None,
                placeholder="Select logit lens parameter...",
                multi=False,
                className="module-dropdown"
            )
        ], className="dropdown-container"),
        
            # Action buttons
            html.Div([
                html.Button(
                    "Run Analysis", 
                    id="run-analysis-btn",
                    className="action-button primary-button",
                    disabled=True
                ),
                html.Button(
                    "Clear Selections", 
                    id="clear-selections-btn",
                    className="action-button secondary-button"
                )
            ], className="button-container")
        ], id="sidebar-content", className="sidebar-content")
        
    ])
