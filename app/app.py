"""
Main Dash application with callbacks for the transformer visualization dashboard.
"""

import logging
import traceback
from dash import Dash, html, dcc, Input, Output, State, callback, ctx
from dash_extensions.enrich import DashProxy, ServersideOutputTransform
import dash_bootstrap_components as dbc

# Import our components and services
from .constants import COMPONENT_IDS, ERROR_MESSAGES, SUCCESS_MESSAGES
from .components.top_bar import create_top_bar, get_processed_prompt
from .components.sidebar import create_sidebar, update_sidebar_options, get_current_selections
from .components.dashboard import create_dashboard, render_dashboard_visualization, create_loading_dashboard, create_error_dashboard
from .components.modals import create_layer_modal, render_head_view_content, render_model_view_content, render_loading_content, render_error_content

from .services.model_registry import get_model_and_tokenizer
from .services.module_discovery import discover_model_structure
from .services.activation_pipeline import run_complete_pipeline
from .services.bertviz_renderer import generate_bertviz_outputs
from .services.caching import get_cache_manager, create_cache_key_for_visualization

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Dash app with extensions
app = DashProxy(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
    ],
    transforms=[ServersideOutputTransform()],
    suppress_callback_exceptions=True
)

# App layout
app.layout = html.Div([
    # Top navigation bar
    create_top_bar(),
    
    # Main content area
    html.Div([
        # Left sidebar
        html.Div([
            create_sidebar()
        ], className="col-md-3"),
        
        # Main dashboard
        html.Div([
            create_dashboard()
        ], className="col-md-9")
    ], className="row"),
    
    # Modal for layer details
    create_layer_modal(),
    
    # Hidden stores for caching
    dcc.Store(id=COMPONENT_IDS["cached_data_store"]),
    dcc.Store(id=COMPONENT_IDS["session_store"])
    
], className="container-fluid")

# Global variables for caching
cache_manager = get_cache_manager()
current_activation_data = None
current_bertviz_outputs = None

# Callback for prompt validation
@app.callback(
    Output(COMPONENT_IDS["prompt_status"], "children"),
    Input(COMPONENT_IDS["prompt_input"], "value")
)
def validate_prompt_input(prompt_value):
    """Validate prompt input and show status messages."""
    from .components.top_bar import validate_prompt_length
    
    if not prompt_value or not prompt_value.strip():
        return html.Div("Please enter a prompt", className="status-warning")
    
    processed_prompt, was_truncated, message = validate_prompt_length(prompt_value)
    
    if was_truncated:
        return html.Div([
            html.I(className="fas fa-exclamation-triangle"),
            f" {message}"
        ], className="status-warning")
    
    word_count = len(prompt_value.strip().split())
    return html.Div([
        html.I(className="fas fa-check"),
        f" Ready ({word_count} words)"
    ], className="status-success")

# Callback to enable/disable visualize button based on selections
@app.callback(
    Output(COMPONENT_IDS["visualize_btn"], "disabled"),
    [Input(COMPONENT_IDS["attention_pattern_dropdown"], "value"),
     Input(COMPONENT_IDS["mlp_pattern_dropdown"], "value"),
     Input(COMPONENT_IDS["norm_param_dropdown"], "value"),
     Input(COMPONENT_IDS["logit_param_dropdown"], "value")]
)
def update_visualize_button(attention_pattern, mlp_pattern, norm_param, logit_param):
    """Update the visualize button state based on current selections."""
    from .components.sidebar import validate_selections
    
    is_valid, message = validate_selections(attention_pattern, mlp_pattern, norm_param, logit_param)
    return not is_valid

# Separate callback for sidebar status to avoid conflicts
@app.callback(
    Output(COMPONENT_IDS["sidebar_status"], "children"),
    [Input(COMPONENT_IDS["attention_pattern_dropdown"], "value"),
     Input(COMPONENT_IDS["mlp_pattern_dropdown"], "value"),
     Input(COMPONENT_IDS["norm_param_dropdown"], "value"),
     Input(COMPONENT_IDS["logit_param_dropdown"], "value"),
     Input(COMPONENT_IDS["find_modules_btn"], "n_clicks")],
    [State(COMPONENT_IDS["model_dropdown"], "value")],
    prevent_initial_call=True
)
def update_sidebar_status(attention_pattern, mlp_pattern, norm_param, logit_param, find_clicks, model_name):
    """Update sidebar status based on current state."""
    from .components.sidebar import validate_selections
    import dash
    
    # Determine which input triggered the callback
    ctx = dash.callback_context
    if not ctx.triggered:
        return ""
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # If triggered by find_modules_btn, show discovery status
    if trigger_id == COMPONENT_IDS["find_modules_btn"]:
        if find_clicks and model_name:
            return html.Div([
                html.I(className="fas fa-check"),
                " Module patterns discovered. Select parameters below."
            ], className="status-success")
    
    # Otherwise, show selection validation status
    is_valid, message = validate_selections(attention_pattern, mlp_pattern, norm_param, logit_param)
    
    if is_valid:
        status_component = html.Div([
            html.I(className="fas fa-check"),
            f" {message}"
        ], className="status-success")
        return status_component
    else:
        # Only show validation errors if we have some selections
        if any([attention_pattern, mlp_pattern, norm_param, logit_param]):
            status_component = html.Div([
                html.I(className="fas fa-exclamation-circle"),
                f" {message}"
            ], className="status-warning")
            return status_component
        else:
            return ""

# Callback for finding module names
@app.callback(
    [Output(COMPONENT_IDS["attention_pattern_dropdown"], "options"),
     Output(COMPONENT_IDS["mlp_pattern_dropdown"], "options"),
     Output(COMPONENT_IDS["norm_param_dropdown"], "options"),
     Output(COMPONENT_IDS["logit_param_dropdown"], "options"),
     Output(COMPONENT_IDS["attention_pattern_dropdown"], "disabled"),
     Output(COMPONENT_IDS["mlp_pattern_dropdown"], "disabled"),
     Output(COMPONENT_IDS["norm_param_dropdown"], "disabled"),
     Output(COMPONENT_IDS["logit_param_dropdown"], "disabled")],
    [Input(COMPONENT_IDS["find_modules_btn"], "n_clicks")],
    [State(COMPONENT_IDS["model_dropdown"], "value"),
     State(COMPONENT_IDS["prompt_input"], "value")],
    prevent_initial_call=True
)
def find_module_names(n_clicks, model_name, prompt):
    """
    Discover and populate module and parameter options for the selected model.
    """
    if not n_clicks or not model_name:
        return [], [], [], [], True, True, True, True
    
    try:
        logger.info(f"Finding modules for model: {model_name}")
        
        # Load model and tokenizer
        model, tokenizer = get_model_and_tokenizer(model_name)
        
        # Discover structure
        structure = discover_model_structure(model)
        
        if not structure["success"]:
            return [], [], [], [], True, True, True, True
        
        # Update sidebar options
        sidebar_updates = update_sidebar_options(structure["dropdown_options"])
        
        # Store structure for later use
        global current_model_structure
        current_model_structure = structure
        
        logger.info("Successfully found module names")
        return sidebar_updates
        
    except Exception as e:
        logger.error(f"Failed to find module names: {e}")
        return [], [], [], [], True, True, True, True

# Callback for visualization
@app.callback(
    [Output(COMPONENT_IDS["dashboard_content"], "children"),
     Output(COMPONENT_IDS["cached_data_store"], "data")],
    [Input(COMPONENT_IDS["visualize_btn"], "n_clicks")],
    [State(COMPONENT_IDS["model_dropdown"], "value"),
     State(COMPONENT_IDS["prompt_input"], "value"),
     State(COMPONENT_IDS["attention_pattern_dropdown"], "value"),
     State(COMPONENT_IDS["mlp_pattern_dropdown"], "value"),
     State(COMPONENT_IDS["norm_param_dropdown"], "value"),
     State(COMPONENT_IDS["logit_param_dropdown"], "value")],
    prevent_initial_call=True
)
def run_visualization(n_clicks, model_name, raw_prompt, attention_pattern, mlp_pattern, norm_param, logit_param):
    """
    Run the complete visualization pipeline.
    """
    if not n_clicks:
        return dash.no_update, dash.no_update
    
    # Validate inputs
    if not all([model_name, raw_prompt, attention_pattern, mlp_pattern, norm_param, logit_param]):
        error_dashboard = create_error_dashboard("Please select all required parameters.")
        return error_dashboard, None
    
    try:
        # Show loading state
        loading_dashboard = create_loading_dashboard()
        
        # Process prompt
        prompt = get_processed_prompt(raw_prompt)
        
        logger.info(f"Starting visualization pipeline for {model_name}")
        
        # Load model and tokenizer
        model, tokenizer = get_model_and_tokenizer(model_name)
        
        # Check if we have the structure from module discovery
        global current_model_structure
        if 'current_model_structure' not in globals():
            # Re-discover if needed
            current_model_structure = discover_model_structure(model)
        
        # Run complete pipeline
        pipeline_results = run_complete_pipeline(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            prompt=prompt,
            attention_pattern=attention_pattern,
            mlp_pattern=mlp_pattern,
            norm_param_name=norm_param,
            logit_lens_param_name=logit_param,
            pattern_to_modules=current_model_structure["pattern_to_modules"]
        )
        
        if not pipeline_results["success"]:
            error_dashboard = create_error_dashboard(pipeline_results["message"])
            return error_dashboard, None
        
        # Generate BertViz outputs
        bertviz_outputs = generate_bertviz_outputs(
            pipeline_results["activation_data"],
            pipeline_results["tokens"]
        )
        
        # Render dashboard
        dashboard_viz = render_dashboard_visualization(
            pipeline_results["activation_data"],
            pipeline_results["logit_lens_results"], 
            bertviz_outputs,
            pipeline_results["tokens"]
        )
        
        # Cache results for modal use
        cache_data = {
            'activation_data': pipeline_results["activation_data"],
            'logit_lens_results': pipeline_results["logit_lens_results"],
            'bertviz_outputs': bertviz_outputs,
            'tokens': pipeline_results["tokens"]
        }
        
        # Store globally for modal callbacks
        global current_activation_data, current_bertviz_outputs
        current_activation_data = pipeline_results["activation_data"]
        current_bertviz_outputs = bertviz_outputs
        
        logger.info("Visualization pipeline completed successfully")
        return dashboard_viz, cache_data
        
    except Exception as e:
        logger.error(f"Visualization pipeline failed: {e}")
        logger.error(traceback.format_exc())
        error_dashboard = create_error_dashboard(f"Pipeline failed: {str(e)}")
        return error_dashboard, None

# Callback for opening layer modal (simplified for now)
@app.callback(
    [Output(COMPONENT_IDS["layer_modal"], "style"),
     Output("modal-layer-title", "children"),
     Output(COMPONENT_IDS["modal_content"], "children")],
    [Input("dashboard-graph", "clickData"),
     Input("modal-close-btn", "n_clicks"),
     Input("modal-close-footer", "n_clicks")],
    [State(COMPONENT_IDS["modal_tabs"], "value"),
     State(COMPONENT_IDS["cached_data_store"], "data")],
    prevent_initial_call=True
)
def handle_modal(click_data, close_btn, close_footer, tab_value, cache_data):
    """
    Handle opening and closing the layer modal.
    """
    triggered = ctx.triggered_id
    
    # Close modal
    if triggered in ["modal-close-btn", "modal-close-footer"]:
        return {"display": "none"}, "", ""
    
    # Open modal from graph click
    if triggered == "dashboard-graph" and click_data and cache_data:
        try:
            # For now, show a simple message
            # In a full implementation, we'd extract the layer from click_data
            # and render the appropriate BertViz content
            
            modal_title = "Layer Attention Details"
            
            if tab_value == "head-view":
                content = html.Div([
                    html.H5("Head View"),
                    html.P("Interactive attention head view would be displayed here."),
                    html.P("Click data: " + str(click_data))
                ])
            else:
                content = html.Div([
                    html.H5("Model View"), 
                    html.P("Multi-layer attention view would be displayed here.")
                ])
            
            return {"display": "block"}, modal_title, content
            
        except Exception as e:
            logger.error(f"Failed to open modal: {e}")
            return {"display": "none"}, "", ""
    
    return {"display": "none"}, "", ""

# Callback for modal tab switching
@app.callback(
    Output(COMPONENT_IDS["modal_content"], "children", allow_duplicate=True),
    Input(COMPONENT_IDS["modal_tabs"], "value"),
    State(COMPONENT_IDS["cached_data_store"], "data"),
    prevent_initial_call=True
)
def update_modal_content(tab_value, cache_data):
    """
    Update modal content when tabs are switched.
    """
    if not cache_data:
        return render_error_content("Modal", "No data available")
    
    try:
        if tab_value == "head-view":
            return render_loading_content("Head View")
        else:
            return render_loading_content("Model View")
            
    except Exception as e:
        logger.error(f"Failed to update modal content: {e}")
        return render_error_content("Modal", str(e))

# Run the app
def run_app(debug=True, host="127.0.0.1", port=8050):
    """
    Run the Dash application.
    
    Args:
        debug: Enable debug mode
        host: Host address
        port: Port number
    """
    logger.info(f"Starting visualization dashboard on http://{host}:{port}")
    app.run(debug=debug, host=host, port=port)

if __name__ == "__main__":
    run_app()
