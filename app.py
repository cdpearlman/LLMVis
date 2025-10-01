"""
Modular Dash Cytoscape Visualization Dashboard

A simple, modular dashboard for transformer model visualization using Dash and Cytoscape.
Components are organized for easy understanding and maintenance.
"""

import dash
from dash import html, dcc, Input, Output, State, callback, no_update
import dash_cytoscape as cyto
from utils import load_model_and_get_patterns, execute_forward_pass, format_data_for_cytoscape
from utils.model_config import get_auto_selections, get_model_family

# Import modular components
from components.sidebar import create_sidebar
from components.model_selector import create_model_selector
from components.main_panel import create_main_panel

# Initialize Dash app with external stylesheets
app = dash.Dash(
    __name__, 
    suppress_callback_exceptions=True,
    external_stylesheets=[
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
    ]
)
app.title = "Transformer Visualization Dashboard"

# Define available models
AVAILABLE_MODELS = [
    "Qwen/Qwen2.5-0.5B",
    "gpt2"
]

# Main app layout
app.layout = html.Div([
    # Session storage for activation data
    dcc.Store(id='session-activation-store', storage_type='session'),
    dcc.Store(id='session-patterns-store', storage_type='session'),
    
    # Main container
    html.Div([
        # Header
        html.Div([
            html.H1("Transformer Model Visualization Dashboard", 
                   className="header-title"),
            html.P("Analyze transformer models with interactive visualizations", 
                   className="header-subtitle")
        ], className="header"),
        
        # Main content area
        html.Div([
            # Left sidebar
            html.Div([
                create_sidebar()
            ], className="sidebar"),
            
            # Right main panel
            html.Div([
                create_main_panel()
            ], className="main-panel")
        ], className="content-container")
    ], className="app-container")
], className="app-wrapper")

# Callback to load model patterns when model is selected
@app.callback(
    [Output('session-patterns-store', 'data'),
     Output('attention-modules-dropdown', 'options'),
     Output('mlp-modules-dropdown', 'options'),
     Output('norm-params-dropdown', 'options'),
     Output('logit-lens-dropdown', 'options'),
     Output('attention-modules-dropdown', 'value', allow_duplicate=True),
     Output('mlp-modules-dropdown', 'value', allow_duplicate=True),
     Output('norm-params-dropdown', 'value', allow_duplicate=True),
     Output('logit-lens-dropdown', 'value', allow_duplicate=True),
     Output('loading-indicator', 'children')],
    [Input('model-dropdown', 'value')],
    prevent_initial_call=True
)
def load_model_patterns(selected_model):
    """Load and categorize model patterns when a model is selected."""
    if not selected_model:
        return {}, [], [], [], [], None, None, None, None, None
    
    try:
        # Load model patterns
        module_patterns, param_patterns = load_model_and_get_patterns(selected_model)
        
        # Create options with filtered items first, then all others
        def create_grouped_options(patterns_dict, filter_keywords, option_type):
            filtered_options = []
            other_options = []
            
            for pattern, items in patterns_dict.items():
                pattern_lower = pattern.lower()
                option = {
                    'label': pattern, 
                    'value': pattern,
                    'title': f"{len(items)} {option_type} matching this pattern"  # Tooltip
                }
                
                # Check if pattern matches any filter keywords
                if any(keyword in pattern_lower for keyword in filter_keywords):
                    filtered_options.append(option)
                else:
                    other_options.append(option)
            
            # Combine with separator if both groups exist
            result = []
            if filtered_options:
                result.extend(filtered_options)
                if other_options:
                    result.append({'label': '─── Other Options ───', 'value': '_separator_', 'disabled': True})
                    result.extend(other_options)
            else:
                result.extend(other_options)
            
            return result
        
        # Create grouped options for each dropdown
        attention_options = create_grouped_options(
            module_patterns, ['attn', 'attention'], 'modules'
        )
        mlp_options = create_grouped_options(
            module_patterns, ['mlp'], 'modules'
        )
        norm_options = create_grouped_options(
            param_patterns, ['norm', 'layernorm', 'layer_norm'], 'params'
        )
        # Logit lens can be in either modules or parameters - combine both
        combined_patterns = {**param_patterns, **module_patterns}
        logit_lens_options = create_grouped_options(
            combined_patterns, ['lm_head', 'head', 'classifier', 'embed', 'wte', 'word'], 'items'
        )
        
        # Get auto-selections based on model family
        auto_selections = get_auto_selections(selected_model, module_patterns, param_patterns)
        
        # Store patterns data with family info
        patterns_data = {
            'module_patterns': module_patterns,
            'param_patterns': param_patterns,
            'selected_model': selected_model,
            'family': auto_selections.get('family_name'),
            'family_description': auto_selections.get('family_description', '')
        }
        
        # Prepare loading indicator with family info
        family_name = auto_selections.get('family_name')
        if family_name:
            loading_content = html.Div([
                html.I(className="fas fa-check-circle", style={'color': '#28a745', 'marginRight': '8px'}),
                html.Div([
                    html.Div("Model patterns loaded successfully!"),
                    html.Div(f"Detected family: {auto_selections.get('family_description', family_name)}", 
                            style={'fontSize': '12px', 'color': '#6c757d', 'marginTop': '4px'})
                ])
            ], className="status-success")
        else:
            loading_content = html.Div([
                html.I(className="fas fa-check-circle", style={'color': '#28a745', 'marginRight': '8px'}),
                html.Div([
                    html.Div("Model patterns loaded successfully!"),
                    html.Div("No family detected - manual selection required", 
                            style={'fontSize': '12px', 'color': '#f0ad4e', 'marginTop': '4px'})
                ])
            ], className="status-success")
        
        return (
            patterns_data, 
            attention_options, 
            mlp_options, 
            norm_options, 
            logit_lens_options,
            auto_selections.get('attention_selection', []),
            auto_selections.get('mlp_selection', []),
            auto_selections.get('norm_selection', []),
            auto_selections.get('logit_lens_selection'),
            loading_content
        )
        
    except Exception as e:
        print(f"Error loading model patterns: {e}")
        error_content = html.Div([
            html.I(className="fas fa-exclamation-triangle", style={'color': '#dc3545', 'marginRight': '8px'}),
            f"Error loading model: {str(e)}"
        ], className="status-error")
        return {}, [], [], [], [], None, None, None, None, error_content

# Callback to show loading spinner when model is being processed
@app.callback(
    Output('loading-indicator', 'children', allow_duplicate=True),
    [Input('model-dropdown', 'value')],
    prevent_initial_call=True
)
def show_loading_spinner(selected_model):
    """Show loading spinner when model selection changes."""
    if not selected_model:
        return None
    
    return html.Div([
        html.I(className="fas fa-spinner fa-spin", style={'marginRight': '8px'}),
        "Finding model parameters..."
    ], className="status-loading")

# Callback to clear all selections when Clear button is pressed
@app.callback(
    [Output('attention-modules-dropdown', 'value'),
     Output('mlp-modules-dropdown', 'value'),
     Output('norm-params-dropdown', 'value'),
     Output('logit-lens-dropdown', 'value'),
     Output('session-activation-store', 'data'),
     Output('loading-indicator', 'children', allow_duplicate=True)],
    [Input('clear-selections-btn', 'n_clicks')],
    prevent_initial_call=True
)
def clear_all_selections(n_clicks):
    """Clear all dropdown selections and backend data when Clear button is pressed."""
    if not n_clicks:
        return no_update
    
    # Show cleared status
    cleared_status = html.Div([
        html.I(className="fas fa-broom", style={'color': '#6c757d', 'marginRight': '8px'}),
        "All selections cleared"
    ], className="status-cleared")
    
    return (
        None,  # attention-modules-dropdown value
        None,  # mlp-modules-dropdown value  
        None,  # norm-params-dropdown value
        None,  # logit-lens-dropdown value
        {},    # session-activation-store data
        cleared_status  # loading-indicator children
    )

# Callback to show loading spinner when Run Analysis is clicked
@app.callback(
    Output('analysis-loading-indicator', 'children', allow_duplicate=True),
    [Input('run-analysis-btn', 'n_clicks')],
    prevent_initial_call=True
)
def show_analysis_loading_spinner(n_clicks):
    """Show loading spinner when Run Analysis button is clicked."""
    if not n_clicks:
        return None
    
    return html.Div([
        html.I(className="fas fa-spinner fa-spin", style={'marginRight': '8px'}),
        "Collecting Data..."
    ], className="status-loading")

# Callback to run analysis and generate visualization
@app.callback(
    [Output('model-flow-graph', 'elements'),
     Output('session-activation-store', 'data', allow_duplicate=True),
     Output('analysis-loading-indicator', 'children')],
    [Input('run-analysis-btn', 'n_clicks')],
    [State('model-dropdown', 'value'),
     State('prompt-input', 'value'),
     State('attention-modules-dropdown', 'value'),
     State('mlp-modules-dropdown', 'value'),
     State('norm-params-dropdown', 'value'), 
     State('logit-lens-dropdown', 'value'),
     State('session-patterns-store', 'data')],
    prevent_initial_call=True
)
def run_analysis(n_clicks, model_name, prompt, attn_patterns, mlp_patterns, norm_patterns, logit_pattern, patterns_data):
    """Run forward pass and generate cytoscape visualization."""
    print(f"\n=== DEBUG: run_analysis START ===")
    print(f"DEBUG: n_clicks={n_clicks}, model_name={model_name}, prompt='{prompt}'")
    print(f"DEBUG: mlp_patterns={mlp_patterns}")
    print(f"DEBUG: logit_pattern={logit_pattern}")
    
    if not n_clicks or not model_name or not prompt or not mlp_patterns:
        print("DEBUG: Missing required inputs, returning empty")
        return [], {}, None
    
    try:
        # Load model for execution
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation='eager')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.eval()
        
        # Build config from selected patterns
        module_patterns = patterns_data.get('module_patterns', {})
        param_patterns = patterns_data.get('param_patterns', {})
        all_patterns = {**module_patterns, **param_patterns}
        
        config = {
            'attention_modules': [mod for pattern in (attn_patterns or []) for mod in module_patterns.get(pattern, [])],
            'mlp_modules': [mod for pattern in mlp_patterns for mod in module_patterns.get(pattern, [])],
            'norm_parameters': [param for pattern in (norm_patterns or []) for param in param_patterns.get(pattern, [])],
            'logit_lens_parameter': all_patterns.get(logit_pattern, [None])[0] if logit_pattern else None
        }
        
        print(f"DEBUG: config = {config}")
        
        # Execute forward pass
        activation_data = execute_forward_pass(model, tokenizer, prompt, config)
        
        # Format for cytoscape
        elements = format_data_for_cytoscape(activation_data, model, tokenizer)
        
        print(f"DEBUG: Created {len(elements)} elements for cytoscape")
        
        # Store only essential data to avoid quota issues
        essential_data = {
            'model': model_name,  # Fix: use 'model' key that BertViz expects
            'prompt': prompt,
            'attention_outputs': activation_data.get('attention_outputs', {}),
            'input_ids': activation_data.get('input_ids', [])
        }
        
        # Show success message
        success_message = html.Div([
            html.I(className="fas fa-check-circle", style={'color': '#28a745', 'marginRight': '8px'}),
            "Analysis completed successfully!"
        ], className="status-success")
        
        print(f"=== DEBUG: run_analysis END ===\n")
        return elements, essential_data, success_message
        
    except Exception as e:
        print(f"Analysis error: {e}")
        import traceback
        traceback.print_exc()
        
        # Show error message
        error_message = html.Div([
            html.I(className="fas fa-exclamation-triangle", style={'color': '#dc3545', 'marginRight': '8px'}),
            f"Analysis error: {str(e)}"
        ], className="status-error")
        
        return [], {}, error_message

# Enable Run Analysis button when requirements are met
@app.callback(
    Output('run-analysis-btn', 'disabled'),
    [Input('model-dropdown', 'value'),
     Input('prompt-input', 'value'),
     Input('mlp-modules-dropdown', 'value')]
)
def enable_run_button(model, prompt, mlp_modules):
    """Enable Run Analysis button when model, prompt, and MLP modules are selected."""
    return not (model and prompt and mlp_modules)

# Node click callback for analysis results  
@app.callback(
    Output('results-container', 'children'),
    [Input('model-flow-graph', 'tapNodeData')],
    [State('session-activation-store', 'data')],
    prevent_initial_call=True
)
def show_layer_analysis(node_data, activation_data):
    """Show BertViz analysis when a layer node is clicked."""
    if not node_data or not activation_data:
        return html.P("Click a layer node to see detailed analysis.", className="placeholder-text")
    
    try:
        from utils import generate_bertviz_html
        layer_num = node_data['layer_num']
        
        # Generate BertViz HTML for this layer
        bertviz_html = generate_bertviz_html(activation_data, layer_num, 'full')
        
        return html.Div([
            html.H4(f"Layer {layer_num} Analysis"),
            html.Iframe(
                srcDoc=bertviz_html,
                style={'width': '100%', 'height': '500px', 'border': 'none'}
            )
        ])
        
    except Exception as e:
        return html.P(f"Error loading analysis: {str(e)}", className="placeholder-text")

# Edge hover callback for token information
@app.callback(
    Output('model-status', 'children'),
    [Input('model-flow-graph', 'mouseoverEdgeData')]
)
def show_edge_info(hover_data):
    """Show token and probability info when hovering over edges."""
    if hover_data and 'token' in hover_data:
        token = hover_data['token']
        prob = hover_data['probability']
        return html.Div([
            html.Span(f"Token: {token} | Probability: {prob:.3f}", 
                     style={'font-size': '14px', 'color': '#495057'})
        ])
    return None

if __name__ == '__main__':
    app.run(debug=True, port=8050)
