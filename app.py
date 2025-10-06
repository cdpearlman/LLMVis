"""
Modular Dash Cytoscape Visualization Dashboard

A simple, modular dashboard for transformer model visualization using Dash and Cytoscape.
Components are organized for easy understanding and maintenance.
"""

import dash
from dash import html, dcc, Input, Output, State, callback, no_update
import dash_cytoscape as cyto
from utils import (load_model_and_get_patterns, execute_forward_pass, format_data_for_cytoscape, 
                   categorize_single_layer_heads, format_categorization_summary,
                   compare_attention_layers, compare_output_probabilities, format_comparison_summary)
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

# Helper function for highlighting divergent layers
def _highlight_divergent_layers(elements, divergent_layer_nums):
    """Add red border styling to divergent layer nodes."""
    if not divergent_layer_nums:
        return elements
    
    # Create a set of divergent layer IDs
    divergent_ids = set(f'layer_{num}' for num in divergent_layer_nums)
    
    # Update elements with divergent styling
    updated_elements = []
    for element in elements:
        if element.get('data', {}).get('id') in divergent_ids:
            # Add classes or style to indicate divergence
            element['classes'] = 'divergent-layer'
        updated_elements.append(element)
    
    return updated_elements

# Helper function for creating category detail view with BertViz visualizations
def _create_category_detail_view(categorized_heads, activation_data):
    """Create BertViz visualizations organized by attention head category."""
    from utils import generate_category_bertviz_html
    
    category_colors = {
        'previous_token': '#ff7979',
        'first_token': '#74b9ff',
        'bow': '#ffeaa7',
        'syntactic': '#a29bfe',
        'other': '#dfe6e9'
    }
    
    category_names = {
        'previous_token': 'Previous-Token Heads',
        'first_token': 'First/Positional Heads',
        'bow': 'Bag-of-Words Heads',
        'syntactic': 'Syntactic Heads',
        'other': 'Other Heads'
    }
    
    sections = []
    
    for category_key, display_name in category_names.items():
        heads = categorized_heads.get(category_key, [])
        color = category_colors.get(category_key, '#dfe6e9')
        
        # Create head badges for summary
        head_badges = []
        for head_info in heads:
            badge = html.Span(
                head_info['label'],
                style={
                    'display': 'inline-block',
                    'padding': '4px 8px',
                    'margin': '4px',
                    'backgroundColor': color,
                    'borderRadius': '4px',
                    'fontSize': '12px',
                    'fontWeight': 'bold'
                }
            )
            head_badges.append(badge)
        
        # Generate BertViz visualization for this category
        if heads and activation_data:
            bertviz_html = generate_category_bertviz_html(activation_data, heads)
            viz_content = html.Iframe(
                srcDoc=bertviz_html,
                style={'width': '100%', 'height': '400px', 'border': '1px solid #ddd', 'borderRadius': '4px', 'marginTop': '10px'}
            )
        else:
            viz_content = html.P("No heads in this category.", style={'color': '#6c757d', 'fontSize': '13px', 'fontStyle': 'italic'})
        
        # Create category section with badges and visualization
        section = html.Div([
            html.H5(f"{display_name} ({len(heads)})", 
                   style={'marginBottom': '0.5rem', 'color': '#495057', 'borderLeft': f'4px solid {color}', 'paddingLeft': '10px'}),
            html.Div(head_badges if head_badges else html.P("None", style={'color': '#6c757d', 'fontSize': '13px'}),
                    style={'marginBottom': '0.5rem'}),
            viz_content
        ], style={'marginBottom': '2rem', 'padding': '1rem', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px'})
        
        sections.append(section)
    
    return html.Div(sections)

# Main app layout
app.layout = html.Div([
    # Session storage for activation data
    dcc.Store(id='session-activation-store', storage_type='session'),
    dcc.Store(id='session-patterns-store', storage_type='session'),
    # Sidebar collapse state (default: collapsed = True)
    dcc.Store(id='sidebar-collapse-store', storage_type='session', data=True),
    # Comparison mode state (default: not comparing)
    dcc.Store(id='comparison-mode-store', storage_type='session', data=False),
    # Second prompt activation data
    dcc.Store(id='session-activation-store-2', storage_type='session'),
    # Submitted check token (updated on Submit button click)
    dcc.Store(id='submitted-check-token-store', storage_type='session', data=''),
    
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
            ], id="sidebar-container", className="sidebar collapsed"),  # Start collapsed
            
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
     Output('block-modules-dropdown', 'options'),
     Output('norm-params-dropdown', 'options'),
     Output('logit-lens-dropdown', 'options'),
     Output('attention-modules-dropdown', 'value', allow_duplicate=True),
     Output('block-modules-dropdown', 'value', allow_duplicate=True),
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
        # Block options - layer/block modules (residual stream outputs)
        block_options = create_grouped_options(
            module_patterns, ['layers', 'h.', 'blocks', 'decoder.layers'], 'modules'
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
            block_options, 
            norm_options, 
            logit_lens_options,
            auto_selections.get('attention_selection', []),
            auto_selections.get('block_selection', []),
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
     Output('block-modules-dropdown', 'value'),
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
        None,  # block-modules-dropdown value  
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

# Callback to update submitted check token
@app.callback(
    Output('submitted-check-token-store', 'data'),
    [Input('submit-check-token-btn', 'n_clicks')],
    [State('check-token-input', 'value')],
    prevent_initial_call=True
)
def submit_check_token(n_clicks, check_token):
    """Store the check token when Submit is clicked."""
    if not n_clicks:
        return ''
    return check_token if check_token else ''

# Callback to run analysis and generate visualization
@app.callback(
    [Output('model-flow-graph', 'elements'),
     Output('model-flow-graph-2', 'elements'),
     Output('session-activation-store', 'data', allow_duplicate=True),
     Output('session-activation-store-2', 'data'),
     Output('analysis-loading-indicator', 'children'),
     Output('second-visualization-section', 'style', allow_duplicate=True),
     Output('comparison-section', 'style'),
     Output('comparison-container', 'children')],
    [Input('run-analysis-btn', 'n_clicks')],
    [State('model-dropdown', 'value'),
     State('prompt-input', 'value'),
     State('prompt-input-2', 'value'),
     State('submitted-check-token-store', 'data'),  # Use submitted token instead of input
     State('attention-modules-dropdown', 'value'),
     State('block-modules-dropdown', 'value'),
     State('norm-params-dropdown', 'value'), 
     State('logit-lens-dropdown', 'value'),
     State('session-patterns-store', 'data')],
    prevent_initial_call=True
)
def run_analysis(n_clicks, model_name, prompt, prompt2, check_token, attn_patterns, block_patterns, norm_patterns, logit_pattern, patterns_data):
    """Run forward pass and generate cytoscape visualization (handles 1 or 2 prompts)."""
    print(f"\n=== DEBUG: run_analysis START ===")
    print(f"DEBUG: n_clicks={n_clicks}, model_name={model_name}, prompt='{prompt}', prompt2='{prompt2}'")
    print(f"DEBUG: block_patterns={block_patterns}")
    print(f"DEBUG: logit_pattern={logit_pattern}")
    
    if not n_clicks or not model_name or not prompt or not block_patterns:
        print("DEBUG: Missing required inputs, returning empty")
        comparison_placeholder = html.P("Comparison analysis will appear here when two prompts are provided.", className="placeholder-text")
        return [], [], {}, {}, None, {'display': 'none'}, {'display': 'none'}, comparison_placeholder
    
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
        
        # Use block patterns (full layer outputs / residual stream) for logit lens
        config = {
            'attention_modules': [mod for pattern in (attn_patterns or []) for mod in module_patterns.get(pattern, [])],
            'block_modules': [mod for pattern in block_patterns for mod in module_patterns.get(pattern, [])],
            'norm_parameters': param_patterns.get(norm_patterns, []) if norm_patterns else [],
            'logit_lens_parameter': all_patterns.get(logit_pattern, [None])[0] if logit_pattern else None
        }
        
        print(f"DEBUG: config = {config}")
        
        # Execute forward pass for first prompt
        activation_data = execute_forward_pass(model, tokenizer, prompt, config)
        elements = format_data_for_cytoscape(activation_data, model, tokenizer, check_token)
        
        print(f"DEBUG: Created {len(elements)} elements for cytoscape (prompt 1)")
        
        # Store essential data for first prompt
        essential_data = {
            'model': model_name,
            'prompt': prompt,
            'attention_outputs': activation_data.get('attention_outputs', {}),
            'input_ids': activation_data.get('input_ids', [])
        }
        
        # Process second prompt if provided
        elements2 = []
        essential_data2 = {}
        second_viz_style = {'display': 'none'}  # Default: hide second viz
        comparison_style = {'display': 'none'}  # Default: hide comparison
        comparison_display = html.P("Comparison analysis will appear here when two prompts are provided.", className="placeholder-text")
        
        if prompt2 and prompt2.strip():
            activation_data2 = execute_forward_pass(model, tokenizer, prompt2, config)
            elements2 = format_data_for_cytoscape(activation_data2, model, tokenizer, check_token)
            print(f"DEBUG: Created {len(elements2)} elements for cytoscape (prompt 2)")
            
            essential_data2 = {
                'model': model_name,
                'prompt': prompt2,
                'attention_outputs': activation_data2.get('attention_outputs', {}),
                'input_ids': activation_data2.get('input_ids', [])
            }
            
            # Show second visualization if we have data
            if elements2:
                second_viz_style = {'display': 'block'}
            
            # Compute comparison between two prompts
            comparison_results = compare_attention_layers(activation_data, activation_data2)
            prob_differences = compare_output_probabilities(activation_data, activation_data2, model, tokenizer)
            comparison_summary = format_comparison_summary(comparison_results, prob_differences)
            
            # Get divergent layer numbers for highlighting
            divergent_layer_nums = set(ld['layer'] for ld in comparison_results['divergent_layers'])
            
            # Highlight divergent layers in both visualizations
            elements = _highlight_divergent_layers(elements, divergent_layer_nums)
            elements2 = _highlight_divergent_layers(elements2, divergent_layer_nums)
            
            # Create comparison display
            comparison_display = html.Div([
                html.Pre(comparison_summary, style={'whiteSpace': 'pre-wrap', 'fontFamily': 'monospace', 'fontSize': '13px'}),
                html.Hr(),
                html.Div([
                    html.H4("Divergent Layers", style={'marginTop': '1rem', 'color': '#e53e3e'}),
                    html.P(f"{len(divergent_layer_nums)} layers show significant differences between prompts.", 
                          style={'color': '#6c757d', 'fontSize': '14px'}),
                    html.P("These layers are highlighted with red borders in the visualizations above.",
                          style={'color': '#6c757d', 'fontSize': '14px'})
                ])
            ])
            
            # Show comparison section
            comparison_style = {'display': 'block'}
        
        # Show success message
        success_message = html.Div([
            html.I(className="fas fa-check-circle", style={'color': '#28a745', 'marginRight': '8px'}),
            "Analysis completed successfully!" + (" (2 prompts)" if prompt2 and prompt2.strip() else "")
        ], className="status-success")
        
        print(f"=== DEBUG: run_analysis END ===\n")
        return elements, elements2, essential_data, essential_data2, success_message, second_viz_style, comparison_style, comparison_display
        
    except Exception as e:
        print(f"Analysis error: {e}")
        import traceback
        traceback.print_exc()
        
        # Show error message
        error_message = html.Div([
            html.I(className="fas fa-exclamation-triangle", style={'color': '#dc3545', 'marginRight': '8px'}),
            f"Analysis error: {str(e)}"
        ], className="status-error")
        
        comparison_placeholder = html.P("Comparison analysis will appear here when two prompts are provided.", className="placeholder-text")
        return [], [], {}, {}, error_message, {'display': 'none'}, {'display': 'none'}, comparison_placeholder

# Enable Run Analysis button when requirements are met
@app.callback(
    Output('run-analysis-btn', 'disabled'),
    [Input('model-dropdown', 'value'),
     Input('prompt-input', 'value'),
     Input('block-modules-dropdown', 'value')]
)
def enable_run_button(model, prompt, block_modules):
    """Enable Run Analysis button when model, prompt, and layer blocks are selected."""
    return not (model and prompt and block_modules)

# Sidebar collapse toggle
@app.callback(
    [Output('sidebar-collapse-store', 'data'),
     Output('sidebar-content', 'style'),
     Output('sidebar-container', 'className'),
     Output('sidebar-toggle-btn', 'children')],
    [Input('sidebar-toggle-btn', 'n_clicks')],
    [State('sidebar-collapse-store', 'data')],
    prevent_initial_call=False
)
def toggle_sidebar(n_clicks, is_collapsed):
    """Toggle sidebar collapse state and update content visibility."""
    # On initial load, is_collapsed is True (default)
    if n_clicks is None:
        # Initial render: collapsed
        return True, {'display': 'none'}, 'sidebar collapsed', html.I(className="fas fa-chevron-right")
    
    # Toggle state
    new_collapsed = not is_collapsed
    style = {'display': 'none'} if new_collapsed else {'display': 'block'}
    class_name = 'sidebar collapsed' if new_collapsed else 'sidebar'
    # Icon changes: chevron-right when collapsed, chevron-left when expanded
    icon = html.I(className="fas fa-chevron-right") if new_collapsed else html.I(className="fas fa-chevron-left")
    
    return new_collapsed, style, class_name, icon

# Toggle comparison mode (show/hide second prompt)
@app.callback(
    [Output('comparison-mode-store', 'data'),
     Output('second-prompt-container', 'style'),
     Output('second-visualization-section', 'style'),
     Output('compare-prompts-btn', 'children'),
     Output('compare-prompts-btn', 'className')],
    [Input('compare-prompts-btn', 'n_clicks')],
    [State('comparison-mode-store', 'data')],
    prevent_initial_call=True
)
def toggle_comparison_mode(n_clicks, is_comparing):
    """Toggle comparison mode and show/hide second prompt input and visualization."""
    if not n_clicks:
        return False, {'display': 'none'}, {'display': 'none'}, [html.I(className="fas fa-plus", style={"marginRight": "5px"}), "Compare"], 'compare-button'
    
    # Toggle comparison mode
    new_comparing = not is_comparing
    style = {'display': 'block'} if new_comparing else {'display': 'none'}
    
    # Update button text and styling
    if new_comparing:
        # When comparing, show "Remove -" button in red
        button_content = [html.I(className="fas fa-minus", style={"marginRight": "5px"}), "Remove"]
        button_class = 'compare-button remove'
    else:
        # When not comparing, show "Compare +" button in green
        button_content = [html.I(className="fas fa-plus", style={"marginRight": "5px"}), "Compare"]
        button_class = 'compare-button'
    
    return new_comparing, style, style, button_content, button_class

# Node click callback for analysis results  
@app.callback(
    Output('results-container', 'children'),
    [Input('model-flow-graph', 'tapNodeData')],
    [State('session-activation-store', 'data')],
    prevent_initial_call=True
)
def show_layer_analysis(node_data, activation_data):
    """Show BertViz analysis and head categorization when a layer node is clicked."""
    if not node_data or not activation_data:
        return html.P("Click a layer node to see detailed attention analysis and head categorization.", className="placeholder-text")
    
    try:
        from utils import generate_bertviz_html
        layer_num = node_data['layer_num']
        
        # Generate BertViz HTML for this layer (full model_view)
        bertviz_html = generate_bertviz_html(activation_data, layer_num, 'full')
        
        # Categorize heads for this specific layer
        categorized_heads = categorize_single_layer_heads(activation_data, layer_num)
        category_summary = format_categorization_summary(categorized_heads)
        
        # Create category detail view with BertViz visualizations
        category_detail = _create_category_detail_view(categorized_heads, activation_data)
        
        return html.Div([
            html.H4(f"Layer {layer_num} - Full Model View"),
            html.Iframe(
                srcDoc=bertviz_html,
                style={'width': '100%', 'height': '500px', 'border': 'none', 'marginBottom': '2rem'}
            ),
            html.Hr(),
            html.H4(f"Layer {layer_num} - Attention Head Categorization"),
            html.Pre(category_summary, style={'whiteSpace': 'pre-wrap', 'fontFamily': 'monospace', 'fontSize': '13px', 'marginTop': '1rem'}),
            html.Hr(),
            html.H4("Category Details with BertViz Visualizations", style={'marginTop': '1rem'}),
            category_detail
        ])
        
    except Exception as e:
        return html.P(f"Error loading analysis: {str(e)}", className="placeholder-text")

# Edge hover tooltips
@app.callback(
    Output('edge-tooltip', 'children'),
    Output('edge-tooltip', 'style'),
    [Input('model-flow-graph', 'mouseoverEdgeData'),
     Input('model-flow-graph', 'mouseoverNodeData')]
)
def show_edge_tooltip(edge_data, node_data):
    """Show tooltip when hovering over edges in first visualization."""
    if edge_data and 'token' in edge_data:
        token = edge_data['token']
        prob = edge_data['probability']
        return (
            f"{token} | {prob:.3f}",
            {
                'position': 'absolute',
                'top': '10px',
                'left': '50%',
                'transform': 'translateX(-50%)',
                'backgroundColor': 'rgba(0, 0, 0, 0.85)',
                'color': 'white',
                'padding': '8px 12px',
                'borderRadius': '6px',
                'fontSize': '13px',
                'fontWeight': '500',
                'whiteSpace': 'nowrap',
                'zIndex': '1000',
                'pointerEvents': 'none',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.3)'
            }
        )
    return '', {'display': 'none'}

@app.callback(
    Output('edge-tooltip-2', 'children'),
    Output('edge-tooltip-2', 'style'),
    [Input('model-flow-graph-2', 'mouseoverEdgeData'),
     Input('model-flow-graph-2', 'mouseoverNodeData')]
)
def show_edge_tooltip_2(edge_data, node_data):
    """Show tooltip when hovering over edges in second visualization."""
    if edge_data and 'token' in edge_data:
        token = edge_data['token']
        prob = edge_data['probability']
        return (
            f"{token} | {prob:.3f}",
            {
                'position': 'absolute',
                'top': '10px',
                'left': '50%',
                'transform': 'translateX(-50%)',
                'backgroundColor': 'rgba(0, 0, 0, 0.85)',
                'color': 'white',
                'padding': '8px 12px',
                'borderRadius': '6px',
                'fontSize': '13px',
                'fontWeight': '500',
                'whiteSpace': 'nowrap',
                'zIndex': '1000',
                'pointerEvents': 'none',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.3)'
            }
        )
    return '', {'display': 'none'}

if __name__ == '__main__':
    app.run(debug=True, port=8050)
