"""
Modular Dash Cytoscape Visualization Dashboard

A simple, modular dashboard for transformer model visualization using Dash and Cytoscape.
Components are organized for easy understanding and maintenance.
"""

import dash
from dash import html, dcc, Input, Output, State, callback, no_update, ALL, MATCH
import json
import torch
from utils import (load_model_and_get_patterns, execute_forward_pass, extract_layer_data,
                   categorize_single_layer_heads, format_categorization_summary,
                   compute_layer_wise_summaries, perform_beam_search, compute_sequence_trajectory,
                   execute_forward_pass_with_head_ablation, evaluate_sequence_ablation, score_sequence,
                   compute_position_layer_matrix)
from utils.model_config import get_auto_selections, get_model_family

# Import modular components
from components.sidebar import create_sidebar
from components.model_selector import create_model_selector
from components.main_panel import create_main_panel
from components.glossary import create_glossary_modal

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

def _create_token_probability_delta_chart(layer_data, layer_num, global_top5_tokens, title_suffix=""):
    """Create a horizontal bar chart showing probability changes for global top 5 tokens."""
    import plotly.graph_objs as go
    
    # Get deltas for global top 5 tokens
    deltas = layer_data.get('deltas', {})
    
    # Filter for global top 5 tokens
    tokens = []
    delta_values = []
    colors = []
    
    for token_info in global_top5_tokens:
        token = token_info.get('token', '')
        # Handle merging logic here if needed, but deltas should already key by merged token
        # Check if token exists in deltas (try exact match first)
        delta = deltas.get(token, 0.0)
        
        tokens.append(token)
        delta_values.append(delta)
        colors.append('#28a745' if delta >= 0 else '#dc3545')
        
    # Create horizontal bar chart
    fig = go.Figure(go.Bar(
        x=delta_values,
        y=tokens,
        orientation='h',
        marker_color=colors,
        text=[f"{val:+.2%}" for val in delta_values],
        textposition='auto',
        hoverinfo='text+y',
        hovertext=[f"Token: {t}<br>Change: {v:+.4f}" for t, v in zip(tokens, delta_values)]
    ))
    
    fig.update_layout(
        title=f"Probability Changes (L{layer_num-1} → L{layer_num}) {title_suffix}",
        xaxis_title="Change in Probability",
        yaxis_title="Token",
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def _create_comparison_delta_chart(layer1, layer2, layer_num, global_top5_1, global_top5_2):
    """Create grouped bar chart comparing deltas for two models/prompts."""
    import plotly.graph_objs as go
    
    # Combine unique tokens from both top 5 sets
    tokens1 = {t.get('token') for t in global_top5_1}
    tokens2 = {t.get('token') for t in global_top5_2}
    all_tokens = sorted(list(tokens1.union(tokens2)))
    
    deltas1 = layer1.get('deltas', {})
    deltas2 = layer2.get('deltas', {})
    
    values1 = [deltas1.get(t, 0.0) for t in all_tokens]
    values2 = [deltas2.get(t, 0.0) for t in all_tokens]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=all_tokens,
        x=values1,
        name='Prompt 1',
        orientation='h',
        marker_color='#74b9ff'
    ))
    fig.add_trace(go.Bar(
        y=all_tokens,
        x=values2,
        name='Prompt 2',
        orientation='h',
        marker_color='#ff7979'
    ))
    
    fig.update_layout(
        title=f"Probability Changes Comparison (Layer {layer_num})",
        barmode='group',
        xaxis_title="Change in Probability",
        yaxis_title="Token",
        margin=dict(l=20, r=20, t=40, b=20),
        height=400
    )
    
    return fig

def _create_top5_by_layer_graph(layer_wise_probs, significant_layers, global_top5):
    """Create line graph of top 5 token probabilities across layers."""
    import plotly.graph_objs as go
    
    fig = go.Figure()
    
    layers = sorted([int(k) for k in layer_wise_probs.keys()])
    if not layers:
        return None
        
    for i, token_info in enumerate(global_top5):
        token = token_info.get('token', '')
        probs = []
        for layer in layers:
            layer_probs = layer_wise_probs.get(layer, {}) # Use layer key directly (int)
            # Try to find probability for this token
            prob = layer_probs.get(token, 0.0)
            probs.append(prob)
            
        fig.add_trace(go.Scatter(
            x=layers,
            y=probs,
            mode='lines+markers',
            name=token,
            line=dict(width=2),
            marker=dict(size=6)
        ))
        
    # Add highlighting for significant layers
    shapes = []
    for layer in significant_layers:
        shapes.append(dict(
            type="rect",
            xref="x",
            yref="paper",
            x0=layer - 0.4,
            x1=layer + 0.4,
            y0=0,
            y1=1,
            fillcolor="yellow",
            opacity=0.2,
            layer="below",
            line_width=0,
        ))
        
    fig.update_layout(
        title="Top 5 Token Probabilities Across Layers",
        xaxis_title="Layer",
        yaxis_title="Probability",
        hovermode="x unified",
        shapes=shapes,
        margin=dict(l=20, r=20, t=40, b=20),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def _create_actual_output_display(activation_data):
    """Create display for the actual final output token."""
    actual_output = activation_data.get('actual_output')
    if not actual_output:
        return None
        
    token = actual_output.get('token', '')
    prob = actual_output.get('probability', 0.0)
    
    return html.Div([
        html.Strong("Final Output: ", style={'color': '#495057'}),
        html.Span(token, style={'fontFamily': 'monospace', 'fontWeight': 'bold', 'backgroundColor': '#e2e8f0', 'padding': '2px 6px', 'borderRadius': '4px'}),
        html.Span(f" (p={prob:.2%})", style={'color': '#6c757d', 'marginLeft': '8px', 'fontSize': '13px'})
    ], style={'marginTop': '10px', 'padding': '8px', 'backgroundColor': '#f8f9fa', 'borderRadius': '4px', 'borderLeft': '4px solid #28a745'})


# Main app layout
app.layout = html.Div([
    # Glossary Modal
    create_glossary_modal(),
    
    # Session storage for activation data
    dcc.Store(id='session-activation-store', storage_type='memory'),
    dcc.Store(id='session-patterns-store', storage_type='session'),
    # Store original activation data before ablation for comparison
    dcc.Store(id='session-activation-store-original', storage_type='memory'),
    # Sidebar collapse state (default: collapsed = True)
    dcc.Store(id='sidebar-collapse-store', storage_type='session', data=True),
    # Comparison mode state (default: not comparing)
    dcc.Store(id='comparison-mode-store', storage_type='session', data=False),
    # Second prompt activation data
    dcc.Store(id='session-activation-store-2', storage_type='memory'),
    # Generation results store
    dcc.Store(id='generation-results-store', storage_type='session'),
    
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

# Glossary Callbacks
@app.callback(
    [Output("glossary-modal-overlay", "style"),
     Output("glossary-modal-content", "style")],
    [Input("open-glossary-btn", "n_clicks"),
     Input("close-glossary-btn", "n_clicks"),
     Input("glossary-modal-overlay", "n_clicks")],
    prevent_initial_call=True
)
def toggle_glossary(open_clicks, close_clicks, overlay_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "open-glossary-btn":
        return {'display': 'flex'}, {'display': 'block'}
    else:
        return {'display': 'none'}, {'display': 'none'}

# Callback to load model patterns when model is selected
@app.callback(
    [Output('session-patterns-store', 'data'),
     Output('attention-modules-dropdown', 'options'),
     Output('block-modules-dropdown', 'options'),
     Output('norm-params-dropdown', 'options'),
     Output('attention-modules-dropdown', 'value', allow_duplicate=True),
     Output('block-modules-dropdown', 'value', allow_duplicate=True),
     Output('norm-params-dropdown', 'value', allow_duplicate=True),
     Output('loading-indicator', 'children')],
    [Input('model-dropdown', 'value')],
    prevent_initial_call=True
)
def load_model_patterns(selected_model):
    """Load and categorize model patterns when a model is selected."""
    if not selected_model:
        return {}, [], [], [], None, None, None, None
    
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
            auto_selections.get('attention_selection', []),
            auto_selections.get('block_selection', []),
            auto_selections.get('norm_selection', []),
            loading_content
        )
        
    except Exception as e:
        print(f"Error loading model patterns: {e}")
        error_content = html.Div([
            html.I(className="fas fa-exclamation-triangle", style={'color': '#dc3545', 'marginRight': '8px'}),
            f"Error loading model: {str(e)}"
        ], className="status-error")
        return {}, [], [], [], None, None, None, error_content

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
     Output('session-activation-store', 'data', allow_duplicate=True),
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
        {},    # session-activation-store data
        cleared_status  # loading-indicator children
    )

# Enable Run Analysis button when requirements are met
@app.callback(
    Output('generate-btn', 'disabled'),
    [Input('model-dropdown', 'value'),
     Input('prompt-input', 'value'),
     Input('block-modules-dropdown', 'value'),
     Input('norm-params-dropdown', 'value')]
)
def enable_run_button(model, prompt, block_modules, norm_params):
    """Enable Generate button when model, prompt, layer blocks, and norm parameters are selected."""
    return not (model and prompt and block_modules and norm_params)

# Callback to Run Generation / Analysis
@app.callback(
    [Output('generation-results-container', 'children', allow_duplicate=True),
     Output('generation-results-store', 'data', allow_duplicate=True),
     Output('analysis-view-container', 'style', allow_duplicate=True),
     Output('session-activation-store', 'data', allow_duplicate=True)],
    [Input('generate-btn', 'n_clicks')],
    [State('model-dropdown', 'value'),
     State('prompt-input', 'value'),
     State('max-new-tokens-slider', 'value'),
     State('beam-width-slider', 'value'),
     State('session-patterns-store', 'data'),
     State('attention-modules-dropdown', 'value'),
     State('block-modules-dropdown', 'value'),
     State('norm-params-dropdown', 'value')],
    prevent_initial_call=True
)
def run_generation(n_clicks, model_name, prompt, max_new_tokens, beam_width, patterns_data, attn_patterns, block_patterns, norm_patterns):
    if not n_clicks or not model_name or not prompt:
        return no_update, no_update, no_update, no_update

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # Load model once
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation='eager')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.eval()

        # Perform Beam Search
        results = perform_beam_search(model, tokenizer, prompt, beam_width, max_new_tokens)
        
        # Results UI
        results_ui = []
        if max_new_tokens > 1:
            results_ui.append(html.H4("Generated Sequences (Ranked)", className="section-title"))
            for i, result in enumerate(results):
                text = result['text']
                score = result['score']
                results_ui.append(html.Div([
                    html.Div([
                        html.Span(f"Rank {i+1}", style={'fontWeight': 'bold', 'marginRight': '10px', 'color': '#667eea'}),
                        html.Span(f"Score: {score:.4f}", style={'fontSize': '12px', 'color': '#6c757d'})
                    ], style={'marginBottom': '5px'}),
                    html.Div(text, style={'fontFamily': 'monospace', 'backgroundColor': '#fff', 'padding': '10px', 'borderRadius': '4px', 'border': '1px solid #dee2e6'}),
                    html.Button(
                        "Analyze This Sequence",
                        id={'type': 'result-item', 'index': i},
                        n_clicks=0,
                        className="action-button secondary-button",
                        style={'marginTop': '10px', 'fontSize': '12px', 'padding': '5px 10px'}
                    )
                ], style={'marginBottom': '15px', 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '6px', 'border': '1px solid #e9ecef'}))
            
            # Return just the list, hide analyzer
            return results_ui, results, {'display': 'none'}, {}
            
        else:
            # Single token case: Run analysis immediately
            result = results[0]
            text = result['text']
            
            # Use defaults if patterns not selected
            module_patterns = patterns_data.get('module_patterns', {})
            param_patterns = patterns_data.get('param_patterns', {})
            
            config = {
                'attention_modules': [mod for pattern in (attn_patterns or []) for mod in module_patterns.get(pattern, [])],
                'block_modules': [mod for pattern in (block_patterns or []) for mod in module_patterns.get(pattern, [])],
                'norm_parameters': [param for pattern in (norm_patterns or []) for param in param_patterns.get(pattern, [])]
            }
            
            if not config['block_modules']:
                 return html.Div("Please select modules in the sidebar first.", style={'color': 'red'}), results, {'display': 'none'}, {}

            # Run forward pass on the Generated Text
            activation_data = execute_forward_pass(model, tokenizer, text, config)
            
            return results_ui, results, {'display': 'block'}, activation_data

    except Exception as e:
        import traceback
        traceback.print_exc()
        return html.Div(f"Error: {e}", style={'color': 'red'}), [], {'display': 'none'}, {}

# Callback to Analyze a specific sequence from results list
@app.callback(
    [Output('session-activation-store', 'data', allow_duplicate=True),
     Output('analysis-view-container', 'style', allow_duplicate=True)],
    Input({'type': 'result-item', 'index': ALL}, 'n_clicks'),
    [State('generation-results-store', 'data'),
     State('model-dropdown', 'value'),
     State('session-patterns-store', 'data'),
     State('attention-modules-dropdown', 'value'),
     State('block-modules-dropdown', 'value'),
     State('norm-params-dropdown', 'value')],
    prevent_initial_call=True
)
def analyze_selected_sequence(n_clicks_list, results_data, model_name, patterns_data, attn_patterns, block_patterns, norm_patterns):
    if not any(n_clicks_list) or not results_data:
        return no_update, no_update
    
    # Find which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update
        
    triggered_id = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
    index = triggered_id['index']
    
    try:
        result = results_data[index]
        text = result['text']
        
        # Load model & tokenizer
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation='eager')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.eval()
        
        # Config
        module_patterns = patterns_data.get('module_patterns', {})
        param_patterns = patterns_data.get('param_patterns', {})
        
        config = {
            'attention_modules': [mod for pattern in (attn_patterns or []) for mod in module_patterns.get(pattern, [])],
            'block_modules': [mod for pattern in (block_patterns or []) for mod in module_patterns.get(pattern, [])],
            'norm_parameters': [param for pattern in (norm_patterns or []) for param in param_patterns.get(pattern, [])]
        }
        
        if not config['block_modules']:
             return no_update, no_update
             
        # Run forward pass
        activation_data = execute_forward_pass(model, tokenizer, text, config)
        
        return activation_data, {'display': 'block'}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return no_update, no_update

# Render heatmap visualization (replaces accordion view)
@app.callback(
    [Output('heatmap-container', 'children'),
     Output('comparison-toggle-container', 'style'),
     Output('ablation-toggle-container', 'style')],
    [Input('session-activation-store', 'data'),
     Input('session-activation-store-2', 'data'),
     Input('session-activation-store-original', 'data'),
     Input('heatmap-mode-store', 'data')],
    [State('model-dropdown', 'value')]
)
def render_heatmap(activation_data, activation_data2, original_activation_data, mode_data, model_name):
    """Render Position x Layer heatmap visualization."""
    hide_style = {'display': 'none'}
    show_style = {'display': 'flex', 'marginRight': '20px'}
    
    if not activation_data or not model_name:
        return html.P("Run analysis to see layer-by-layer predictions.", className="placeholder-text"), hide_style, hide_style
    
    # Safety check for invalid data structure
    if isinstance(activation_data, list):
        return html.P("Error: Invalid activation data format. Please refresh the page.", className="placeholder-text", style={'color': 'red'}), hide_style, hide_style
    
    if isinstance(activation_data2, list):
        activation_data2 = None
        
    if isinstance(original_activation_data, list):
        original_activation_data = None

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import plotly.graph_objs as go
        
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation='eager')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Determine which data to visualize based on mode
        comparison_mode = mode_data.get('comparison', 'prompt1') if mode_data else 'prompt1'
        ablation_mode = mode_data.get('ablation', 'original') if mode_data else 'original'
        
        # Show toggles based on available data
        show_comparison_toggle = activation_data2 is not None
        show_ablation_toggle = original_activation_data is not None and activation_data.get('ablated', False)
        
        # Select the appropriate activation data
        if show_ablation_toggle and ablation_mode == 'original':
            active_data = original_activation_data
        elif show_comparison_toggle and comparison_mode == 'prompt2':
            active_data = activation_data2
        else:
            active_data = activation_data
        
        # Compute the position-layer matrix
        matrix_data = compute_position_layer_matrix(active_data, model, tokenizer)
        
        if not matrix_data['matrix'] or not matrix_data['layer_nums']:
            return html.P("No layer data available for heatmap.", className="placeholder-text"), hide_style, hide_style
        
        # Create heatmap
        z_data = matrix_data['matrix']
        tokens = matrix_data['tokens']
        layer_nums = matrix_data['layer_nums']
        top_tokens = matrix_data['top_tokens']
        
        # Reverse layer order so L0 is at bottom
        z_data_reversed = list(reversed(z_data))
        layer_nums_reversed = list(reversed(layer_nums))
        top_tokens_reversed = list(reversed(top_tokens))
        
        # Create custom hover text
        hover_text = []
        for layer_idx, layer_row in enumerate(z_data_reversed):
            hover_row = []
            for pos_idx, delta in enumerate(layer_row):
                token = tokens[pos_idx] if pos_idx < len(tokens) else ''
                top_tok = top_tokens_reversed[layer_idx][pos_idx] if pos_idx < len(top_tokens_reversed[layer_idx]) else ''
                hover_row.append(f"Token: {token}<br>Layer: L{layer_nums_reversed[layer_idx]}<br>Top: '{top_tok}'<br>Delta: {delta:.4f}")
            hover_text.append(hover_row)
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data_reversed,
            x=[f"{i}: {t[:8]}..." if len(t) > 8 else f"{i}: {t}" for i, t in enumerate(tokens)],
            y=[f"L{ln}" for ln in layer_nums_reversed],
            colorscale='Blues',
            hoverinfo='text',
            text=hover_text,
            colorbar=dict(title='Delta', titleside='right')
        ))
        
        fig.update_layout(
            title='Position × Layer Heatmap (Click cell for details)',
            xaxis_title='Token Position',
            yaxis_title='Layer',
            height=max(400, len(layer_nums) * 25 + 100),
            margin=dict(l=60, r=20, t=50, b=80),
            xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
            yaxis=dict(tickfont=dict(size=10))
        )
        
        heatmap_graph = dcc.Graph(
            id='heatmap-graph',
            figure=fig,
            config={'displayModeBar': True, 'scrollZoom': False}
        )
        
        # Return heatmap and toggle visibility
        comparison_style = show_style if show_comparison_toggle else hide_style
        ablation_style = show_style if show_ablation_toggle else hide_style
        
        return heatmap_graph, comparison_style, ablation_style
        
    except Exception as e:
        print(f"Error rendering heatmap: {e}")
        import traceback
        traceback.print_exc()
        return html.P(f"Error rendering heatmap: {str(e)}", className="placeholder-text"), {'display': 'none'}, {'display': 'none'}


# Helper function to create modal content for a specific layer/position (reused from old accordion logic)
def _create_layer_detail_content(activation_data, layer_num, position, model, tokenizer):
    """Create detailed content for a layer at a specific position (for modal display)."""
    import copy
    import plotly.graph_objs as go
    
    # Slice data to the specific position
    def slice_data(data, pos):
        if not data:
            return data
        sliced = copy.deepcopy(data)
        
        if 'block_outputs' in sliced:
            for mod in sliced['block_outputs']:
                out = sliced['block_outputs'][mod]['output']
                if isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
                    if pos < len(out[0]):
                        sliced['block_outputs'][mod]['output'] = [[out[0][pos]]]
        
        if 'attention_outputs' in sliced:
            for mod in sliced['attention_outputs']:
                out = sliced['attention_outputs'][mod]['output']
                if len(out) > 1:
                    attns = out[1]
                    if isinstance(attns, list) and len(attns) > 0:
                        batch_0 = attns[0]
                        new_batch_0 = []
                        for head in batch_0:
                            if pos < len(head):
                                new_batch_0.append([head[pos]])
                        sliced['attention_outputs'][mod]['output'] = [out[0], [new_batch_0]] + out[2:]
        
        if 'input_ids' in sliced:
            ids = sliced['input_ids'][0]
            if pos < len(ids):
                sliced['input_ids'][0] = ids[:pos+1]
        
        return sliced
    
    sliced_data = slice_data(activation_data, position)
    layer_data_list = extract_layer_data(sliced_data, model, tokenizer)
    
    # Find the specific layer
    layer_info = None
    for ld in layer_data_list:
        if ld.get('layer_num') == layer_num:
            layer_info = ld
            break
    
    if not layer_info:
        return html.P("Layer data not found.")
    
    content_items = []
    
    # Top-5 token probabilities bar chart
    top_5 = layer_info.get('top_5_tokens', [])
    if top_5:
        tokens_list = [t[0] for t in top_5]
        probs = [t[1] for t in top_5]
        deltas = layer_info.get('deltas', {})
        
        fig = go.Figure(data=[
            go.Bar(
                x=tokens_list,
                y=probs,
                marker_color='#667eea',
                text=[f"Δ{deltas.get(t, 0):+.3f}" for t in tokens_list],
                textposition='outside'
            )
        ])
        fig.update_layout(
            title=f"Top-5 Token Probabilities at Layer {layer_num}",
            xaxis_title="Token",
            yaxis_title="Probability",
            height=300,
            margin=dict(l=40, r=20, t=40, b=40)
        )
        content_items.append(dcc.Graph(figure=fig, config={'displayModeBar': False}))
    
    # Top attended tokens
    top_attended = layer_info.get('top_attended_tokens', [])
    if top_attended:
        attended_text = ", ".join([f"'{t}' ({w:.3f})" for t, w in top_attended])
        content_items.append(html.Div([
            html.H5("Top Attended Tokens", style={'marginTop': '15px', 'color': '#495057'}),
            html.P(attended_text, style={'fontSize': '14px', 'color': '#6c757d'})
        ]))
    
    # Attention head categories
    try:
        categorized_heads = categorize_single_layer_heads(sliced_data, layer_num)
        if categorized_heads:
            total_heads = sum(len(heads) for heads in categorized_heads.values())
            if total_heads > 0:
                content_items.append(html.Hr(style={'margin': '20px 0'}))
                content_items.append(html.H5(f"Attention Head Categories ({total_heads} heads)", 
                                           style={'marginBottom': '10px', 'color': '#495057'}))
                
                category_colors = {
                    'previous_token': '#ff7979',
                    'first_token': '#74b9ff',
                    'bow': '#ffeaa7',
                    'syntactic': '#a29bfe',
                    'other': '#dfe6e9'
                }
                category_names = {
                    'previous_token': 'Previous-Token',
                    'first_token': 'First/Positional',
                    'bow': 'Bag-of-Words',
                    'syntactic': 'Syntactic',
                    'other': 'Other'
                }
                
                for cat_key, display_name in category_names.items():
                    heads = categorized_heads.get(cat_key, [])
                    if heads:
                        badges = [html.Span(f"H{h['head']}", style={
                            'display': 'inline-block', 'padding': '4px 8px', 'margin': '2px',
                            'backgroundColor': category_colors.get(cat_key, '#dfe6e9'),
                            'borderRadius': '4px', 'fontSize': '11px'
                        }) for h in heads]
                        content_items.append(html.Div([
                            html.Strong(f"{display_name}: ", style={'fontSize': '13px'}),
                            html.Span(badges)
                        ], style={'marginBottom': '8px'}))
    except Exception as e:
        print(f"Warning: Could not categorize heads: {e}")
    
    # Ablation controls placeholder
    num_heads = model.config.num_attention_heads if hasattr(model.config, 'num_attention_heads') else 12
    content_items.append(html.Hr(style={'margin': '20px 0'}))
    content_items.append(html.Div([
        html.H5("Ablation Experiment", style={'marginBottom': '10px', 'color': '#495057'}),
        html.P("Select heads to ablate and observe how predictions change.", 
              style={'fontSize': '13px', 'color': '#6c757d', 'marginBottom': '10px'}),
        html.Div([
            html.Button(f"H{h}", id={'type': 'modal-head-btn', 'layer': layer_num, 'head': h},
                       n_clicks=0, style={
                           'padding': '4px 10px', 'margin': '3px',
                           'backgroundColor': '#f8f9fa', 'border': '1px solid #dee2e6',
                           'borderRadius': '4px', 'cursor': 'pointer', 'fontSize': '12px'
                       }) for h in range(num_heads)
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '10px'}),
        html.Button("Run Ablation", id={'type': 'modal-run-ablation', 'layer': layer_num},
                   className="action-button primary-button", 
                   style={'fontSize': '13px', 'padding': '8px 16px'})
    ]))
    
    return html.Div(content_items)


# Update tokenization display
@app.callback(
    [Output('tokenization-panel', 'style'),
     Output('tokenization-display-container', 'children')],
    [Input('session-activation-store', 'data'),
     Input('session-activation-store-2', 'data')],
    [State('model-dropdown', 'value')]
)
def update_tokenization_display(activation_data, activation_data2, model_name):
    """Populate tokenization panel with actual token data from analysis."""
    if not activation_data or not model_name:
        # Hide panel if no analysis has been run
        return {'display': 'none'}, []
    
    try:
        from transformers import AutoTokenizer
        from components.tokenization_panel import create_tokenization_display
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Decode tokens for first prompt
        input_ids = activation_data['input_ids'][0]  # First batch element
        tokens = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in input_ids]
        
        # Create first tokenization display
        displays = []
        displays.append(html.Div([
            html.H4("Prompt 1:" if activation_data2 else "Your Prompt:", 
                   style={'marginTop': '0', 'color': '#495057'}),
            create_tokenization_display(tokens, input_ids)
        ]))
        
        # Check for comparison mode
        if activation_data2 and activation_data2.get('model') == model_name:
            input_ids2 = activation_data2['input_ids'][0]
            tokens2 = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in input_ids2]
            displays.append(html.Div([
                html.H4("Prompt 2:", 
                       style={'marginTop': '2rem', 'color': '#495057'}),
                create_tokenization_display(tokens2, input_ids2)
            ]))
        
        # Show panel and return displays (stacked vertically)
        return {'display': 'block'}, displays
        
    except Exception as e:
        print(f"Error updating tokenization display: {e}")
        import traceback
        traceback.print_exc()
        return {'display': 'none'}, []


# Heatmap toggle button callbacks
@app.callback(
    [Output('heatmap-mode-store', 'data'),
     Output('heatmap-prompt1-btn', 'style'),
     Output('heatmap-prompt2-btn', 'style'),
     Output('heatmap-original-btn', 'style'),
     Output('heatmap-ablated-btn', 'style')],
    [Input('heatmap-prompt1-btn', 'n_clicks'),
     Input('heatmap-prompt2-btn', 'n_clicks'),
     Input('heatmap-original-btn', 'n_clicks'),
     Input('heatmap-ablated-btn', 'n_clicks')],
    [State('heatmap-mode-store', 'data')],
    prevent_initial_call=True
)
def update_heatmap_mode(p1_clicks, p2_clicks, orig_clicks, abl_clicks, current_mode):
    """Update heatmap mode based on toggle button clicks."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update, no_update, no_update
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    mode = current_mode.copy() if current_mode else {'comparison': 'prompt1', 'ablation': 'original'}
    
    # Comparison toggle styles
    p1_active_style = {'padding': '6px 16px', 'marginRight': '4px', 'border': '1px solid #667eea',
                       'borderRadius': '4px 0 0 4px', 'backgroundColor': '#667eea', 'color': 'white',
                       'cursor': 'pointer', 'fontSize': '13px'}
    p1_inactive_style = {'padding': '6px 16px', 'marginRight': '4px', 'border': '1px solid #667eea',
                         'borderRadius': '4px 0 0 4px', 'backgroundColor': 'white', 'color': '#667eea',
                         'cursor': 'pointer', 'fontSize': '13px'}
    p2_active_style = {'padding': '6px 16px', 'border': '1px solid #667eea',
                       'borderRadius': '0 4px 4px 0', 'backgroundColor': '#667eea', 'color': 'white',
                       'cursor': 'pointer', 'fontSize': '13px'}
    p2_inactive_style = {'padding': '6px 16px', 'border': '1px solid #667eea',
                         'borderRadius': '0 4px 4px 0', 'backgroundColor': 'white', 'color': '#667eea',
                         'cursor': 'pointer', 'fontSize': '13px'}
    
    # Ablation toggle styles
    orig_active_style = {'padding': '6px 16px', 'marginRight': '4px', 'border': '1px solid #28a745',
                         'borderRadius': '4px 0 0 4px', 'backgroundColor': '#28a745', 'color': 'white',
                         'cursor': 'pointer', 'fontSize': '13px'}
    orig_inactive_style = {'padding': '6px 16px', 'marginRight': '4px', 'border': '1px solid #28a745',
                           'borderRadius': '4px 0 0 4px', 'backgroundColor': 'white', 'color': '#28a745',
                           'cursor': 'pointer', 'fontSize': '13px'}
    abl_active_style = {'padding': '6px 16px', 'border': '1px solid #28a745',
                        'borderRadius': '0 4px 4px 0', 'backgroundColor': '#28a745', 'color': 'white',
                        'cursor': 'pointer', 'fontSize': '13px'}
    abl_inactive_style = {'padding': '6px 16px', 'border': '1px solid #28a745',
                          'borderRadius': '0 4px 4px 0', 'backgroundColor': 'white', 'color': '#28a745',
                          'cursor': 'pointer', 'fontSize': '13px'}
    
    # Update mode based on which button was clicked
    if triggered_id == 'heatmap-prompt1-btn':
        mode['comparison'] = 'prompt1'
    elif triggered_id == 'heatmap-prompt2-btn':
        mode['comparison'] = 'prompt2'
    elif triggered_id == 'heatmap-original-btn':
        mode['ablation'] = 'original'
    elif triggered_id == 'heatmap-ablated-btn':
        mode['ablation'] = 'ablated'
    
    # Determine button styles
    p1_style = p1_active_style if mode['comparison'] == 'prompt1' else p1_inactive_style
    p2_style = p2_active_style if mode['comparison'] == 'prompt2' else p2_inactive_style
    orig_style = orig_active_style if mode['ablation'] == 'original' else orig_inactive_style
    abl_style = abl_active_style if mode['ablation'] == 'ablated' else abl_inactive_style
    
    return mode, p1_style, p2_style, orig_style, abl_style


# Heatmap click -> Modal callback
@app.callback(
    [Output('heatmap-modal-overlay', 'style'),
     Output('heatmap-modal-title', 'children'),
     Output('heatmap-modal-content', 'children')],
    [Input('heatmap-graph', 'clickData'),
     Input('heatmap-modal-close', 'n_clicks')],
    [State('session-activation-store', 'data'),
     State('session-activation-store-2', 'data'),
     State('session-activation-store-original', 'data'),
     State('heatmap-mode-store', 'data'),
     State('model-dropdown', 'value')],
    prevent_initial_call=True
)
def handle_heatmap_click(click_data, close_clicks, activation_data, activation_data2, 
                         original_activation_data, mode_data, model_name):
    """Handle clicks on heatmap cells to show modal with layer details."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Modal styles
    hidden_style = {'position': 'fixed', 'top': '0', 'left': '0', 'width': '100%', 'height': '100%',
                    'backgroundColor': 'rgba(0,0,0,0.5)', 'zIndex': '1000', 'display': 'none',
                    'alignItems': 'center', 'justifyContent': 'center'}
    visible_style = {'position': 'fixed', 'top': '0', 'left': '0', 'width': '100%', 'height': '100%',
                     'backgroundColor': 'rgba(0,0,0,0.5)', 'zIndex': '1000', 'display': 'flex',
                     'alignItems': 'center', 'justifyContent': 'center'}
    
    # Handle close button
    if triggered_id == 'heatmap-modal-close':
        return hidden_style, '', ''
    
    # Handle heatmap click
    if triggered_id == 'heatmap-graph' and click_data:
        try:
            point = click_data['points'][0]
            # Extract position and layer from click
            # x is like "0: Hello" -> extract position index
            x_label = point['x']
            position = int(x_label.split(':')[0])
            # y is like "L5" -> extract layer number
            y_label = point['y']
            layer_num = int(y_label[1:])  # Remove 'L' prefix
            
            # Select appropriate data based on mode
            comparison_mode = mode_data.get('comparison', 'prompt1') if mode_data else 'prompt1'
            ablation_mode = mode_data.get('ablation', 'original') if mode_data else 'original'
            
            show_ablation = original_activation_data is not None and activation_data.get('ablated', False)
            
            if show_ablation and ablation_mode == 'original':
                active_data = original_activation_data
            elif activation_data2 and comparison_mode == 'prompt2':
                active_data = activation_data2
            else:
                active_data = activation_data
            
            if not active_data or not model_name:
                return hidden_style, '', html.P("No data available.")
            
            # Load model for detail content
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation='eager')
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Get token at this position
            input_ids = active_data.get('input_ids', [[]])[0]
            token_str = tokenizer.decode([input_ids[position]]) if position < len(input_ids) else f"Position {position}"
            
            title = f"Layer {layer_num} at Position {position}: '{token_str}'"
            content = _create_layer_detail_content(active_data, layer_num, position, model, tokenizer)
            
            return visible_style, title, content
            
        except Exception as e:
            print(f"Error handling heatmap click: {e}")
            import traceback
            traceback.print_exc()
            return hidden_style, '', html.P(f"Error: {str(e)}")
    
    return no_update, no_update, no_update


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
     Output('compare-prompts-btn', 'children'),
     Output('compare-prompts-btn', 'className')],
    [Input('compare-prompts-btn', 'n_clicks')],
    [State('comparison-mode-store', 'data')],
    prevent_initial_call=True
)
def toggle_comparison_mode(n_clicks, is_comparing):
    """Toggle comparison mode and show/hide second prompt input and visualization."""
    if not n_clicks:
        return False, {'display': 'none'}, [html.I(className="fas fa-plus", style={"marginRight": "5px"}), "Compare"], 'compare-button'
    
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
    
    return new_comparing, style, button_content, button_class

# Callback to show full model BertViz visualization
@app.callback(
    Output('full-bertviz-container', 'children'),
    Input('full-bertviz-btn', 'n_clicks'),
    State('session-activation-store', 'data'),
    prevent_initial_call=True
)
def show_full_bertviz(n_clicks, activation_data):
    """Show/hide full model BertViz visualization (all layers, all heads)."""
    if not n_clicks or not activation_data:
        return None
    
    # Toggle: even clicks hide, odd clicks show
    if n_clicks % 2 == 0:
        return None
    
    try:
        from utils import generate_bertviz_html
        
        # Generate full model BertViz HTML (all layers using model_view)
        # Pass layer_num=0 as placeholder (function shows all layers with model_view)
        bertviz_html = generate_bertviz_html(activation_data, 0, 'full')
        
        return html.Div([
            html.Div([
                html.I(className="fas fa-info-circle", style={'marginRight': '8px', 'color': '#764ba2'}),
                html.Span("Interactive visualization of all attention heads across all layers. ", style={'fontSize': '13px', 'color': '#6c757d'}),
                html.Span("Use the controls to explore different layers and heads.", style={'fontSize': '13px', 'color': '#6c757d', 'fontStyle': 'italic'})
            ], style={'marginBottom': '10px', 'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '6px'}),
            html.Iframe(
                srcDoc=bertviz_html,
                style={'width': '100%', 'height': '600px', 'border': '2px solid #764ba2', 'borderRadius': '8px'}
            )
        ])
        
    except Exception as e:
        return html.Div([
            html.I(className="fas fa-exclamation-triangle", style={'marginRight': '8px', 'color': '#dc3545'}),
            f"Error loading BertViz: {str(e)}"
        ], style={'color': '#dc3545', 'fontSize': '13px', 'padding': '10px'})


# Toggle experiments section visibility
@app.callback(
    Output({'type': 'experiments-section', 'layer': MATCH}, 'style'),
    Input({'type': 'explore-button', 'layer': MATCH}, 'n_clicks'),
    prevent_initial_call=True
)
def toggle_experiments_section(n_clicks):
    """Toggle visibility of experiments section."""
    if n_clicks and n_clicks % 2 == 1:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


# Handle head selection (toggle button states)
@app.callback(
    [Output({'type': 'head-select-btn', 'layer': MATCH, 'head': ALL}, 'style'),
     Output({'type': 'selected-heads-store', 'layer': MATCH}, 'data'),
     Output({'type': 'run-ablation-btn', 'layer': MATCH}, 'disabled')],
    Input({'type': 'head-select-btn', 'layer': MATCH, 'head': ALL}, 'n_clicks'),
    State({'type': 'selected-heads-store', 'layer': MATCH}, 'data'),
    prevent_initial_call=True
)
def handle_head_selection(n_clicks_list, selected_heads):
    """Toggle head selection and update Run Ablation button state."""
    if not n_clicks_list:
        return no_update, no_update, no_update
    
    # Initialize selected_heads if None
    if selected_heads is None:
        selected_heads = []
    
    # Determine which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update
    
    triggered_id = ctx.triggered[0]['prop_id']
    
    # Parse the triggered button's head index
    try:
        button_id = json.loads(triggered_id.split('.')[0])
        head_idx = button_id['head']
        
        # Toggle selection
        if head_idx in selected_heads:
            selected_heads.remove(head_idx)
        else:
            selected_heads.append(head_idx)
    except:
        pass
    
    # Update button styles based on selection
    styles = []
    for i, _ in enumerate(n_clicks_list):
        if i in selected_heads:
            # Selected style
            styles.append({
                'padding': '6px 12px',
                'margin': '4px',
                'backgroundColor': '#667eea',
                'color': 'white',
                'border': '1px solid #667eea',
                'borderRadius': '4px',
                'cursor': 'pointer',
                'fontSize': '12px',
                'fontWeight': '600',
                'transition': 'all 0.2s'
            })
        else:
            # Unselected style
            styles.append({
                'padding': '6px 12px',
                'margin': '4px',
                'backgroundColor': '#f8f9fa',
                'color': '#495057',
                'border': '1px solid #dee2e6',
                'borderRadius': '4px',
                'cursor': 'pointer',
                'fontSize': '12px',
                'transition': 'all 0.2s'
            })
    
    # Enable Run Ablation button if at least one head is selected
    run_btn_disabled = len(selected_heads) == 0
    
    return styles, selected_heads, run_btn_disabled


# Run ablation experiment
@app.callback(
    [Output('session-activation-store', 'data', allow_duplicate=True),
     Output('session-activation-store-original', 'data'),
     Output('model-status', 'children', allow_duplicate=True)],
    Input({'type': 'run-ablation-btn', 'layer': ALL}, 'n_clicks'),
    [State({'type': 'selected-heads-store', 'layer': ALL}, 'data'),
     State('session-activation-store', 'data'),
     State('model-dropdown', 'value'),
     State('prompt-input', 'value'),
     State('generation-results-store', 'data')],
    prevent_initial_call=True
)
def run_head_ablation(n_clicks_list, selected_heads_list, activation_data, model_name, prompt_input, generation_results):
    """Run forward pass with selected heads ablated."""
    # Identify which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered or not ctx.triggered_id:
        return no_update, no_update, no_update
    
    # Get the layer number from the triggered button ID
    layer_num = ctx.triggered_id.get('layer')
    if layer_num is None:
        return no_update, no_update, no_update
    
    # Find the index in the states_list that corresponds to this layer
    button_index = None
    if hasattr(ctx, 'states_list') and ctx.states_list:
        # states_list[0] corresponds to selected-heads-store
        for idx, state_info in enumerate(ctx.states_list[0]):
            if state_info['id'].get('layer') == layer_num:
                button_index = idx
                break
    
    # Fallback: if states_list doesn't work, try matching by iterating
    if button_index is None:
        return no_update, no_update, html.Div([
            html.I(className="fas fa-exclamation-circle", style={'marginRight': '8px', 'color': '#dc3545'}),
            f"Could not determine button index for layer {layer_num}"
        ], className="status-error")
    
    # Get the selected heads for this specific button
    selected_heads = None
    if isinstance(selected_heads_list, list) and button_index < len(selected_heads_list):
        selected_heads = selected_heads_list[button_index]
    
    if not selected_heads or not activation_data:
        return no_update, no_update, no_update
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Save original activation data before ablation
        import copy
        original_data = copy.deepcopy(activation_data)
        
        # Determine the sequence to analyze (prefer activation data prompt over input box)
        sequence_text = activation_data.get('prompt', prompt_input)
        
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation='eager')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Get config from original activation data
        config = {
            'attention_modules': activation_data.get('attention_modules', []),
            'block_modules': activation_data.get('block_modules', []),
            'norm_parameters': activation_data.get('norm_parameters', [])
        }
        
        # 1. Run Standard Ablation (Forward Pass)
        ablated_data = execute_forward_pass_with_head_ablation(
            model, tokenizer, sequence_text, config, layer_num, selected_heads
        )
        
        # 2. Compute Full Sequence Metrics (KL Divergence, Delta Probs)
        # This requires re-running passes (Original & Ablated) on the full sequence
        # We use a helper that handles the ablation hooking internally for the metric pass
        seq_metrics = evaluate_sequence_ablation(
            model, tokenizer, sequence_text, config, 
            ablation_type='head', ablation_target=(layer_num, selected_heads)
        )
        ablated_data['sequence_metrics'] = seq_metrics
        
        # 3. Re-score Top Generated Sequences (if available)
        if generation_results:
            top_sequences_comparison = []
            
            # Helper to run ablation for scoring (we need to apply hook again)
            # Since we can't easily pass 'ablated model' around, we re-apply hooks
            # Simplification: We already have 'evaluate_sequence_ablation'.
            # But that compares Ref vs Abl.
            # Here we just want Ablated Score.
            # Actually, `score_sequence` runs valid forward pass. 
            # We need to apply ablation hooks to validly score.
            
            # Define localized hook manager for scoring
            def get_ablated_score(seq_text):
                # Apply hook
                hooks = []
                def head_ablation_hook(module, input, output):
                   # Similar to evaluate_sequence_ablation hook
                    if isinstance(output, tuple): h = output[0]
                    else: h = output
                    if not isinstance(h, torch.Tensor): h = torch.tensor(h)
                    
                    num_heads = model.config.num_attention_heads
                    head_dim = h.shape[-1] // num_heads
                    new_shape = h.shape[:-1] + (num_heads, head_dim)
                    reshaped = h.view(new_shape).clone()
                    for h_idx in selected_heads: reshaped[..., h_idx, :] = 0
                    ablated = reshaped.view(h.shape)
                    return (ablated,) + output[1:] if isinstance(output, tuple) else ablated

                # Register
                target_module = None
                for name, mod in model.named_modules():
                    if f"layers.{layer_num}.self_attn" in name or f"h.{layer_num}.attn" in name:
                        if "k_proj" not in name: 
                            target_module = mod; break
                
                if target_module:
                    hooks.append(target_module.register_forward_hook(head_ablation_hook))
                
                try:
                    score = score_sequence(model, tokenizer, seq_text)
                finally:
                    for hook in hooks: hook.remove()
                return score

            for res in generation_results:
                txt = res['text']
                orig_score = res['score']
                new_score = get_ablated_score(txt)
                top_sequences_comparison.append({
                    'text': txt,
                    'original_score': orig_score,
                    'ablated_score': new_score,
                    'delta': new_score - orig_score
                })
            
            ablated_data['top_sequences_comparison'] = top_sequences_comparison

        
        # Update activation data with ablated results
        ablated_data['ablated'] = True
        ablated_data['ablated_layer'] = layer_num
        ablated_data['ablated_heads'] = selected_heads
        
        # Preserve input_ids if needed
        if 'input_ids' not in ablated_data and 'input_ids' in activation_data:
            ablated_data['input_ids'] = activation_data['input_ids']
        
        # Success message
        heads_str = ', '.join([f"H{h}" for h in sorted(selected_heads)])
        success_message = html.Div([
            html.I(className="fas fa-check-circle", style={'marginRight': '8px', 'color': '#28a745'}),
            f"Ablation complete: Layer {layer_num}, Heads {heads_str} removed. Scroll down for sequence analysis."
        ], className="status-success")
        
        return ablated_data, original_data, success_message
        
    except Exception as e:
        print(f"Error running ablation: {e}")
        import traceback
        traceback.print_exc()
        
        error_message = html.Div([
            html.I(className="fas fa-exclamation-circle", style={'marginRight': '8px', 'color': '#dc3545'}),
            f"Ablation error: {str(e)}"
        ], className="status-error")
        
        return no_update, no_update, error_message


# Show/hide Reset Ablation button based on ablation mode
@app.callback(
    Output('reset-ablation-container', 'style'),
    Input('session-activation-store', 'data'),
    prevent_initial_call=False
)
def toggle_reset_ablation_button(activation_data):
    """Show Reset Ablation button when in ablation mode, hide otherwise."""
    if activation_data and activation_data.get('ablated', False):
        return {'display': 'block'}
    else:
        return {'display': 'none'}


# Reset ablation experiment
@app.callback(
    [Output('session-activation-store', 'data', allow_duplicate=True),
     Output('session-activation-store-original', 'data', allow_duplicate=True),
     Output('model-status', 'children', allow_duplicate=True)],
    Input('reset-ablation-btn', 'n_clicks'),
    [State('session-activation-store-original', 'data')],
    prevent_initial_call=True
)
def reset_ablation(n_clicks, original_data):
    """Reset ablation by restoring original data and clearing the original store."""
    if not n_clicks:
        return no_update, no_update, no_update
    
    if not original_data:
        error_message = html.Div([
            html.I(className="fas fa-exclamation-circle", style={'marginRight': '8px', 'color': '#dc3545'}),
            "No original data to restore"
        ], className="status-error")
        return no_update, no_update, error_message
    
    # Restore original data to main store and clear original store
    success_message = html.Div([
        html.I(className="fas fa-undo", style={'marginRight': '8px', 'color': '#28a745'}),
        "Ablation reset - original data restored"
    ], className="status-success")
    
    return original_data, {}, success_message



# Callback to update sequence ablation analysis view
@app.callback(
    [Output('sequence-ablation-results-container', 'children'),
     Output('sequence-ablation-results-container', 'style')],
    Input('session-activation-store', 'data'),
    prevent_initial_call=False
)
def update_sequence_ablation_view(activation_data):
    """Update the sequence ablation results view (KL Divergence, Sequence Comparison)."""
    if not activation_data or not activation_data.get('ablated', False):
        return [], {'display': 'none'}
    
    try:
        import plotly.graph_objs as go
        from dash import html, dcc
        
        children = []
        
        # 1. Header
        children.append(html.H3("Full Sequence Ablation Analysis", style={'marginTop': '0', 'marginBottom': '20px', 'color': '#2d3748'}))
        
        # 2. Top-5 Sequence Comparison Table
        top_seqs = activation_data.get('top_sequences_comparison', [])
        if top_seqs:
            rows = []
            for i, seq in enumerate(top_seqs):
                delta = seq['delta']
                delta_color = '#28a745' if delta > 0 else '#dc3545' if delta < 0 else '#6c757d'
                
                rows.append(html.Tr([
                    html.Td(f"#{i+1}", style={'fontWeight': 'bold'}),
                    html.Td(seq['text'], style={'fontFamily': 'monospace', 'maxWidth': '400px', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'}),
                    html.Td(f"{seq['original_score']:.4f}"),
                    html.Td(f"{seq['ablated_score']:.4f}"),
                    html.Td(f"{delta:+.4f}", style={'color': delta_color, 'fontWeight': 'bold'})
                ]))
            
            table_header = html.Thead(html.Tr([
                html.Th("Rank"), html.Th("Sequence"), html.Th("Original Score"), html.Th("Ablated Score"), html.Th("Delta")
            ]))
            table_body = html.Tbody(rows)
            
            children.append(html.Div([
                html.H5("Top Sequences Impact", style={'marginBottom': '10px'}),
                html.Table([table_header, table_body], className="table table-striped table-bordered")
            ], style={'marginBottom': '30px', 'padding': '15px', 'backgroundColor': '#fff', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.05)'}))
        
        # 3. KL Divergence Chart
        seq_metrics = activation_data.get('sequence_metrics', {})
        kl_divs = seq_metrics.get('kl_divergence', [])
        tokens = seq_metrics.get('tokens', [])
        
        if kl_divs:
            # KL Chart
            fig_kl = go.Figure()
            fig_kl.add_trace(go.Scatter(
                x=list(range(len(kl_divs))),
                y=kl_divs,
                mode='lines+markers',
                name='KL Divergence',
                line=dict(color='#6610f2', width=2),
                hovertext=[f"Token: {t}<br>KL: {v:.4f}" for t, v in zip(tokens, kl_divs)],
                hoverinfo='text'
            ))
            
            fig_kl.update_layout(
                title="KL Divergence per Position (Distribution Shift)",
                xaxis_title="Position / Token",
                yaxis_title="KL Divergence (nats)",
                margin=dict(l=20, r=20, t=40, b=20),
                height=300,
                xaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(tokens))),
                    ticktext=tokens
                )
            )
            
            children.append(html.Div([
                dcc.Graph(figure=fig_kl, config={'displayModeBar': False})
            ], style={'marginBottom': '20px', 'padding': '15px', 'backgroundColor': '#fff', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.05)'}))
        
        # 4. Target Probability Deltas Chart
        prob_deltas = seq_metrics.get('probability_deltas', [])
        if prob_deltas:
            # Shift tokens for x-axis (deltas are for prediction of next token)
            # Input: T0, T1, T2
            # Delta 0: Change in P(T1|T0)
            # So x-axis should be T1, T2...
            target_tokens = tokens[1:] if len(tokens) > 1 else []
            
            fig_delta = go.Figure()
            fig_delta.add_trace(go.Bar(
                x=list(range(len(prob_deltas))),
                y=prob_deltas,
                name='Prob Delta',
                marker_color=['#28a745' if v >= 0 else '#dc3545' for v in prob_deltas],
                hovertext=[f"Target: {t}<br>Change: {v:+.4f}" for t, v in zip(target_tokens, prob_deltas)],
                hoverinfo='text'
            ))
            
            fig_delta.update_layout(
                title="Target Probability Change per Position",
                xaxis_title="Target Token",
                yaxis_title="Probability Delta",
                margin=dict(l=20, r=20, t=40, b=20),
                height=300,
                xaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(target_tokens))),
                    ticktext=target_tokens
                )
            )
            
            children.append(html.Div([
                dcc.Graph(figure=fig_delta, config={'displayModeBar': False})
            ], style={'marginBottom': '20px', 'padding': '15px', 'backgroundColor': '#fff', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.05)'}))
        
        return children, {'display': 'block', 'marginTop': '30px', 'paddingTop': '30px', 'borderTop': '1px solid #dee2e6'}
        
    except Exception as e:
        print(f"Error in ablation view: {e}")
        import traceback
        traceback.print_exc()
        return html.Div(f"Error loading visualization: {str(e)}"), {'display': 'block'}


if __name__ == '__main__':
    app.run(debug=True, port=8050)
