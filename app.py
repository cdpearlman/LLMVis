"""
Modular Dash Cytoscape Visualization Dashboard

A simple, modular dashboard for transformer model visualization using Dash and Cytoscape.
Components are organized for easy understanding and maintenance.
"""

import dash
from dash import html, dcc, Input, Output, State, callback, no_update, ALL
from utils import (load_model_and_get_patterns, execute_forward_pass, extract_layer_data,
                   categorize_single_layer_heads, format_categorization_summary,
                   compare_attention_layers, compare_output_probabilities, format_comparison_summary,
                   get_check_token_probabilities, execute_forward_pass_with_layer_ablation)
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
    # Check token graph data
    dcc.Store(id='check-token-graph-store', storage_type='memory'),
    # Ablation experiment stores
    dcc.Store(id='ablation-selection-store', storage_type='session'),
    dcc.Store(id='ablation-results-flag', storage_type='session', data=False),
    
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

# Callback to run analysis
@app.callback(
    [Output('session-activation-store', 'data', allow_duplicate=True),
     Output('session-activation-store-2', 'data'),
     Output('analysis-loading-indicator', 'children'),
     Output('comparison-section', 'style'),
     Output('comparison-container', 'children'),
     Output('check-token-graph-store', 'data')],
    [Input('run-analysis-btn', 'n_clicks')],
    [State('model-dropdown', 'value'),
     State('prompt-input', 'value'),
     State('prompt-input-2', 'value'),
     State('check-token-input', 'value'),
     State('attention-modules-dropdown', 'value'),
     State('block-modules-dropdown', 'value'),
     State('norm-params-dropdown', 'value'), 
     State('logit-lens-dropdown', 'value'),
     State('session-patterns-store', 'data')],
    prevent_initial_call=True
)
def run_analysis(n_clicks, model_name, prompt, prompt2, check_token, attn_patterns, block_patterns, norm_patterns, logit_pattern, patterns_data):
    """Run forward pass and store activation data (handles 1 or 2 prompts)."""
    print(f"\n=== DEBUG: run_analysis START ===")
    print(f"DEBUG: n_clicks={n_clicks}, model_name={model_name}, prompt='{prompt}', prompt2='{prompt2}'")
    print(f"DEBUG: block_patterns={block_patterns}")
    print(f"DEBUG: logit_pattern={logit_pattern}")
    
    if not n_clicks or not model_name or not prompt or not block_patterns:
        print("DEBUG: Missing required inputs, returning empty")
        comparison_placeholder = html.P("Comparison analysis will appear here when two prompts are provided.", className="placeholder-text")
        return {}, {}, None, {'display': 'none'}, comparison_placeholder, None
    
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
        
        print(f"DEBUG: Executed forward pass for prompt 1")
        
        # Compute check token probabilities if provided
        check_token_data = get_check_token_probabilities(activation_data, model, tokenizer, check_token) if check_token else None
        
        # Store data needed for accordion display and analysis
        essential_data = {
            'model': model_name,
            'prompt': prompt,
            'attention_outputs': activation_data.get('attention_outputs', {}),
            'input_ids': activation_data.get('input_ids', []),
            'block_modules': activation_data.get('block_modules', []),
            'block_outputs': activation_data.get('block_outputs', {}),
            'logit_lens_parameter': activation_data.get('logit_lens_parameter'),
            'norm_parameters': activation_data.get('norm_parameters', [])
        }
        
        # Process second prompt if provided
        essential_data2 = {}
        comparison_style = {'display': 'none'}  # Default: hide comparison
        comparison_display = html.P("Comparison analysis will appear here when two prompts are provided.", className="placeholder-text")
        
        if prompt2 and prompt2.strip():
            activation_data2 = execute_forward_pass(model, tokenizer, prompt2, config)
            print(f"DEBUG: Executed forward pass for prompt 2")
            
            essential_data2 = {
                'model': model_name,
                'prompt': prompt2,
                'attention_outputs': activation_data2.get('attention_outputs', {}),
                'input_ids': activation_data2.get('input_ids', []),
                'block_modules': activation_data2.get('block_modules', []),
                'block_outputs': activation_data2.get('block_outputs', {}),
                'logit_lens_parameter': activation_data2.get('logit_lens_parameter'),
                'norm_parameters': activation_data2.get('norm_parameters', [])
            }
            
            # Compute comparison between two prompts
            comparison_results = compare_attention_layers(activation_data, activation_data2)
            prob_differences = compare_output_probabilities(activation_data, activation_data2, model, tokenizer)
            comparison_summary = format_comparison_summary(comparison_results, prob_differences)
            
            # Get divergent layer numbers for highlighting
            divergent_layer_nums = set(ld['layer'] for ld in comparison_results['divergent_layers'])
            
            # Create comparison display
            comparison_display = html.Div([
                html.Pre(comparison_summary, style={'whiteSpace': 'pre-wrap', 'fontFamily': 'monospace', 'fontSize': '13px'})
            ])
            
            # Show comparison section
            comparison_style = {'display': 'block'}
        
        # Show success message
        success_message = html.Div([
            html.I(className="fas fa-check-circle", style={'color': '#28a745', 'marginRight': '8px'}),
            "Analysis completed successfully!" + (" (2 prompts)" if prompt2 and prompt2.strip() else "")
        ], className="status-success")
        
        print(f"=== DEBUG: run_analysis END ===\n")
        return essential_data, essential_data2, success_message, comparison_style, comparison_display, check_token_data
        
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
        return {}, {}, error_message, {'display': 'none'}, comparison_placeholder, None

# Callback to update check token graph
@app.callback(
    [Output('check-token-graph', 'figure'),
     Output('check-token-graph-container', 'style')],
    [Input('check-token-graph-store', 'data')]
)
def update_check_token_graph(check_token_data):
    """Update the check token probability graph."""
    if not check_token_data:
        return {}, {'flex': '1', 'minWidth': '300px', 'display': 'none'}
    
    import plotly.graph_objs as go
    
    figure = go.Figure(
        data=[go.Scatter(
            x=check_token_data['layers'],
            y=check_token_data['probabilities'],
            mode='lines+markers',
            line={'color': '#667eea', 'width': 2},
            marker={'size': 6},
            name=check_token_data['token']
        )],
        layout=go.Layout(
            title=f"Token: {check_token_data['token']}",
            xaxis={'title': 'Layer'},
            yaxis={'title': 'Probability', 'nticks': 8}
        )
    )
    
    return figure, {'flex': '1', 'minWidth': '300px', 'display': 'block'}

def _create_single_prompt_chart(layer_data):
    """
    Create a single prompt bar chart (existing functionality).
    
    Args:
        layer_data: Layer data dict (with top_5_tokens, deltas, certainty)
    
    Returns:
        Plotly Figure with horizontal bars
    """
    import plotly.graph_objs as go
    
    top_5 = layer_data.get('top_5_tokens', [])
    deltas = layer_data.get('deltas', {})
    certainty = layer_data.get('certainty', 0.0)
    
    if not top_5:
        return go.Figure()
    
    tokens = [tok for tok, _ in top_5]
    probs = [prob for _, prob in top_5]
    
    # Create delta annotations (▲/▼ with color)
    annotations = []
    for idx, (token, prob) in enumerate(top_5):
        delta = deltas.get(token, 0.0)
        if abs(delta) > 0.001:  # Only show meaningful deltas
            symbol = '▲' if delta > 0 else '▼'
            color = '#28a745' if delta > 0 else '#dc3545'
            annotations.append({
                'x': prob,
                'y': idx,
                'text': f'{symbol} {abs(delta):.3f}',
                'showarrow': False,
                'xanchor': 'left',
                'xshift': 10,
                'font': {'size': 10, 'color': color}
            })
    
    # Create Plotly figure
    fig = go.Figure(data=[
        go.Bar(
            x=probs,
            y=tokens,
            orientation='h',
            marker={'color': '#667eea'},
            text=[f'{p:.3f}' for p in probs],
            textposition='auto',
            hovertemplate='%{y}: %{x:.4f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': f'Top 5 Predictions (Certainty: {certainty:.2f})',
            'font': {'size': 14}
        },
        xaxis={'title': 'Probability', 'range': [0, max(probs) * 1.15]},
        yaxis={'title': '', 'autorange': 'reversed'},
        height=250,
        margin={'l': 100, 'r': 80, 't': 50, 'b': 40},
        annotations=annotations,
        hovermode='closest'
    )
    
    return fig


def _create_comparison_bar_chart(layer_data1, layer_data2, layer_num):
    """
    Create a grouped bar chart comparing top-5 predictions from two prompts.
    
    Args:
        layer_data1: Layer data dict for prompt 1 (with top_5_tokens, deltas, certainty)
        layer_data2: Layer data dict for prompt 2 (with top_5_tokens, deltas, certainty)
        layer_num: Layer number for title
    
    Returns:
        Plotly Figure with grouped bars for overlapping tokens and separate bars for non-overlapping
    """
    import plotly.graph_objs as go
    
    top_5_1 = layer_data1.get('top_5_tokens', [])
    top_5_2 = layer_data2.get('top_5_tokens', [])
    deltas_1 = layer_data1.get('deltas', {})
    deltas_2 = layer_data2.get('deltas', {})
    certainty_1 = layer_data1.get('certainty', 0.0)
    certainty_2 = layer_data2.get('certainty', 0.0)
    
    # Build token sets
    tokens_1 = {tok: prob for tok, prob in top_5_1}
    tokens_2 = {tok: prob for tok, prob in top_5_2}
    
    all_tokens = set(tokens_1.keys()) | set(tokens_2.keys())
    overlapping_tokens = set(tokens_1.keys()) & set(tokens_2.keys())
    
    # Sort tokens: overlapping first (by max prob), then unique to prompt 1, then unique to prompt 2
    def token_sort_key(token):
        if token in overlapping_tokens:
            return (0, -max(tokens_1.get(token, 0), tokens_2.get(token, 0)))
        elif token in tokens_1:
            return (1, -tokens_1[token])
        else:
            return (2, -tokens_2[token])
    
    sorted_tokens = sorted(all_tokens, key=token_sort_key)
    
    # Prepare data for grouped bars
    tokens_list = []
    probs_1_list = []
    probs_2_list = []
    
    for token in sorted_tokens:
        tokens_list.append(token)
        probs_1_list.append(tokens_1.get(token, 0))
        probs_2_list.append(tokens_2.get(token, 0))
    
    # Create annotations for deltas
    annotations = []
    for idx, token in enumerate(tokens_list):
        # Prompt 1 delta (if token exists in prompt 1)
        if token in tokens_1:
            delta = deltas_1.get(token, 0.0)
            if abs(delta) > 0.001:
                symbol = '▲' if delta > 0 else '▼'
                color = '#28a745' if delta > 0 else '#dc3545'
                annotations.append({
                    'x': tokens_1[token],
                    'y': idx - 0.2,  # Offset for prompt 1 bar
                    'text': f'{symbol}{abs(delta):.3f}',
                    'showarrow': False,
                    'xanchor': 'left',
                    'xshift': 10,
                    'font': {'size': 9, 'color': color}
                })
        
        # Prompt 2 delta (if token exists in prompt 2)
        if token in tokens_2:
            delta = deltas_2.get(token, 0.0)
            if abs(delta) > 0.001:
                symbol = '▲' if delta > 0 else '▼'
                color = '#28a745' if delta > 0 else '#dc3545'
                annotations.append({
                    'x': tokens_2[token],
                    'y': idx + 0.2,  # Offset for prompt 2 bar
                    'text': f'{symbol}{abs(delta):.3f}',
                    'showarrow': False,
                    'xanchor': 'left',
                    'xshift': 10,
                    'font': {'size': 9, 'color': color}
                })
    
    # Create figure with grouped bars
    fig = go.Figure()
    
    # Add Prompt 1 bars
    fig.add_trace(go.Bar(
        name='Prompt 1',
        x=probs_1_list,
        y=tokens_list,
        orientation='h',
        marker={'color': '#667eea'},
        text=[f'{p:.3f}' if p > 0 else '' for p in probs_1_list],
        textposition='auto',
        hovertemplate='Prompt 1 - %{y}: %{x:.4f}<extra></extra>'
    ))
    
    # Add Prompt 2 bars
    fig.add_trace(go.Bar(
        name='Prompt 2',
        x=probs_2_list,
        y=tokens_list,
        orientation='h',
        marker={'color': '#f59e42'},
        text=[f'{p:.3f}' if p > 0 else '' for p in probs_2_list],
        textposition='auto',
        hovertemplate='Prompt 2 - %{y}: %{x:.4f}<extra></extra>'
    ))
    
    # Update layout
    max_prob = max(max(probs_1_list + [0]), max(probs_2_list + [0]))
    
    fig.update_layout(
        title={
            'text': f'Top 5 Predictions Comparison (Certainty: P1={certainty_1:.2f}, P2={certainty_2:.2f})',
            'font': {'size': 14}
        },
        xaxis={'title': 'Probability', 'range': [0, max_prob * 1.2]},
        yaxis={'title': '', 'autorange': 'reversed'},
        barmode='group',
        height=300,
        margin={'l': 100, 'r': 100, 't': 50, 'b': 40},
        annotations=annotations,
        hovermode='closest',
        legend={
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'right',
            'x': 1
        }
    )
    
    return fig


# Callback to create accordion panels from layer data
@app.callback(
    Output('layer-accordions-container', 'children'),
    [Input('session-activation-store', 'data'),
     Input('session-activation-store-2', 'data')],
    [State('model-dropdown', 'value')]
)
def create_layer_accordions(activation_data, activation_data2, model_name):
    """Create accordion panels for each layer with top-5 bar charts, deltas, and certainty."""
    if not activation_data or not model_name:
        return html.P("Run analysis to see layer-by-layer predictions.", className="placeholder-text")
    
    # Check if this is ablation results
    ablated_layer = activation_data.get('ablated_layer')
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import plotly.graph_objs as go
        
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation='eager')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Extract layer data for first prompt
        layer_data = extract_layer_data(activation_data, model, tokenizer)
        
        if not layer_data:
            return html.P("No layer data available.", className="placeholder-text")
        
        # Check if second prompt exists and extract its layer data
        layer_data2 = None
        divergent_layers = set()
        comparison_mode = activation_data2 and activation_data2.get('model') == model_name
        
        if comparison_mode:
            layer_data2 = extract_layer_data(activation_data2, model, tokenizer)
            
            # Compute divergence between prompts
            comparison_results = compare_attention_layers(activation_data, activation_data2)
            divergent_layers = set(ld['layer'] for ld in comparison_results['divergent_layers'])
        
        # Create accordion panels (reversed to show final layer first)
        accordions = []
        for i, layer in enumerate(reversed(layer_data)):
            layer_num = layer['layer_num']
            top_token = layer.get('top_token', 'N/A')
            top_prob = layer.get('top_prob', 0.0)
            top_5 = layer.get('top_5_tokens', [])
            deltas = layer.get('deltas', {})
            certainty = layer.get('certainty', 0.0)
            
            # Create summary header - different format for comparison mode
            if comparison_mode and layer_data2:
                # Find corresponding layer in second prompt
                layer2 = next((l for l in layer_data2 if l['layer_num'] == layer_num), None)
                if layer2:
                    top_token2 = layer2.get('top_token', 'N/A')
                    top_prob2 = layer2.get('top_prob', 0.0)
                    
                    # Determine if layers diverge
                    is_divergent = layer_num in divergent_layers
                    status = "diverges" if is_divergent else "similar"
                    
                    if top_token and top_token2:
                        summary_text = f"Layer L{layer_num}: '{top_token}' vs '{top_token2}' ({status})"
                    elif top_token:
                        summary_text = f"Layer L{layer_num}: '{top_token}' vs (no prediction) ({status})"
                    elif top_token2:
                        summary_text = f"Layer L{layer_num}: (no prediction) vs '{top_token2}' ({status})"
                    else:
                        summary_text = f"Layer L{layer_num}: (no prediction) vs (no prediction) ({status})"
                else:
                    summary_text = f"Layer L{layer_num}: '{top_token}' vs (no data) (diverges)"
            else:
                # Single prompt mode
                if top_token:
                    summary_text = f"Layer L{layer_num}: '{top_token}' (p={top_prob:.3f}, certainty={certainty:.2f})"
                else:
                    summary_text = f"Layer L{layer_num}: (no prediction)"
            
            # Create accordion panel content
            content_items = []
            
            if top_5:
                # Create chart - either single prompt or comparison
                if comparison_mode and layer_data2:
                    # Find corresponding layer in second prompt
                    layer2 = next((l for l in layer_data2 if l['layer_num'] == layer_num), None)
                    if layer2 and layer2.get('top_5_tokens'):
                        # Use comparison chart
                        fig = _create_comparison_bar_chart(layer, layer2, layer_num)
                    else:
                        # Fallback to single prompt chart if no second prompt data
                        fig = _create_single_prompt_chart(layer)
                else:
                    # Single prompt mode
                    fig = _create_single_prompt_chart(layer)
                
                content_items.append(
                    dcc.Graph(
                        figure=fig,
                        config={'displayModeBar': False},
                        style={'marginBottom': '10px'}
                    )
                )
                
                # Add certainty tooltip explanation
                if comparison_mode and layer_data2:
                    layer2 = next((l for l in layer_data2 if l['layer_num'] == layer_num), None)
                    if layer2:
                        certainty2 = layer2.get('certainty', 0.0)
                        content_items.append(html.Div([
                            html.Small([
                                html.I(className="fas fa-info-circle", style={'marginRight': '5px', 'color': '#667eea'}),
                                f"Certainty: P1={certainty:.2f}, P2={certainty2:.2f}. ",
                                "Higher values indicate more confident predictions."
                            ], style={'color': '#6c757d', 'fontStyle': 'italic'})
                        ], style={'marginTop': '5px'}))
                    else:
                        content_items.append(html.Div([
                            html.Small([
                                html.I(className="fas fa-info-circle", style={'marginRight': '5px', 'color': '#667eea'}),
                                f"Certainty = 1 − H(p_top5)/log(5), where H is Shannon entropy. ",
                                "Higher values indicate more confident predictions."
                            ], style={'color': '#6c757d', 'fontStyle': 'italic'})
                        ], style={'marginTop': '5px'}))
                else:
                    content_items.append(html.Div([
                        html.Small([
                            html.I(className="fas fa-info-circle", style={'marginRight': '5px', 'color': '#667eea'}),
                            f"Certainty = 1 − H(p_top5)/log(5), where H is Shannon entropy. ",
                            "Higher values indicate more confident predictions."
                        ], style={'color': '#6c757d', 'fontStyle': 'italic'})
                    ], style={'marginTop': '5px'}))
            else:
                content_items.append(html.P("No predictions available"))
            
            # Add attention head categorization section
            top_attended = layer.get('top_attended_tokens', [])
            
            # Always show attention section
            content_items.append(html.Hr(style={'margin': '15px 0'}))
            
            if top_attended:
                # Categorize attention heads for this layer
                total_heads = 0
                try:
                    from utils import generate_category_bertviz_html
                    
                    categorized_heads = categorize_single_layer_heads(activation_data, layer_num)
                    total_heads = sum(len(heads) for heads in categorized_heads.values())
                    
                    if total_heads > 0:
                        # Display head categorization with explanation
                        content_items.append(html.Div([
                            html.H6("Attention Head Categories", style={'marginBottom': '4px', 'fontSize': '14px', 'color': '#495057', 'display': 'inline-block'}),
                            html.Small([
                                html.I(className="fas fa-info-circle", style={'marginLeft': '8px', 'marginRight': '4px', 'color': '#667eea'}),
                                f"({total_heads} heads total)"
                            ], style={'color': '#6c757d', 'fontSize': '11px'})
                        ], style={'marginBottom': '8px'}))
                        
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
                        
                        # Create expandable category sections with BertViz visualizations
                        for cat_key, display_name in category_names.items():
                            heads = categorized_heads.get(cat_key, [])
                            if heads:
                                color = category_colors.get(cat_key, '#dfe6e9')
                                
                                # Generate BertViz visualization for this category
                                bertviz_html = generate_category_bertviz_html(activation_data, heads)
                                
                                # Create collapsible category section
                                category_section = html.Details([
                                    html.Summary([
                                        html.Span([
                                            html.Strong(f"{display_name}: "),
                                            f"{len(heads)} heads"
                                        ], style={
                                            'display': 'inline-block',
                                            'padding': '4px 10px',
                                            'backgroundColor': color,
                                            'borderRadius': '4px',
                                            'fontSize': '12px',
                                            'fontWeight': '500',
                                            'color': '#2d3748'
                                        })
                                    ], style={'cursor': 'pointer', 'padding': '4px 0'}),
                                    html.Div([
                                        html.Iframe(
                                            srcDoc=bertviz_html,
                                            style={
                                                'width': '100%',
                                                'height': '400px',
                                                'border': '1px solid #ddd',
                                                'borderRadius': '4px',
                                                'marginTop': '10px'
                                            }
                                        )
                                    ])
                                ], style={'marginBottom': '8px'})
                                
                                content_items.append(category_section)
                        
                except Exception as e:
                    print(f"Warning: Could not categorize heads for layer {layer_num}: {e}")
                    import traceback
                    traceback.print_exc()
                
                content_items.append(html.Hr(style={'margin': '10px 0'}))
            
            content_items.append(html.H6("Attention (current position)", style={'marginBottom': '8px', 'fontSize': '14px', 'color': '#495057'}))
            
            if top_attended:
                # Show top-3 attended tokens
                attention_items = []
                for token, weight in top_attended:
                    attention_items.append(html.Div([
                        html.Span(f"'{token}'", style={'fontWeight': 'bold', 'marginRight': '8px'}),
                        html.Span(f"{weight:.3f}", style={'color': '#6c757d', 'fontSize': '13px'})
                    ], style={'marginBottom': '4px'}))
                
                content_items.append(html.Div(attention_items, style={'marginBottom': '10px'}))
                
                # Add button to open full BertViz view
                button_text = f"View all {total_heads} heads interactively (BertViz)" if total_heads > 0 else "View attention heads interactively (BertViz)"
                content_items.append(html.Button(
                    [html.I(className="fas fa-chart-network", style={'marginRight': '5px'}), button_text],
                    id={'type': 'bertviz-btn', 'layer': layer_num},
                    className='bertviz-button',
                    title="Opens an interactive visualization showing attention patterns for all heads in this layer",
                    style={
                        'padding': '6px 12px',
                        'fontSize': '12px',
                        'backgroundColor': '#667eea',
                        'color': 'white',
                        'border': 'none',
                        'borderRadius': '4px',
                        'cursor': 'pointer',
                        'transition': 'background-color 0.2s'
                    }
                ))
                content_items.append(html.Div(id={'type': 'bertviz-container', 'layer': layer_num}, style={'marginTop': '10px'}))
            else:
                # Show message when no attention data is available
                content_items.append(html.Div([
                    html.Small([
                        html.I(className="fas fa-info-circle", style={'marginRight': '5px', 'color': '#f0ad4e'}),
                        "No attention data available. Select attention modules in the sidebar to see attention patterns."
                    ], style={'color': '#6c757d', 'fontStyle': 'italic'})
                ]))
            
            # Determine if this layer is ablated
            panel_class = "layer-accordion"
            if ablated_layer is not None and layer_num == ablated_layer:
                panel_class += " layer-ablated"
            
            panel = html.Details([
                html.Summary(summary_text, className="layer-summary"),
                html.Div(content_items, className="layer-content")
            ], className=panel_class)
            
            accordions.append(panel)
        
        return html.Div(accordions)
        
    except Exception as e:
        print(f"Error creating accordions: {e}")
        import traceback
        traceback.print_exc()
        return html.P(f"Error creating layer view: {str(e)}", className="placeholder-text")

# Show experiments section after analysis completes
@app.callback(
    Output('experiments-section', 'style'),
    [Input('session-activation-store', 'data')],
    prevent_initial_call=False
)
def show_experiments_section(activation_data):
    """Show experiments section after analysis is run."""
    if activation_data and activation_data.get('block_modules') and len(activation_data.get('block_modules', [])) > 0:
        return {'display': 'block'}
    return {'display': 'none'}

# Populate ablation layer buttons
@app.callback(
    Output('ablation-layer-buttons', 'children'),
    [Input('session-activation-store', 'data')]
)
def populate_ablation_buttons(activation_data):
    """Create layer buttons for ablation experiment."""
    if not activation_data or not activation_data.get('block_modules'):
        return html.P("Run analysis first to select layers.", style={'color': '#6c757d', 'fontSize': '14px'})
    
    # Extract layer numbers from block modules
    import re
    layer_modules = activation_data.get('block_modules', [])
    layer_numbers = sorted([
        int(re.findall(r'\d+', name)[0])
        for name in layer_modules if re.findall(r'\d+', name)
    ])
    
    if not layer_numbers:
        return html.P("No layers found.", style={'color': '#6c757d', 'fontSize': '14px'})
    
    # Create buttons for each layer
    buttons = []
    for layer_num in layer_numbers:
        buttons.append(
            html.Button(
                f"L{layer_num}",
                id={'type': 'ablate-layer-btn', 'layer': layer_num},
                className='ablation-layer-btn',
                n_clicks=0
            )
        )
    
    return buttons

# Handle layer button selection
@app.callback(
    [Output('ablation-selection-store', 'data'),
     Output({'type': 'ablate-layer-btn', 'layer': ALL}, 'className')],
    [Input({'type': 'ablate-layer-btn', 'layer': ALL}, 'n_clicks')],
    [State('ablation-selection-store', 'data'),
     State({'type': 'ablate-layer-btn', 'layer': ALL}, 'id')]
)
def handle_layer_selection(n_clicks_list, current_selection, button_ids):
    """Handle layer button clicks and update selection state."""
    from dash import callback_context
    
    if not callback_context.triggered or not button_ids:
        # Initial render or no buttons
        class_names = ['ablation-layer-btn'] * len(button_ids)
        if current_selection is not None:
            # Apply selected style to current selection
            for i, btn_id in enumerate(button_ids):
                if btn_id['layer'] == current_selection:
                    class_names[i] = 'ablation-layer-btn selected'
        return current_selection, class_names
    
    # Find which button was clicked
    triggered_id = callback_context.triggered[0]['prop_id']
    if 'ablate-layer-btn' not in triggered_id:
        # Not a button click
        class_names = ['ablation-layer-btn'] * len(button_ids)
        if current_selection is not None:
            for i, btn_id in enumerate(button_ids):
                if btn_id['layer'] == current_selection:
                    class_names[i] = 'ablation-layer-btn selected'
        return current_selection, class_names
    
    # Parse the triggered button's layer
    import json
    triggered_dict = json.loads(triggered_id.split('.')[0])
    clicked_layer = triggered_dict['layer']
    
    # Toggle selection: if already selected, deselect; otherwise select
    new_selection = None if current_selection == clicked_layer else clicked_layer
    
    # Update button classes
    class_names = []
    for btn_id in button_ids:
        if btn_id['layer'] == new_selection:
            class_names.append('ablation-layer-btn selected')
        else:
            class_names.append('ablation-layer-btn')
    
    return new_selection, class_names

# Run ablation experiment callback
@app.callback(
    [Output('session-activation-store', 'data', allow_duplicate=True),
     Output('session-activation-store-2', 'data', allow_duplicate=True),
     Output('ablation-results-flag', 'data'),
     Output('analysis-loading-indicator', 'children', allow_duplicate=True)],
    [Input('run-ablation-btn', 'n_clicks')],
    [State('ablation-selection-store', 'data'),
     State('session-activation-store', 'data'),
     State('session-activation-store-2', 'data'),
     State('model-dropdown', 'value'),
     State('prompt-input', 'value'),
     State('prompt-input-2', 'value'),
     State('attention-modules-dropdown', 'value'),
     State('block-modules-dropdown', 'value'),
     State('norm-params-dropdown', 'value'),
     State('logit-lens-dropdown', 'value'),
     State('session-patterns-store', 'data')],
    prevent_initial_call=True
)
def run_ablation_experiment(n_clicks, selected_layer, activation_data, activation_data2, 
                            model_name, prompt, prompt2, attn_patterns, block_patterns, 
                            norm_patterns, logit_pattern, patterns_data):
    """Run ablation experiment on selected layer for both prompts."""
    if not n_clicks or selected_layer is None or not activation_data:
        return no_update, no_update, False, no_update
    
    try:
        # Load model
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
            'block_modules': [mod for pattern in block_patterns for mod in module_patterns.get(pattern, [])],
            'norm_parameters': param_patterns.get(norm_patterns, []) if norm_patterns else [],
            'logit_lens_parameter': all_patterns.get(logit_pattern, [None])[0] if logit_pattern else None
        }
        
        # Run ablation on first prompt
        ablated_data = execute_forward_pass_with_layer_ablation(
            model, tokenizer, prompt, config, selected_layer, activation_data
        )
        
        # Build essential data for first prompt
        essential_data = {
            'model': ablated_data.get('model'),
            'prompt': ablated_data.get('prompt'),
            'input_ids': ablated_data.get('input_ids'),
            'attention_modules': ablated_data.get('attention_modules'),
            'attention_outputs': ablated_data.get('attention_outputs'),
            'block_modules': ablated_data.get('block_modules'),
            'block_outputs': ablated_data.get('block_outputs'),
            'logit_lens_parameter': ablated_data.get('logit_lens_parameter'),
            'actual_output': ablated_data.get('actual_output'),
            'ablated_layer': selected_layer
        }
        
        # Run ablation on second prompt if it exists
        essential_data2 = {}
        if activation_data2 and prompt2 and prompt2.strip():
            ablated_data2 = execute_forward_pass_with_layer_ablation(
                model, tokenizer, prompt2, config, selected_layer, activation_data2
            )
            
            essential_data2 = {
                'model': ablated_data2.get('model'),
                'prompt': ablated_data2.get('prompt'),
                'input_ids': ablated_data2.get('input_ids'),
                'attention_modules': ablated_data2.get('attention_modules'),
                'attention_outputs': ablated_data2.get('attention_outputs'),
                'block_modules': ablated_data2.get('block_modules'),
                'block_outputs': ablated_data2.get('block_outputs'),
                'logit_lens_parameter': ablated_data2.get('logit_lens_parameter'),
                'actual_output': ablated_data2.get('actual_output'),
                'ablated_layer': selected_layer
            }
        
        # Show success message
        success_message = html.Div([
            html.I(className="fas fa-check-circle", style={'color': '#28a745', 'marginRight': '8px'}),
            f"Ablation complete for Layer L{selected_layer}!"
        ], className="status-success")
        
        return essential_data, essential_data2, True, success_message
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        error_message = html.Div([
            html.I(className="fas fa-exclamation-triangle", style={'color': '#dc3545', 'marginRight': '8px'}),
            f"Ablation error: {str(e)}"
        ], className="status-error")
        
        return no_update, no_update, False, error_message

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

# Callback to show BertViz visualization when button is clicked
@app.callback(
    Output({'type': 'bertviz-container', 'layer': dash.dependencies.MATCH}, 'children'),
    [Input({'type': 'bertviz-btn', 'layer': dash.dependencies.MATCH}, 'n_clicks')],
    [State('session-activation-store', 'data'),
     State({'type': 'bertviz-btn', 'layer': dash.dependencies.MATCH}, 'id')],
    prevent_initial_call=True
)
def toggle_bertviz_view(n_clicks, activation_data, button_id):
    """Show/hide BertViz full model view when button is clicked."""
    if not n_clicks or not activation_data:
        return None
    
    try:
        from utils import generate_bertviz_html
        layer_num = button_id['layer']
        
        # Toggle: even clicks hide, odd clicks show
        if n_clicks % 2 == 0:
            return None
        
        # Generate BertViz HTML for this layer
        bertviz_html = generate_bertviz_html(activation_data, layer_num, 'full')
        
        return html.Iframe(
            srcDoc=bertviz_html,
            style={'width': '100%', 'height': '500px', 'border': '1px solid #ddd', 'borderRadius': '4px', 'marginTop': '10px'}
        )
        
    except Exception as e:
        return html.P(f"Error loading BertViz: {str(e)}", style={'color': '#dc3545', 'fontSize': '13px'})


if __name__ == '__main__':
    app.run(debug=True, port=8050)
