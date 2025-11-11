"""
Modular Dash Cytoscape Visualization Dashboard

A simple, modular dashboard for transformer model visualization using Dash and Cytoscape.
Components are organized for easy understanding and maintenance.
"""

import dash
from dash import html, dcc, Input, Output, State, callback, no_update, ALL, MATCH
from utils import (load_model_and_get_patterns, execute_forward_pass, extract_layer_data,
                   categorize_single_layer_heads, format_categorization_summary,
                   compute_layer_wise_summaries)
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
    # Store original activation data before ablation for comparison
    dcc.Store(id='session-activation-store-original', storage_type='session'),
    # Sidebar collapse state (default: collapsed = True)
    dcc.Store(id='sidebar-collapse-store', storage_type='session', data=True),
    # Comparison mode state (default: not comparing)
    dcc.Store(id='comparison-mode-store', storage_type='session', data=False),
    # Second prompt activation data
    dcc.Store(id='session-activation-store-2', storage_type='session'),
    
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
     Output('analysis-loading-indicator', 'children')],
    [Input('run-analysis-btn', 'n_clicks')],
    [State('model-dropdown', 'value'),
     State('prompt-input', 'value'),
     State('prompt-input-2', 'value'),
     State('attention-modules-dropdown', 'value'),
     State('block-modules-dropdown', 'value'),
     State('norm-params-dropdown', 'value'),
     State('session-patterns-store', 'data')],
    prevent_initial_call=True
)
def run_analysis(n_clicks, model_name, prompt, prompt2, attn_patterns, block_patterns, norm_patterns, patterns_data):
    """Run forward pass and store activation data (handles 1 or 2 prompts)."""
    print(f"\n=== DEBUG: run_analysis START ===")
    print(f"DEBUG: n_clicks={n_clicks}, model_name={model_name}, prompt='{prompt}', prompt2='{prompt2}'")
    print(f"DEBUG: block_patterns={block_patterns}")
    
    if not n_clicks or not model_name or not prompt or not block_patterns:
        print("DEBUG: Missing required inputs, returning empty")
        return {}, {}, None, None
    
    try:
        # Load model for execution
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation='eager')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.eval()
        
        # Build config from selected patterns
        module_patterns = patterns_data.get('module_patterns', {})
        param_patterns = patterns_data.get('param_patterns', {})
        
        # Use block patterns (full layer outputs / residual stream) for logit lens
        config = {
            'attention_modules': [mod for pattern in (attn_patterns or []) for mod in module_patterns.get(pattern, [])],
            'block_modules': [mod for pattern in block_patterns for mod in module_patterns.get(pattern, [])],
            'norm_parameters': [param for pattern in (norm_patterns or []) for param in param_patterns.get(pattern, [])]
        }
        
        print(f"DEBUG: config = {config}")
        
        # Execute forward pass for first prompt
        activation_data = execute_forward_pass(model, tokenizer, prompt, config)
        
        print(f"DEBUG: Executed forward pass for prompt 1")
        
        # Store data needed for accordion display and analysis
        essential_data = {
            'model': model_name,
            'prompt': prompt,
            'attention_modules': activation_data.get('attention_modules', []),
            'attention_outputs': activation_data.get('attention_outputs', {}),
            'input_ids': activation_data.get('input_ids', []),
            'block_modules': activation_data.get('block_modules', []),
            'block_outputs': activation_data.get('block_outputs', {}),
            'norm_parameters': activation_data.get('norm_parameters', []),
            'global_top5_tokens': activation_data.get('global_top5_tokens', [])
        }
        
        # Process second prompt if provided
        essential_data2 = {}
        
        if prompt2 and prompt2.strip():
            activation_data2 = execute_forward_pass(model, tokenizer, prompt2, config)
            print(f"DEBUG: Executed forward pass for prompt 2")
            
            essential_data2 = {
                'model': model_name,
                'prompt': prompt2,
                'attention_modules': activation_data2.get('attention_modules', []),
                'attention_outputs': activation_data2.get('attention_outputs', {}),
                'input_ids': activation_data2.get('input_ids', []),
                'block_modules': activation_data2.get('block_modules', []),
                'block_outputs': activation_data2.get('block_outputs', {}),
                'norm_parameters': activation_data2.get('norm_parameters', []),
                'global_top5_tokens': activation_data2.get('global_top5_tokens', [])
            }
        
        # Show success message
        success_message = html.Div([
            html.I(className="fas fa-check-circle", style={'color': '#28a745', 'marginRight': '8px'}),
            "Analysis completed successfully!" + (" (2 prompts)" if prompt2 and prompt2.strip() else "")
        ], className="status-success")
        
        print(f"=== DEBUG: run_analysis END ===\n")
        return essential_data, essential_data2, success_message
        
    except Exception as e:
        print(f"Analysis error: {e}")
        import traceback
        traceback.print_exc()
        
        # Show error message
        error_message = html.Div([
            html.I(className="fas fa-exclamation-triangle", style={'color': '#dc3545', 'marginRight': '8px'}),
            f"Analysis error: {str(e)}"
        ], className="status-error")
        
        return {}, {}, error_message

def _create_top5_by_layer_graph(layer_wise_probs, significant_layers, global_top5_tokens):
    """
    Create line graph showing top 5 tokens' probabilities across layers.
    
    Args:
        layer_wise_probs: Dict mapping layer_num -> {token: prob}
        significant_layers: List of layer numbers with significant increases
        global_top5_tokens: List of (token, prob) tuples for final top 5
    
    Returns:
        Plotly Figure with line graph
    """
    import plotly.graph_objs as go
    
    if not layer_wise_probs or not global_top5_tokens:
        return None
    
    # Extract layer numbers (sorted)
    layer_nums = sorted(layer_wise_probs.keys())
    
    # Create a line for each of the global top 5 tokens
    traces = []
    colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b']
    
    for idx, (token, _) in enumerate(global_top5_tokens[:5]):
        probs = [layer_wise_probs[layer].get(token, 0.0) for layer in layer_nums]
        
        traces.append(go.Scatter(
            x=layer_nums,
            y=probs,
            mode='lines+markers',
            name=f"'{token}'",
            line={'color': colors[idx % len(colors)], 'width': 2},
            marker={'size': 6}
        ))
    
    # Create figure with highlighted significant layers
    fig = go.Figure(data=traces)
    
    # Add yellow highlighting for significant layers
    for sig_layer in significant_layers:
        fig.add_vrect(
            x0=sig_layer - 0.3, x1=sig_layer + 0.3,
            fillcolor='yellow', opacity=0.2,
            layer='below', line_width=0
        )
    
    fig.update_layout(
        title="Top 5 Token Probabilities Across Layers",
        xaxis_title="Layer Number",
        yaxis_title="Probability",
        hovermode='closest',
        legend={'title': 'Token'},
        height=400,
        margin={'l': 60, 'r': 20, 't': 40, 'b': 40}
    )
    
    return fig


def _create_single_prompt_chart(layer_data, title_suffix=''):
    """
    Create a single prompt bar chart (existing functionality).
    
    Args:
        layer_data: Layer data dict (with top_5_tokens, deltas)
        title_suffix: Optional suffix to add to title (e.g., "Before Ablation", "After Ablation")
    
    Returns:
        Plotly Figure with horizontal bars
    """
    import plotly.graph_objs as go
    
    top_5 = layer_data.get('top_5_tokens', [])
    deltas = layer_data.get('deltas', {})
    
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
    
    # Build title with optional suffix
    title_text = f'Top 5 Predictions'
    if title_suffix:
        title_text = f'Top 5 Predictions {title_suffix}'
    
    fig.update_layout(
        title={
            'text': title_text,
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
        layer_data1: Layer data dict for prompt 1 (with top_5_tokens, deltas)
        layer_data2: Layer data dict for prompt 2 (with top_5_tokens, deltas)
        layer_num: Layer number for title
    
    Returns:
        Plotly Figure with grouped bars for overlapping tokens and separate bars for non-overlapping
    """
    import plotly.graph_objs as go
    
    top_5_1 = layer_data1.get('top_5_tokens', [])
    top_5_2 = layer_data2.get('top_5_tokens', [])
    deltas_1 = layer_data1.get('deltas', {})
    deltas_2 = layer_data2.get('deltas', {})
    
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
            'text': f'Top 5 Predictions Comparison',
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


def _create_token_probability_delta_chart(layer_data, layer_num, global_top5_tokens, title_suffix=''):
    """
    Create horizontal bar chart showing change in probabilities for global top 5 tokens.
    
    Args:
        layer_data: Layer data dict with global_top5_deltas
        layer_num: Layer number for title
        global_top5_tokens: List of (token, prob) tuples for final global top 5
        title_suffix: Optional suffix to add to title (e.g., "Before Ablation", "After Ablation")
    
    Returns:
        Plotly Figure with horizontal bars (green for positive, red for negative)
    """
    import plotly.graph_objs as go
    
    global_top5_deltas = layer_data.get('global_top5_deltas', {})
    global_top5_probs = layer_data.get('global_top5_probs', {})
    
    if not global_top5_tokens:
        return None
    
    # Extract tokens and deltas for the global top 5
    tokens = [token for token, _ in global_top5_tokens]
    deltas = [global_top5_deltas.get(token, 0.0) for token in tokens]
    current_probs = [global_top5_probs.get(token, 0.0) for token in tokens]
    
    # Calculate previous probabilities
    prev_probs = [current_probs[i] - deltas[i] for i in range(len(tokens))]
    
    # Create bar colors (green for positive, red for negative)
    bar_colors = ['#28a745' if delta > 0 else '#dc3545' for delta in deltas]
    
    # Create hover text with previous, current, and delta
    hover_texts = [
        f"{tokens[i]}<br>Previous: {prev_probs[i]:.4f}<br>Current: {current_probs[i]:.4f}<br>Change: {deltas[i]:+.4f}"
        for i in range(len(tokens))
    ]
    
    # Create figure
    fig = go.Figure(data=[
        go.Bar(
            x=deltas,
            y=tokens,
            orientation='h',
            marker={'color': bar_colors},
            text=[f'{d:+.4f}' for d in deltas],
            textposition='outside',
            hovertext=hover_texts,
            hoverinfo='text'
        )
    ])
    
    # Determine x-axis range (symmetric around 0 for better visualization)
    max_abs_delta = max(abs(d) for d in deltas) if deltas else 0.01
    x_range = [-max_abs_delta * 1.2, max_abs_delta * 1.2]
    
    # Update layout
    prev_layer_text = "Embedding" if layer_num == 0 else f"Layer {layer_num - 1}"
    title_text = f'Change in Token Probabilities (from {prev_layer_text} to Layer {layer_num})'
    if title_suffix:
        title_text = f'Change in Token Probabilities {title_suffix} (from {prev_layer_text} to Layer {layer_num})'
    
    fig.update_layout(
        title={
            'text': title_text,
            'font': {'size': 13}
        },
        xaxis={'title': 'Probability Change', 'range': x_range, 'zeroline': True, 'zerolinewidth': 2, 'zerolinecolor': '#999'},
        yaxis={'title': '', 'autorange': 'reversed'},
        height=250,
        margin={'l': 100, 'r': 80, 't': 50, 'b': 40},
        hovermode='closest',
        showlegend=False
    )
    
    return fig


def _create_comparison_delta_chart(layer_data1, layer_data2, layer_num, global_top5_1, global_top5_2):
    """
    Create grouped bar chart comparing delta changes for two prompts.
    
    Args:
        layer_data1: Layer data dict for prompt 1
        layer_data2: Layer data dict for prompt 2
        layer_num: Layer number for title
        global_top5_1: Global top 5 tokens for prompt 1
        global_top5_2: Global top 5 tokens for prompt 2
    
    Returns:
        Plotly Figure with grouped bars showing deltas for both prompts
    """
    import plotly.graph_objs as go
    
    deltas_1 = layer_data1.get('global_top5_deltas', {})
    deltas_2 = layer_data2.get('global_top5_deltas', {})
    
    # Merge token sets from both prompts
    tokens_1 = {token for token, _ in global_top5_1}
    tokens_2 = {token for token, _ in global_top5_2}
    all_tokens = sorted(tokens_1 | tokens_2, key=lambda t: -max(abs(deltas_1.get(t, 0)), abs(deltas_2.get(t, 0))))
    
    # Get deltas for all tokens
    deltas_1_list = [deltas_1.get(token, 0.0) for token in all_tokens]
    deltas_2_list = [deltas_2.get(token, 0.0) for token in all_tokens]
    
    # Create figure with grouped bars
    fig = go.Figure()
    
    # Add Prompt 1 bars
    fig.add_trace(go.Bar(
        name='Prompt 1',
        x=deltas_1_list,
        y=all_tokens,
        orientation='h',
        marker={'color': '#667eea'},
        text=[f'{d:+.3f}' if abs(d) > 0.001 else '' for d in deltas_1_list],
        textposition='outside',
        hovertemplate='Prompt 1 - %{y}: %{x:+.4f}<extra></extra>'
    ))
    
    # Add Prompt 2 bars
    fig.add_trace(go.Bar(
        name='Prompt 2',
        x=deltas_2_list,
        y=all_tokens,
        orientation='h',
        marker={'color': '#f59e42'},
        text=[f'{d:+.3f}' if abs(d) > 0.001 else '' for d in deltas_2_list],
        textposition='outside',
        hovertemplate='Prompt 2 - %{y}: %{x:+.4f}<extra></extra>'
    ))
    
    # Determine x-axis range
    max_abs_delta = max(
        max(abs(d) for d in deltas_1_list + deltas_2_list) if (deltas_1_list + deltas_2_list) else 0.01,
        0.01
    )
    x_range = [-max_abs_delta * 1.3, max_abs_delta * 1.3]
    
    # Update layout
    prev_layer_text = "Embedding" if layer_num == 0 else f"Layer {layer_num - 1}"
    fig.update_layout(
        title={
            'text': f'Change in Token Probabilities (from {prev_layer_text} to Layer {layer_num})',
            'font': {'size': 13}
        },
        xaxis={'title': 'Probability Change', 'range': x_range, 'zeroline': True, 'zerolinewidth': 2, 'zerolinecolor': '#999'},
        yaxis={'title': '', 'autorange': 'reversed'},
        barmode='group',
        height=300,
        margin={'l': 100, 'r': 100, 't': 50, 'b': 40},
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


def _create_actual_output_display(activation_data):
    """
    Create a display element showing the actual output token with tooltip.
    
    Args:
        activation_data: Activation data containing actual_output
    
    Returns:
        Dash HTML component displaying the actual output token
    """
    actual_output = activation_data.get('actual_output')
    if not actual_output:
        return None
    
    token = actual_output.get('token', 'N/A')
    probability = actual_output.get('probability', 0.0)
    
    tooltip_text = ("The actual output token may differ from the highest probability token shown in the final layer. "
                   "This is because the model uses residual connections (skip links) that add information across layers. "
                   "The final output is determined after all residual streams are combined. "
                   "See transformer layer implementations for details.")
    
    return html.Div([
        html.Div([
            html.Strong("Actual Model Output: ", style={'color': '#495057', 'fontSize': '14px'}),
            html.Span(f'"{token}"', style={
                'backgroundColor': '#e8f5e9', 
                'padding': '4px 10px', 
                'borderRadius': '4px',
                'fontFamily': 'monospace',
                'fontSize': '14px',
                'fontWeight': '600',
                'color': '#2e7d32',
                'border': '1px solid #4caf50'
            }),
            html.Span(f" (probability: {probability:.4f})", style={
                'color': '#6c757d',
                'fontSize': '13px',
                'marginLeft': '8px'
            }),
            html.I(
                className="fas fa-info-circle",
                id="actual-output-info-icon",
                style={
                    'marginLeft': '10px',
                    'color': '#667eea',
                    'cursor': 'pointer',
                    'fontSize': '14px'
                }
            )
        ], style={'marginBottom': '8px'}),
        html.Div([
            html.I(className="fas fa-lightbulb", style={'marginRight': '6px', 'color': '#ffa726'}),
            tooltip_text
        ], style={
            'fontSize': '12px',
            'color': '#6c757d',
            'backgroundColor': '#fff8e1',
            'padding': '10px',
            'borderRadius': '5px',
            'borderLeft': '3px solid #ffa726',
            'lineHeight': '1.6'
        })
    ], style={
        'marginTop': '15px',
        'padding': '12px',
        'backgroundColor': '#f8f9fa',
        'borderRadius': '6px',
        'border': '1px solid #dee2e6'
    })


# Callback to create accordion panels from layer data
@app.callback(
    Output('layer-accordions-container', 'children'),
    [Input('session-activation-store', 'data'),
     Input('session-activation-store-2', 'data'),
     Input('session-activation-store-original', 'data')],
    [State('model-dropdown', 'value')]
)
def create_layer_accordions(activation_data, activation_data2, original_activation_data, model_name):
    """Create accordion panels for each layer with top-5 bar charts and deltas."""
    if not activation_data or not model_name:
        return html.P("Run analysis to see layer-by-layer predictions.", className="placeholder-text")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import plotly.graph_objs as go
        
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation='eager')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Check if we're in ablation mode
        ablation_mode = activation_data.get('ablated', False) and original_activation_data
        
        # Extract layer data for current activation (may be ablated)
        layer_data = extract_layer_data(activation_data, model, tokenizer)
        
        if not layer_data:
            return html.P("No layer data available.", className="placeholder-text")
        
        # Compute layer-wise probability tracking
        tracking_data = compute_layer_wise_summaries(layer_data, activation_data)
        layer_wise_probs = tracking_data.get('layer_wise_top5_probs', {})
        significant_layers = tracking_data.get('significant_layers', [])
        global_top5 = activation_data.get('global_top5_tokens', [])
        
        # If in ablation mode, also extract original layer data
        original_layer_data = None
        original_layer_wise_probs = {}
        original_significant_layers = []
        original_global_top5 = []
        
        if ablation_mode:
            original_layer_data = extract_layer_data(original_activation_data, model, tokenizer)
            original_tracking_data = compute_layer_wise_summaries(original_layer_data, original_activation_data)
            original_layer_wise_probs = original_tracking_data.get('layer_wise_top5_probs', {})
            original_significant_layers = original_tracking_data.get('significant_layers', [])
            original_global_top5 = original_activation_data.get('global_top5_tokens', [])
        
        # Check if second prompt exists and extract its layer data
        layer_data2 = None
        layer_wise_probs2 = {}
        significant_layers2 = []
        global_top5_2 = []
        comparison_mode = activation_data2 and activation_data2.get('model') == model_name
        
        if comparison_mode:
            layer_data2 = extract_layer_data(activation_data2, model, tokenizer)
            tracking_data2 = compute_layer_wise_summaries(layer_data2, activation_data2)
            layer_wise_probs2 = tracking_data2.get('layer_wise_top5_probs', {})
            significant_layers2 = tracking_data2.get('significant_layers', [])
            global_top5_2 = activation_data2.get('global_top5_tokens', [])
        
        # Create accordion panels (reversed to show final layer first)
        accordions = []
        for i, layer in enumerate(reversed(layer_data)):
            layer_num = layer['layer_num']
            top_token = layer.get('top_token', 'N/A')
            top_prob = layer.get('top_prob', 0.0)
            top_5 = layer.get('top_5_tokens', [])
            deltas = layer.get('deltas', {})
            
            # Create summary header - different format for comparison mode
            if comparison_mode and layer_data2:
                # Find corresponding layer in second prompt
                layer2 = next((l for l in layer_data2 if l['layer_num'] == layer_num), None)
                if layer2:
                    top_token2 = layer2.get('top_token', 'N/A')
                    top_prob2 = layer2.get('top_prob', 0.0)
                    
                    if top_token and top_token2:
                        summary_text = f"Layer L{layer_num}: '{top_token}' vs '{top_token2}'"
                    elif top_token:
                        summary_text = f"Layer L{layer_num}: '{top_token}' vs (no prediction)"
                    elif top_token2:
                        summary_text = f"Layer L{layer_num}: (no prediction) vs '{top_token2}'"
                    else:
                        summary_text = f"Layer L{layer_num}: (no prediction) vs (no prediction)"
                else:
                    summary_text = f"Layer L{layer_num}: '{top_token}' vs (no data)"
            else:
                # Single prompt mode
                if top_token:
                    summary_text = f"Layer L{layer_num}: '{top_token}' (p={top_prob:.3f})"
                else:
                    summary_text = f"Layer L{layer_num}: (no prediction)"
            
            # Create accordion panel content
            content_items = []
            
            # Store delta chart for later (will be added after attention head categories)
            if comparison_mode and layer_data2:
                # Comparison mode: show grouped delta bars
                layer2 = next((l for l in layer_data2 if l['layer_num'] == layer_num), None)
                if layer2:
                    delta_fig = _create_comparison_delta_chart(layer, layer2, layer_num, global_top5, global_top5_2)
                else:
                    delta_fig = _create_token_probability_delta_chart(layer, layer_num, global_top5)
            else:
                # Single prompt mode: show delta bars
                delta_fig = _create_token_probability_delta_chart(layer, layer_num, global_top5)
            
            # Store button section for later (will be added after delta chart)
            num_heads = model.config.num_attention_heads if hasattr(model.config, 'num_attention_heads') else 12
            explore_button_section = html.Div([
                # Button to toggle experiments section
                html.Button(
                    "Explore These Changes",
                    id={'type': 'explore-button', 'layer': layer_num},
                    n_clicks=0,
                    style={
                        'padding': '8px 16px',
                        'backgroundColor': '#667eea',
                        'color': 'white',
                        'border': 'none',
                        'borderRadius': '6px',
                        'cursor': 'pointer',
                        'fontSize': '13px',
                        'fontWeight': '500',
                        'transition': 'all 0.2s'
                    }
                ),
                
                # Collapsible experiments section (initially hidden)
                html.Div([
                    html.Hr(style={'margin': '15px 0'}),
                    
                    # Ablation experiment description
                    html.Div([
                        html.H6("Attention Head Ablation", style={'marginBottom': '8px', 'color': '#495057', 'fontSize': '14px'}),
                        html.P(
                            "Ablation experiments help us understand which attention heads are important by removing them and seeing what changes. "
                            "When we 'ablate' a head, we zero out its contribution to the layer's output. "
                            "If the model's predictions change a lot, that head was important. If they stay similar, that head wasn't doing much.",
                            style={'fontSize': '12px', 'color': '#6c757d', 'lineHeight': '1.5', 'marginBottom': '10px'}
                        ),
                        html.Div([
                            html.Strong("What we zero out: ", style={'color': '#495057', 'fontSize': '12px'}),
                            html.Span(
                                "Each attention head produces a set of values (one per token). We set all these values to zero, "
                                "effectively removing that head's influence. The model then continues processing without that head's contribution.",
                                style={'fontSize': '12px', 'color': '#6c757d'}
                            )
                        ], style={'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '4px', 'marginBottom': '15px', 'border': '1px solid #dee2e6'}),
                        html.P(
                            "Select one or more attention heads below, then click 'Run Ablation' to see the results.",
                            style={'fontSize': '12px', 'color': '#6c757d', 'lineHeight': '1.5', 'marginBottom': '15px'}
                        )
                    ]),
                    
                    # Head selection interface
                    html.Div([
                        html.Label("Select heads to ablate:", style={'fontSize': '13px', 'fontWeight': '500', 'color': '#495057', 'marginBottom': '8px', 'display': 'block'}),
                        html.Div([
                            html.Button(
                                f"Head {h}",
                                id={'type': 'head-select-btn', 'layer': layer_num, 'head': h},
                                n_clicks=0,
                                style={
                                    'padding': '6px 12px',
                                    'margin': '4px',
                                    'backgroundColor': '#f8f9fa',
                                    'color': '#495057',
                                    'border': '1px solid #dee2e6',
                                    'borderRadius': '4px',
                                    'cursor': 'pointer',
                                    'fontSize': '12px',
                                    'transition': 'all 0.2s'
                                }
                            ) for h in range(num_heads)
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '15px'}),
                        
                        # Run ablation button
                        html.Button(
                            "Run Ablation",
                            id={'type': 'run-ablation-btn', 'layer': layer_num},
                            n_clicks=0,
                            disabled=True,
                            style={
                                'padding': '8px 16px',
                                'backgroundColor': '#28a745',
                                'color': 'white',
                                'border': 'none',
                                'borderRadius': '6px',
                                'cursor': 'pointer',
                                'fontSize': '13px',
                                'fontWeight': '500',
                                'transition': 'all 0.2s'
                            }
                        ),
                        
                        # Store for selected heads
                        dcc.Store(id={'type': 'selected-heads-store', 'layer': layer_num}, data=[])
                    ])
                ], id={'type': 'experiments-section', 'layer': layer_num}, style={'display': 'none'})
            ], style={'marginBottom': '15px'})
            
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
                        
                        # Category descriptions for tooltips
                        category_descriptions = {
                            'previous_token': "These heads mainly look at the word right before the current word. They help the model understand word order and grammar, like knowing that 'the' usually comes before a noun.",
                            'first_token': "These heads pay a lot of attention to the first word in the sentence. They help the model remember the overall topic or structure of the sentence.",
                            'bow': "These heads look at many words in the sentence at once, without focusing on any particular one. They help the model get the general meaning by combining information from across the whole sentence.",
                            'syntactic': "These heads look for grammatical relationships between words, like connecting a verb to its subject. For example, in 'The dog runs,' they connect 'dog' to 'runs.'",
                            'other': "These heads have attention patterns that don't fit the other categories. They might be learning special patterns unique to certain tasks or contexts."
                        }
                        
                        # BertViz usage instructions (show once before categories)
                        bertviz_instructions = html.Div([
                            html.Small([
                                html.I(className="fas fa-lightbulb", style={'marginRight': '6px', 'color': '#ffc107'}),
                                html.Strong("How to read the visualizations: ", style={'color': '#495057'}),
                                "The left side shows the words asking for attention (Query), and the right side shows the words being looked at (Key). "
                                "Lines connect words that pay attention to each other - thicker lines mean stronger attention. "
                                "Each color is a different attention head. Double-click a color to see just that head. "
                                "Hover over lines to see the exact attention strength."
                            ], style={'fontSize': '11px', 'color': '#6c757d', 'lineHeight': '1.5', 'display': 'block', 'padding': '10px', 'backgroundColor': '#fff9e6', 'borderRadius': '4px', 'border': '1px solid #ffc107'})
                        ], style={'marginBottom': '12px'})
                        content_items.append(bertviz_instructions)
                        
                        # Create expandable category sections with BertViz visualizations
                        for cat_key, display_name in category_names.items():
                            heads = categorized_heads.get(cat_key, [])
                            if heads:
                                color = category_colors.get(cat_key, '#dfe6e9')
                                description = category_descriptions.get(cat_key, '')
                                
                                # Generate BertViz visualization for this category
                                bertviz_html = generate_category_bertviz_html(activation_data, heads)
                                
                                # Create collapsible category section with description
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
                                        # Category description
                                        html.P(description, style={
                                            'fontSize': '12px',
                                            'color': '#6c757d',
                                            'marginBottom': '10px',
                                            'lineHeight': '1.5',
                                            'fontStyle': 'italic'
                                        }),
                                        # BertViz visualization
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
            
            # Add delta chart after attention head categories
            content_items.append(html.Hr(style={'margin': '15px 0'}))
            
            # If in ablation mode, show before/after comparison
            if ablation_mode and original_layer_data:
                # Find corresponding original layer
                original_layer = next((l for l in original_layer_data if l['layer_num'] == layer_num), None)
                
                if original_layer:
                    # Add explanatory note about ablation comparison
                    content_items.append(html.Div([
                        html.I(className="fas fa-info-circle", style={'marginRight': '8px', 'color': '#667eea'}),
                        f"Comparing probabilities before and after ablating Layer {activation_data.get('ablated_layer')}, " +
                        f"Heads {', '.join([f'H{h}' for h in sorted(activation_data.get('ablated_heads', []))])}"
                    ], style={'fontSize': '12px', 'color': '#6c757d', 'marginBottom': '15px', 'padding': '10px', 
                             'backgroundColor': '#f8f9fa', 'borderRadius': '6px', 'border': '1px solid #dee2e6'}))
                    
                    # Before Ablation Section
                    content_items.append(html.Div([
                        html.H6("Before Ablation", style={
                            'marginBottom': '10px', 'color': '#495057', 'fontSize': '14px', 
                            'fontWeight': '600', 'borderLeft': '4px solid #74b9ff', 'paddingLeft': '10px'
                        }),
                        dcc.Graph(
                            figure=_create_token_probability_delta_chart(original_layer, layer_num, original_global_top5, '(Before Ablation)'),
                            config={'displayModeBar': False},
                            style={'marginBottom': '10px'}
                        )
                    ], style={'padding': '15px', 'backgroundColor': '#e3f2fd', 'borderRadius': '8px', 'marginBottom': '15px'}))
                    
                    # After Ablation Section  
                    content_items.append(html.Div([
                        html.H6("After Ablation", style={
                            'marginBottom': '10px', 'color': '#495057', 'fontSize': '14px',
                            'fontWeight': '600', 'borderLeft': '4px solid #ffb74d', 'paddingLeft': '10px'
                        }),
                        dcc.Graph(
                            figure=_create_token_probability_delta_chart(layer, layer_num, global_top5, '(After Ablation)'),
                            config={'displayModeBar': False},
                            style={'marginBottom': '10px'}
                        )
                    ], style={'padding': '15px', 'backgroundColor': '#fff3e0', 'borderRadius': '8px', 'marginBottom': '15px'}))
                else:
                    # Fallback if original layer not found
                    if delta_fig:
                        content_items.append(
                            dcc.Graph(
                                figure=delta_fig,
                                config={'displayModeBar': False},
                                style={'marginBottom': '15px'}
                            )
                        )
            else:
                # Normal mode (not ablation): show single delta chart
                if delta_fig:
                    content_items.append(
                        dcc.Graph(
                            figure=delta_fig,
                            config={'displayModeBar': False},
                            style={'marginBottom': '15px'}
                        )
                    )
                else:
                    content_items.append(html.P("No probability changes available", style={'color': '#6c757d', 'fontSize': '13px'}))
            
            # Add "Explore These Changes" button after delta chart
            content_items.append(explore_button_section)
            
            # Add CSS class for significant layers (yellow highlighting)
            accordion_classes = "layer-accordion"
            if layer_num in significant_layers or (comparison_mode and layer_num in significant_layers2):
                accordion_classes += " significant-layer"
            
            panel = html.Details([
                html.Summary(summary_text, className="layer-summary"),
                html.Div(content_items, className="layer-content")
            ], className=accordion_classes)
            
            accordions.append(panel)
        
        # Create line graph(s) for top 5 tokens across layers
        line_graphs = []
        
        # If in ablation mode, show before/after comparison
        if ablation_mode and original_layer_wise_probs and original_global_top5:
            # Before Ablation Graph
            fig_before = _create_top5_by_layer_graph(original_layer_wise_probs, original_significant_layers, original_global_top5)
            if fig_before:
                # Update title to indicate "Before Ablation"
                fig_before.update_layout(title="Top 5 Token Probabilities Across Layers (Before Ablation)")
                
                # Build children for before ablation graph
                before_children = [
                    html.H5("Before Ablation", style={
                        'marginBottom': '10px', 'color': '#495057', 'fontSize': '16px',
                        'fontWeight': '600', 'borderLeft': '4px solid #74b9ff', 'paddingLeft': '10px'
                    }),
                    html.Div([
                        html.I(className="fas fa-info-circle", 
                              style={'marginRight': '8px', 'color': '#667eea'}),
                        "This graph shows how the model's confidence in the final top 5 predictions evolves through each layer before ablation."
                    ], style={'fontSize': '13px', 'color': '#6c757d', 'marginBottom': '10px', 'lineHeight': '1.5'}),
                    dcc.Graph(figure=fig_before, config={'displayModeBar': False})
                ]
                
                # Add actual output display
                actual_output_display_before = _create_actual_output_display(original_activation_data)
                if actual_output_display_before:
                    before_children.append(actual_output_display_before)
                
                graph_container_before = html.Div(before_children, 
                    style={'padding': '15px', 'backgroundColor': '#e3f2fd', 'borderRadius': '8px', 'marginBottom': '20px'})
                
                line_graphs.append(graph_container_before)
            
            # After Ablation Graph
            if layer_wise_probs and global_top5:
                fig_after = _create_top5_by_layer_graph(layer_wise_probs, significant_layers, global_top5)
                if fig_after:
                    # Update title to indicate "After Ablation"
                    fig_after.update_layout(title="Top 5 Token Probabilities Across Layers (After Ablation)")
                    
                    ablated_layer = activation_data.get('ablated_layer')
                    ablated_heads = activation_data.get('ablated_heads', [])
                    heads_str = ', '.join([f'H{h}' for h in sorted(ablated_heads)])
                    
                    # Build children for after ablation graph
                    after_children = [
                        html.H5("After Ablation", style={
                            'marginBottom': '10px', 'color': '#495057', 'fontSize': '16px',
                            'fontWeight': '600', 'borderLeft': '4px solid #ffb74d', 'paddingLeft': '10px'
                        }),
                        html.Div([
                            html.I(className="fas fa-info-circle", 
                                  style={'marginRight': '8px', 'color': '#f57c00'}),
                            f"This graph shows how probabilities changed after removing Layer {ablated_layer}, Heads {heads_str}. " +
                            "Compare with the graph above to see the impact of the ablation."
                        ], style={'fontSize': '13px', 'color': '#6c757d', 'marginBottom': '10px', 'lineHeight': '1.5'}),
                        dcc.Graph(figure=fig_after, config={'displayModeBar': False})
                    ]
                    
                    # Add actual output display
                    actual_output_display_after = _create_actual_output_display(activation_data)
                    if actual_output_display_after:
                        after_children.append(actual_output_display_after)
                    
                    # Add merge note at the end
                    after_children.append(
                        html.Small("Note: Tokens with and without leading spaces (e.g., ' cat' and 'cat') are automatically merged.", 
                                  style={'fontSize': '11px', 'color': '#6c757d', 'fontStyle': 'italic'})
                    )
                    
                    graph_container_after = html.Div(after_children, 
                        style={'padding': '15px', 'backgroundColor': '#fff3e0', 'borderRadius': '8px', 'marginBottom': '20px'})
                    
                    line_graphs.append(graph_container_after)
        
        # Normal mode (not ablation): show single line graph
        elif layer_wise_probs and global_top5:
            fig = _create_top5_by_layer_graph(layer_wise_probs, significant_layers, global_top5)
            if fig:
                tooltip_text = ("This graph shows how confident the model is in its top 5 predictions as it processes through each layer. "
                               "Yellow highlights mark layers where the model's confidence in the actual output token doubled (100% or more increase). "
                               "These are the layers where the model made important decisions. "
                               "Click on the Transformer Layers section to see what each layer did.")
                
                merge_note = ("Note: Some tokens appear with a space before them (like ' cat') and some without (like 'cat'). "
                             "We automatically combine these to make the graph easier to read.")
                
                # Create list of children for graph container
                graph_children = [
                    html.Div([
                        html.I(className="fas fa-info-circle", 
                              style={'marginRight': '8px', 'color': '#667eea'}),
                        tooltip_text
                    ], style={'fontSize': '13px', 'color': '#6c757d', 'marginBottom': '10px', 'lineHeight': '1.5'}),
                    dcc.Graph(figure=fig, config={'displayModeBar': False})
                ]
                
                # Add actual output display
                actual_output_display = _create_actual_output_display(activation_data)
                if actual_output_display:
                    graph_children.append(actual_output_display)
                
                # Add merge note at the end
                graph_children.append(
                    html.Small(merge_note, 
                              style={'fontSize': '11px', 'color': '#6c757d', 'fontStyle': 'italic'})
                )
                
                graph_container = html.Div(graph_children, style={'marginBottom': '20px'})
                
                line_graphs.append(graph_container)
        
        # In comparison mode (two prompts), create a second graph or side-by-side display
        if comparison_mode and layer_wise_probs2 and global_top5_2:
            fig2 = _create_top5_by_layer_graph(layer_wise_probs2, significant_layers2, global_top5_2)
            if fig2:
                # Build children for second prompt graph
                children2 = [
                    html.H6("Prompt 2", style={'color': '#495057', 'marginBottom': '10px'}),
                    dcc.Graph(figure=fig2, config={'displayModeBar': False})
                ]
                
                # Add actual output display for second prompt
                actual_output_display2 = _create_actual_output_display(activation_data2)
                if actual_output_display2:
                    children2.append(actual_output_display2)
                
                graph_container2 = html.Div(children2, style={'marginTop': '20px'})
                line_graphs.append(graph_container2)
        
        # Create stacked visual representation for collapsed state
        num_layers = len(layer_data)
        stacked_layers = []
        for i in range(min(5, num_layers)):  # Show first 5 layers as preview
            stacked_layers.append(
                html.Div(f"L{i}", className="stacked-layer-card")
            )
        if num_layers > 5:
            stacked_layers.append(
                html.Div("...", className="stacked-layer-card")
            )
        
        # Create "How This Layer Works" diagram to display in main container
        how_layer_works = html.Div([
            html.H6("How This Layer Works", style={'marginBottom': '10px', 'color': '#495057', 'fontSize': '14px'}),
            html.Div([
                # Input vector
                html.Div([
                    html.Div("[ ... ]", className="flow-box", title="This layer receives the output from the previous layer. Each layer builds on what earlier layers learned, gradually understanding the text better."),
                    html.Div("Input", style={'fontSize': '11px', 'color': '#6c757d', 'textAlign': 'center'})
                ], style={'display': 'inline-block', 'verticalAlign': 'middle'}),
                
                # Arrow to Self-Attention
                html.Div("→", style={'display': 'inline-block', 'margin': '0 10px', 'fontSize': '20px', 'color': '#667eea'}),
                
                # Self-Attention box
                html.Div([
                    html.Div("Self-Attention", className="flow-box attention-box", title="Self-attention lets each word 'look at' all other words in the sentence to understand context. For example, in 'The cat sat on it,' the word 'it' can look back at 'cat' to understand what 'it' refers to."),
                    html.Div("Attention", style={'fontSize': '11px', 'color': '#6c757d', 'textAlign': 'center'})
                ], style={'display': 'inline-block', 'verticalAlign': 'middle'}),
                
                # Container for the split arrows showing green arrows going from Self-Attention
                html.Div([
                    # Green arrow up to Feed-Forward
                    html.Div("↗", style={'fontSize': '20px', 'color': '#28a745', 'lineHeight': '1'}),
                    # Green arrow down to Residual (mirrored)
                    html.Div("↘", style={'fontSize': '20px', 'color': '#28a745', 'lineHeight': '1'})
                ], style={'display': 'inline-block', 'verticalAlign': 'middle', 'margin': '0 5px'}),
                
                # Split into two paths (Feed-Forward on top, Residual on bottom)
                html.Div([
                    # Feed-forward path (top)
                    html.Div([
                        html.Div("F(x)", className="flow-box ffn-box", title="The feed-forward network processes the attention results. Think of it as a calculator that transforms the information to extract deeper meaning and patterns."),
                        html.Div("Feed-Forward", style={'fontSize': '11px', 'color': '#6c757d', 'textAlign': 'center', 'marginTop': '2px'})
                    ], style={'marginBottom': '5px'}),
                    
                    # Residual connection path (bottom)
                    html.Div([
                        html.Div("⤷", className="flow-box", style={'fontSize': '24px', 'color': '#28a745', 'transform': 'scaleX(2)'}, title="The residual connection is like a shortcut that adds the original input back to the output. This helps preserve important information and makes training more stable."),
                        html.Div("Residual", style={'fontSize': '11px', 'color': '#6c757d', 'textAlign': 'center', 'marginTop': '2px'})
                    ])
                ], style={'display': 'inline-block', 'verticalAlign': 'middle', 'textAlign': 'center'}),
                
                # Container for the merge arrows showing green arrows going to Output
                html.Div([
                    # Green arrow from Feed-Forward down
                    html.Div("↘", style={'fontSize': '20px', 'color': '#28a745', 'lineHeight': '1'}),
                    # Green arrow from Residual up
                    html.Div("↗", style={'fontSize': '20px', 'color': '#28a745', 'lineHeight': '1'})
                ], style={'display': 'inline-block', 'verticalAlign': 'middle', 'margin': '0 5px'}),
                
                # Output
                html.Div([
                    html.Div("[ ... ]", className="flow-box", title="The layer's output shows how the model's predictions changed after processing through this layer."),
                    html.Div("Output", style={'fontSize': '11px', 'color': '#6c757d', 'textAlign': 'center'})
                ], style={'display': 'inline-block', 'verticalAlign': 'middle'})
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'center',
                'padding': '15px',
                'backgroundColor': '#f8f9fa',
                'borderRadius': '8px',
                'flexWrap': 'wrap'
            })
        ], style={'marginBottom': '15px', 'padding': '15px'})
        
        # Create collapsible container for transformer layers
        layers_container = html.Details([
            html.Summary([
                html.Div([
                    html.H4("Transformer Layers (Click to Expand)", 
                           style={'margin': 0, 'color': '#495057'}),
                    html.Div(stacked_layers, className="stacked-layers-visual")
                ], style={'display': 'flex', 'alignItems': 'center', 'gap': '20px'})
            ], className="transformer-layers-summary"),
            html.Div([
                how_layer_works,  # Add the diagram here
                html.Div(accordions, className="transformer-layers-content")
            ])
        ], className="transformer-layers-container", open=False)  # Start collapsed
        
        # Create full BertViz button section (below all layer accordions)
        full_bertviz_section = html.Div([
            html.Button(
                [
                    html.I(className="fas fa-eye", style={'marginRight': '8px'}),
                    "View All Attention Heads Interactively (BertViz)"
                ],
                id='full-bertviz-btn',
                n_clicks=0,
                style={
                    'padding': '12px 24px',
                    'backgroundColor': '#764ba2',
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '8px',
                    'cursor': 'pointer',
                    'fontSize': '14px',
                    'fontWeight': '500',
                    'width': '100%',
                    'transition': 'all 0.2s',
                    'marginTop': '1rem'
                }
            ),
            # Container for full BertViz visualization (initially hidden)
            html.Div(id='full-bertviz-container', style={'marginTop': '1rem'})
        ], style={'marginTop': '2rem'})
        
        # Return all components
        return html.Div([
            layers_container,  # Collapsible layers container at top
            *line_graphs,  # Line graph(s) below (showing outputs)
            full_bertviz_section  # Full BertViz button at the bottom
        ])
        
    except Exception as e:
        print(f"Error creating accordions: {e}")
        import traceback
        traceback.print_exc()
        return html.P(f"Error creating layer view: {str(e)}", className="placeholder-text")

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

# Enable Run Analysis button when requirements are met
@app.callback(
    Output('run-analysis-btn', 'disabled'),
    [Input('model-dropdown', 'value'),
     Input('prompt-input', 'value'),
     Input('block-modules-dropdown', 'value'),
     Input('norm-params-dropdown', 'value')]
)
def enable_run_button(model, prompt, block_modules, norm_params):
    """Enable Run Analysis button when model, prompt, layer blocks, and norm parameters are selected."""
    return not (model and prompt and block_modules and norm_params)

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
    import json
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
     State('prompt-input', 'value')],
    prevent_initial_call=True
)
def run_head_ablation(n_clicks_list, selected_heads_list, activation_data, model_name, prompt):
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
    # ctx.states_list contains the State values in order
    button_index = None
    if hasattr(ctx, 'states_list') and ctx.states_list:
        # states_list[0] corresponds to selected-heads-store
        for idx, state_info in enumerate(ctx.states_list[0]):
            if state_info['id'].get('layer') == layer_num:
                button_index = idx
                break
    
    # Fallback: if states_list doesn't work, try matching by iterating
    if button_index is None:
        # This shouldn't happen, but as a fallback, just return error
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
        from utils import execute_forward_pass_with_head_ablation
        
        # Save original activation data before ablation
        import copy
        original_data = copy.deepcopy(activation_data)
        
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation='eager')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Get config from original activation data
        config = {
            'attention_modules': activation_data.get('attention_modules', []),
            'block_modules': activation_data.get('block_modules', []),
            'norm_parameters': activation_data.get('norm_parameters', [])
        }
        
        # Run ablation
        ablated_data = execute_forward_pass_with_head_ablation(
            model, tokenizer, prompt, config, layer_num, selected_heads
        )
        
        # Update activation data with ablated results
        # Mark as ablated for visual indication
        ablated_data['ablated'] = True
        ablated_data['ablated_layer'] = layer_num
        ablated_data['ablated_heads'] = selected_heads
        
        # Preserve input_ids from original data if not present (prompt is unchanged)
        if 'input_ids' not in ablated_data and 'input_ids' in activation_data:
            ablated_data['input_ids'] = activation_data['input_ids']
        
        # Success message
        heads_str = ', '.join([f"H{h}" for h in sorted(selected_heads)])
        success_message = html.Div([
            html.I(className="fas fa-check-circle", style={'marginRight': '8px', 'color': '#28a745'}),
            f"Ablation complete: Layer {layer_num}, Heads {heads_str} removed"
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


if __name__ == '__main__':
    app.run(debug=True, port=8050)
