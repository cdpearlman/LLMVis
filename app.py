"""
Transformer Explanation Dashboard

A dashboard focused on explaining how transformer models process input and arrive at predictions.
Uses a linear pipeline visualization with expandable stages.
"""

import dash
from dash import html, dcc, Input, Output, State, callback, no_update, ALL, MATCH
import json
import torch
from utils import (load_model_and_get_patterns, execute_forward_pass, extract_layer_data,
                   categorize_single_layer_heads, perform_beam_search,
                   execute_forward_pass_with_head_ablation,
                   execute_forward_pass_with_multi_layer_head_ablation,
                   evaluate_sequence_ablation, score_sequence,
                   get_head_category_counts)
from utils.head_detection import categorize_all_heads
from utils.model_config import get_auto_selections
from utils.token_attribution import compute_integrated_gradients, compute_simple_gradient_attribution

# Import modular components
from components.sidebar import create_sidebar
from components.model_selector import create_model_selector
from components.glossary import create_glossary_modal
from components.pipeline import (create_pipeline_container, create_tokenization_content,
                                  create_embedding_content, create_attention_content,
                                  create_mlp_content, create_output_content)
from components.investigation_panel import (create_investigation_panel, create_ablation_head_buttons,
                                             create_ablation_results_display, create_attribution_results_display,
                                             create_selected_heads_display)

# Initialize Dash app
app = dash.Dash(
    __name__, 
    suppress_callback_exceptions=True,
    external_stylesheets=[
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
    ]
)
app.title = "Transformer Explanation Dashboard"


# ============================================================================
# APP LAYOUT
# ============================================================================

app.layout = html.Div([
    # Glossary Modal
    create_glossary_modal(),
    
    # Session storage
    dcc.Store(id='session-activation-store', storage_type='memory'),
    dcc.Store(id='session-patterns-store', storage_type='session'),
    dcc.Store(id='session-activation-store-original', storage_type='memory'),
    dcc.Store(id='sidebar-collapse-store', storage_type='session', data=True),
    dcc.Store(id='generation-results-store', storage_type='session'),
    dcc.Store(id='investigation-active-tab', storage_type='session', data='ablation'),
    dcc.Store(id='ablation-selected-heads', storage_type='session', data=[]),
    # Agent F: Stores for separating original prompt analysis from beam generation
    dcc.Store(id='session-original-prompt-store', storage_type='memory'),  # Original user prompt
    dcc.Store(id='session-selected-beam-store', storage_type='memory'),    # Selected beam for comparison
    
    # Main container
    html.Div([
        # Header
        html.Div([
            html.H1("Transformer Explanation Dashboard", className="header-title"),
            html.P("Understand how transformer models process text and make predictions", 
                   className="header-subtitle")
        ], className="header"),
        
        # Main content area
        html.Div([
            # Left sidebar
            html.Div([
                create_sidebar()
            ], id="sidebar-container", className="sidebar collapsed"),
            
            # Right main panel
            html.Div([
                # Generator Interface
                html.Div([
                    html.H3("Input", className="section-title"),
                    create_model_selector(),
                    
                    # Generation Settings
                    html.Div([
                        html.H4("Generation Settings", style={'fontSize': '14px', 'marginTop': '15px', 'marginBottom': '10px'}),
                        
                        html.Div([
                            html.Div([
                                html.Label("Number of New Tokens:", className="input-label"),
                                dcc.Slider(
                                    id='max-new-tokens-slider',
                                    min=1, max=20, step=1, value=1,
                                    marks={1: '1', 5: '5', 10: '10', 20: '20'},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                )
                            ], style={'flex': '1', 'marginRight': '20px'}),
                            
                            html.Div([
                                html.Label("Number of Generation Choices:", className="input-label"),
                                dcc.Slider(
                                    id='beam-width-slider',
                                    min=1, max=5, step=1, value=1,
                                    marks={1: '1', 3: '3', 5: '5'},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                )
                            ], style={'flex': '1'})
                        ], style={'display': 'flex', 'marginBottom': '20px'}),
                        
                        html.Button(
                            [html.I(className="fas fa-play", style={'marginRight': '8px'}), "Analyze"],
                            id="generate-btn",
                            className="action-button primary-button",
                            disabled=True,
                            style={'width': '100%', 'padding': '12px', 'fontSize': '16px'}
                        )
                    ], style={'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px', 'marginTop': '15px'})
                    
                ], className="config-section"),
                
                # Results List (for beam search)
                dcc.Loading(
                    id="generation-loading",
                    type="default",
                    children=html.Div(id="generation-results-container", style={'marginTop': '20px'}),
                    color='#667eea'
                ),
                
                # Pipeline Visualization
                html.Div([
                    html.Hr(style={'margin': '30px 0', 'borderTop': '1px solid #dee2e6'}),
                    create_pipeline_container()
                ], id="pipeline-section"),
                
                # Investigation Panel
                html.Div([
                    html.Hr(style={'margin': '30px 0', 'borderTop': '1px solid #dee2e6'}),
                    create_investigation_panel()
                ], id="investigation-section")
                
            ], className="main-panel")
        ], className="content-container")
    ], className="app-container")
], className="app-wrapper")


# ============================================================================
# CALLBACKS: Glossary
# ============================================================================

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


# ============================================================================
# CALLBACKS: Model Loading
# ============================================================================

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
        module_patterns, param_patterns = load_model_and_get_patterns(selected_model)
        
        def create_grouped_options(patterns_dict, filter_keywords, option_type):
            filtered_options = []
            other_options = []
            for pattern, items in patterns_dict.items():
                pattern_lower = pattern.lower()
                option = {'label': pattern, 'value': pattern, 'title': f"{len(items)} {option_type} matching this pattern"}
                if any(keyword in pattern_lower for keyword in filter_keywords):
                    filtered_options.append(option)
                else:
                    other_options.append(option)
            result = []
            if filtered_options:
                result.extend(filtered_options)
                if other_options:
                    result.append({'label': '─── Other Options ───', 'value': '_separator_', 'disabled': True})
                    result.extend(other_options)
            else:
                result.extend(other_options)
            return result
        
        attention_options = create_grouped_options(module_patterns, ['attn', 'attention'], 'modules')
        block_options = create_grouped_options(module_patterns, ['layers', 'h.', 'blocks', 'decoder.layers'], 'modules')
        norm_options = create_grouped_options(param_patterns, ['norm', 'layernorm', 'layer_norm'], 'params')
        
        auto_selections = get_auto_selections(selected_model, module_patterns, param_patterns)
        
        patterns_data = {
            'module_patterns': module_patterns,
            'param_patterns': param_patterns,
            'selected_model': selected_model,
            'family': auto_selections.get('family_name'),
            'family_description': auto_selections.get('family_description', '')
        }
        
        family_name = auto_selections.get('family_name')
        if family_name:
            loading_content = html.Div([
                html.I(className="fas fa-check-circle", style={'color': '#28a745', 'marginRight': '8px'}),
                html.Div([
                    html.Div("Model loaded successfully!"),
                    html.Div(f"Detected family: {auto_selections.get('family_description', family_name)}", 
                            style={'fontSize': '12px', 'color': '#6c757d', 'marginTop': '4px'})
                ])
            ], className="status-success")
        else:
            loading_content = html.Div([
                html.I(className="fas fa-check-circle", style={'color': '#28a745', 'marginRight': '8px'}),
                "Model loaded - manual selection required"
            ], className="status-success")
        
        return (patterns_data, attention_options, block_options, norm_options,
                auto_selections.get('attention_selection', []),
                auto_selections.get('block_selection', []),
                auto_selections.get('norm_selection', []),
                loading_content)
        
    except Exception as e:
        print(f"Error loading model patterns: {e}")
        error_content = html.Div([
            html.I(className="fas fa-exclamation-triangle", style={'color': '#dc3545', 'marginRight': '8px'}),
            f"Error loading model: {str(e)}"
        ], className="status-error")
        return {}, [], [], [], None, None, None, error_content


@app.callback(
    Output('loading-indicator', 'children', allow_duplicate=True),
    [Input('model-dropdown', 'value')],
    prevent_initial_call=True
)
def show_loading_spinner(selected_model):
    if not selected_model:
        return None
    return html.Div([
        html.I(className="fas fa-spinner fa-spin", style={'marginRight': '8px'}),
        "Loading model..."
    ], className="status-loading")


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
    if not n_clicks:
        return no_update
    cleared_status = html.Div([
        html.I(className="fas fa-broom", style={'color': '#6c757d', 'marginRight': '8px'}),
        "All selections cleared"
    ], className="status-cleared")
    return None, None, None, {}, cleared_status


@app.callback(
    Output('generate-btn', 'disabled'),
    [Input('model-dropdown', 'value'),
     Input('prompt-input', 'value'),
     Input('block-modules-dropdown', 'value'),
     Input('norm-params-dropdown', 'value')]
)
def enable_run_button(model, prompt, block_modules, norm_params):
    return not (model and prompt and block_modules and norm_params)


# ============================================================================
# CALLBACKS: Generation & Analysis
# ============================================================================

@app.callback(
    [Output('generation-results-container', 'children'),
     Output('generation-results-store', 'data'),
     Output('pipeline-container', 'style'),
     Output('investigation-panel', 'style'),
     Output('session-activation-store', 'data', allow_duplicate=True),
     Output('session-original-prompt-store', 'data'),
     Output('session-selected-beam-store', 'data')],
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
    """
    Run generation and analysis.
    
    Agent F refactor: Pipeline analysis (tokenization, embedding, attention, MLP, output) 
    always runs on the ORIGINAL PROMPT only. Beam generation results are stored separately 
    for comparison in experiments.
    """
    if not n_clicks or not model_name or not prompt:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation='eager')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.eval()

        # Always run beam search (even with max_new_tokens=1)
        results = perform_beam_search(model, tokenizer, prompt, beam_width, max_new_tokens)
        
        # Store original prompt for reference
        original_prompt_data = {'prompt': prompt, 'model': model_name}
        
        # Build module config for analysis
        module_patterns = patterns_data.get('module_patterns', {})
        param_patterns = patterns_data.get('param_patterns', {})
        
        config = {
            'attention_modules': [mod for pattern in (attn_patterns or []) for mod in module_patterns.get(pattern, [])],
            'block_modules': [mod for pattern in (block_patterns or []) for mod in module_patterns.get(pattern, [])],
            'norm_parameters': [param for pattern in (norm_patterns or []) for param in param_patterns.get(pattern, [])]
        }
        
        if not config['block_modules']:
            return (html.Div("Please select modules in the sidebar.", style={'color': 'red'}), 
                    results, {'display': 'none'}, {'display': 'none'}, {}, original_prompt_data, {})

        # AGENT F KEY CHANGE: Run analysis on ORIGINAL PROMPT only, not generated text
        # This ensures pipeline stages show how the model processes the user's input,
        # regardless of what tokens are generated.
        activation_data = execute_forward_pass(model, tokenizer, prompt, config)
        
        results_ui = []
        if max_new_tokens > 1:
            # Show generated sequences for user selection
            results_ui.append(html.H4("Generated Sequences", className="section-title"))
            results_ui.append(html.P("Select a sequence to store for comparison after experiments.", 
                                    style={'color': '#6c757d', 'fontSize': '13px', 'marginBottom': '12px'}))
            for i, result in enumerate(results):
                results_ui.append(html.Div([
                    html.Div([
                        html.Span(f"Rank {i+1}", style={'fontWeight': 'bold', 'marginRight': '10px', 'color': '#667eea'})
                    ], style={'marginBottom': '5px'}),
                    html.Div(result['text'], style={'fontFamily': 'monospace', 'backgroundColor': '#fff', 'padding': '10px', 'borderRadius': '4px', 'border': '1px solid #dee2e6'}),
                    html.Button("Select for Comparison", id={'type': 'result-item', 'index': i}, n_clicks=0,
                               className="action-button secondary-button", style={'marginTop': '10px', 'fontSize': '12px'})
                ], style={'marginBottom': '15px', 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '6px'}))
            
            # Show pipeline immediately (analyzing original prompt)
            return (results_ui, results, {'display': 'block'}, {'display': 'block'}, 
                    activation_data, original_prompt_data, {})
            
        else:
            # Single token generation - store the result as selected beam
            selected_beam_data = {'text': results[0]['text'], 'score': results[0].get('score', 0)}
            return (results_ui, results, {'display': 'block'}, {'display': 'block'}, 
                    activation_data, original_prompt_data, selected_beam_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return (html.Div(f"Error: {e}", style={'color': 'red'}), [], 
                {'display': 'none'}, {'display': 'none'}, {}, {}, {})


@app.callback(
    [Output('session-selected-beam-store', 'data', allow_duplicate=True),
     Output('generation-results-container', 'children', allow_duplicate=True)],
    Input({'type': 'result-item', 'index': ALL}, 'n_clicks'),
    [State('generation-results-store', 'data')],
    prevent_initial_call=True
)
def store_selected_beam(n_clicks_list, results_data):
    """
    Agent F: Store selected beam for post-experiment comparison.
    
    The pipeline analysis already runs on the original prompt. This callback
    stores the selected beam text and updates the UI to show only the selected
    sequence, clearing all other options.
    """
    if not any(n_clicks_list) or not results_data:
        return no_update, no_update
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update
        
    triggered_id = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
    index = triggered_id['index']
    
    result = results_data[index]
    
    # Build UI showing only the selected sequence with a "selected" badge
    selected_ui = html.Div([
        html.H4("Selected Sequence", className="section-title"),
        html.Div([
            html.Div([
                html.Span([
                    html.I(className="fas fa-check-circle", style={'marginRight': '8px', 'color': '#28a745'}),
                    "Selected for Comparison"
                ], style={
                    'display': 'inline-flex',
                    'alignItems': 'center',
                    'padding': '6px 12px',
                    'backgroundColor': '#d4edda',
                    'color': '#155724',
                    'borderRadius': '16px',
                    'fontSize': '12px',
                    'fontWeight': '500',
                    'marginBottom': '12px'
                })
            ]),
            html.Div(result['text'], style={
                'fontFamily': 'monospace',
                'backgroundColor': '#fff',
                'padding': '12px',
                'borderRadius': '6px',
                'border': '2px solid #28a745'
            })
        ], style={
            'padding': '16px',
            'backgroundColor': '#f8f9fa',
            'borderRadius': '8px',
            'border': '1px solid #dee2e6'
        })
    ])
    
    # Store the selected beam for comparison after experiments
    return {'text': result['text'], 'score': result.get('score', 0), 'index': index}, selected_ui


# ============================================================================
# CALLBACKS: Pipeline Stage Content
# ============================================================================

@app.callback(
    [Output('stage-1-summary', 'children'),
     Output('stage-1-content', 'children'),
     Output('stage-2-summary', 'children'),
     Output('stage-2-content', 'children'),
     Output('stage-3-summary', 'children'),
     Output('stage-3-content', 'children'),
     Output('stage-4-summary', 'children'),
     Output('stage-4-content', 'children'),
     Output('stage-5-summary', 'children'),
     Output('stage-5-content', 'children')],
    [Input('session-activation-store', 'data')],
    [State('model-dropdown', 'value')]
)
def update_pipeline_content(activation_data, model_name):
    """Update all pipeline stage content based on activation data."""
    empty_outputs = ["Awaiting analysis...", html.P("Run analysis to see details.", style={'color': '#6c757d'})] * 5
    
    if not activation_data or not model_name:
        return tuple(empty_outputs)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation='eager')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Extract data
        input_ids = activation_data.get('input_ids', [[]])[0]
        tokens = [tokenizer.decode([tid]) for tid in input_ids]
        layer_data = extract_layer_data(activation_data, model, tokenizer)
        
        # Get model config info
        hidden_dim = model.config.hidden_size
        num_heads = model.config.num_attention_heads
        num_layers = model.config.num_hidden_layers
        intermediate_dim = getattr(model.config, 'intermediate_size', hidden_dim * 4)
        
        # Get actual output
        actual_output = activation_data.get('actual_output', {})
        predicted_token = actual_output.get('token', '')
        predicted_prob = actual_output.get('probability', 0)
        
        # Get global top 5
        global_top5 = activation_data.get('global_top5_tokens', [])
        if global_top5:
            if isinstance(global_top5[0], dict):
                top_tokens = [(t['token'], t['probability']) for t in global_top5]
            else:
                top_tokens = global_top5
        else:
            top_tokens = []
        
        # Get attention info from first layer
        top_attended = None
        if layer_data:
            top_attended = layer_data[0].get('top_attended_tokens', [])
        
        # Generate BertViz HTML
        from utils import generate_bertviz_html
        attention_html = None
        try:
            attention_html = generate_bertviz_html(activation_data, 0, 'full')
        except:
            pass
        
        # Agent G: Get full head categorization for attention stage UI (expandable categories)
        head_categories = None
        try:
            head_categories = categorize_all_heads(activation_data)
        except:
            pass
        
        # Build outputs for each stage
        outputs = []
        
        # Stage 1: Tokenization
        outputs.append(f"{len(tokens)} tokens")
        outputs.append(create_tokenization_content(tokens, input_ids))
        
        # Stage 2: Embedding
        outputs.append(f"{hidden_dim}-dim vectors")
        outputs.append(create_embedding_content(hidden_dim, len(tokens)))
        
        # Stage 3: Attention (Agent G: now includes head_categories)
        outputs.append(f"{num_heads} heads × {num_layers} layers")
        outputs.append(create_attention_content(attention_html, None, head_categories=head_categories))
        
        # Stage 4: MLP
        outputs.append(f"{num_layers} layers")
        outputs.append(create_mlp_content(num_layers, hidden_dim, intermediate_dim))
        
        # Stage 5: Output
        # Get original prompt for context display
        original_prompt = activation_data.get('prompt', '')
        outputs.append(f"→ {predicted_token}")
        outputs.append(create_output_content(top_tokens, predicted_token, predicted_prob, 
                                             original_prompt=original_prompt))
        
        return tuple(outputs)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return tuple(empty_outputs)


# ============================================================================
# CALLBACKS: Sidebar
# ============================================================================

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
    if n_clicks is None:
        return True, {'display': 'none'}, 'sidebar collapsed', html.I(className="fas fa-chevron-right")
    
    new_collapsed = not is_collapsed
    style = {'display': 'none'} if new_collapsed else {'display': 'block'}
    class_name = 'sidebar collapsed' if new_collapsed else 'sidebar'
    icon = html.I(className="fas fa-chevron-right") if new_collapsed else html.I(className="fas fa-chevron-left")
    
    return new_collapsed, style, class_name, icon


# ============================================================================
# CALLBACKS: Investigation Panel - Tabs
# ============================================================================

@app.callback(
    [Output('investigation-active-tab', 'data'),
     Output('investigation-tab-ablation', 'style'),
     Output('investigation-tab-attribution', 'style'),
     Output('investigation-ablation-content', 'style'),
     Output('investigation-attribution-content', 'style')],
    [Input('investigation-tab-ablation', 'n_clicks'),
     Input('investigation-tab-attribution', 'n_clicks')],
    [State('investigation-active-tab', 'data')],
    prevent_initial_call=True
)
def switch_investigation_tab(abl_clicks, attr_clicks, current_tab):
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update, no_update, no_update
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    active_style = {'padding': '10px 20px', 'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer',
                   'fontSize': '14px', 'fontWeight': '500', 'backgroundColor': '#667eea', 'color': 'white'}
    inactive_style = {'padding': '10px 20px', 'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer',
                     'fontSize': '14px', 'fontWeight': '500', 'backgroundColor': '#f8f9fa', 'color': '#495057'}
    
    if triggered_id == 'investigation-tab-ablation':
        return 'ablation', active_style, inactive_style, {'display': 'block'}, {'display': 'none'}
    else:
        return 'attribution', inactive_style, active_style, {'display': 'none'}, {'display': 'block'}


# ============================================================================
# CALLBACKS: Investigation Panel - Ablation
# ============================================================================

@app.callback(
    Output('ablation-layer-dropdown', 'options'),
    [Input('session-activation-store', 'data')],
    [State('model-dropdown', 'value')]
)
def update_ablation_layer_options(activation_data, model_name):
    if not activation_data or not model_name:
        return []
    
    block_modules = activation_data.get('block_modules', [])
    import re
    layer_options = []
    for mod in block_modules:
        match = re.search(r'\.(\d+)(?:\.|$)', mod)
        if match:
            layer_num = int(match.group(1))
            layer_options.append({'label': f'Layer {layer_num}', 'value': layer_num})
    
    # Sort and dedupe
    layer_options = sorted(list({opt['value']: opt for opt in layer_options}.values()), key=lambda x: x['value'])
    return layer_options


@app.callback(
    Output('ablation-head-grid', 'children'),
    [Input('ablation-layer-dropdown', 'value'),
     Input('ablation-selected-heads', 'data')],
    [State('model-dropdown', 'value')]
)
def update_ablation_head_grid(layer_num, selected_heads, model_name):
    if layer_num is None or not model_name:
        return html.P("Select a layer first.", style={'color': '#6c757d', 'fontStyle': 'italic'})
    
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        num_heads = config.num_attention_heads
        # Filter to only valid dict entries and pass to button creator
        valid_selected = [
            item for item in (selected_heads or [])
            if isinstance(item, dict) and 'layer' in item and 'head' in item
        ]
        return create_ablation_head_buttons(num_heads, layer_num, valid_selected)
    except:
        return html.P("Could not load model config.", style={'color': '#dc3545'})


@app.callback(
    [Output('ablation-selected-heads', 'data'),
     Output('run-ablation-btn', 'disabled'),
     Output('ablation-selected-display', 'children')],
    [Input({'type': 'ablation-head-btn', 'layer': ALL, 'head': ALL}, 'n_clicks')],
    [State('ablation-selected-heads', 'data')],
    prevent_initial_call=True
)
def handle_ablation_head_selection(n_clicks_list, selected_heads):
    """
    Handle head button clicks - stores layer+head combinations and preserves
    selections across layer changes.
    """
    # Check if any button was actually clicked (n_clicks > 0)
    if not n_clicks_list or not any(n for n in n_clicks_list if n and n > 0):
        return no_update, no_update, no_update
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update
    
    # Get the triggered button's n_clicks value
    triggered_prop = ctx.triggered[0]
    triggered_value = triggered_prop.get('value', 0)
    
    # Only process if this is an actual click (not initial render)
    if not triggered_value or triggered_value == 0:
        return no_update, no_update, no_update
    
    # Initialize or clean up selected_heads - ensure it's a list of dicts
    if selected_heads is None:
        selected_heads = []
    else:
        # Filter out any non-dict entries (cleanup old format data)
        selected_heads = [item for item in selected_heads if isinstance(item, dict)]
    
    try:
        triggered_id = json.loads(triggered_prop['prop_id'].split('.')[0])
        layer_num = triggered_id['layer']
        head_idx = triggered_id['head']
        
        # Create layer+head object
        new_item = {'layer': layer_num, 'head': head_idx}
        
        # Check if this layer+head combination is already selected
        found_idx = None
        for i, item in enumerate(selected_heads):
            if isinstance(item, dict) and item.get('layer') == layer_num and item.get('head') == head_idx:
                found_idx = i
                break
        
        if found_idx is not None:
            # Remove if already selected
            selected_heads.pop(found_idx)
        else:
            # Add if not selected
            selected_heads.append(new_item)
    except Exception as e:
        print(f"Error in handle_ablation_head_selection: {e}")
        pass
    
    # Create the display of selected heads
    display = create_selected_heads_display(selected_heads)
    
    return selected_heads, len(selected_heads) == 0, display


@app.callback(
    [Output('ablation-selected-heads', 'data', allow_duplicate=True),
     Output('run-ablation-btn', 'disabled', allow_duplicate=True),
     Output('ablation-selected-display', 'children', allow_duplicate=True)],
    [Input({'type': 'ablation-remove-btn', 'layer': ALL, 'head': ALL}, 'n_clicks')],
    [State('ablation-selected-heads', 'data')],
    prevent_initial_call=True
)
def handle_ablation_head_removal(n_clicks_list, selected_heads):
    """Handle removing individual head selections via the x button on chips."""
    # Check if any button was actually clicked
    if not n_clicks_list or not any(n for n in n_clicks_list if n and n > 0):
        return no_update, no_update, no_update
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update
    
    # Get the triggered button's n_clicks value
    triggered_prop = ctx.triggered[0]
    triggered_value = triggered_prop.get('value', 0)
    
    # Only process if this is an actual click
    if not triggered_value or triggered_value == 0:
        return no_update, no_update, no_update
    
    if selected_heads is None:
        selected_heads = []
    
    try:
        triggered_id = json.loads(triggered_prop['prop_id'].split('.')[0])
        layer_num = triggered_id['layer']
        head_idx = triggered_id['head']
        
        # Remove the matching layer+head combination
        selected_heads = [
            item for item in selected_heads
            if not (isinstance(item, dict) and item.get('layer') == layer_num and item.get('head') == head_idx)
        ]
    except Exception as e:
        print(f"Error in handle_ablation_head_removal: {e}")
        pass
    
    # Create the updated display
    display = create_selected_heads_display(selected_heads)
    
    return selected_heads, len(selected_heads) == 0, display


@app.callback(
    Output('ablation-results-container', 'children'),
    [Input('run-ablation-btn', 'n_clicks')],
    [State('ablation-selected-heads', 'data'),
     State('session-activation-store', 'data'),
     State('model-dropdown', 'value'),
     State('prompt-input', 'value'),
     State('session-selected-beam-store', 'data')],
    prevent_initial_call=True
)
def run_ablation_experiment(n_clicks, selected_heads, activation_data, model_name, prompt, selected_beam):
    """
    Run ablation on ORIGINAL PROMPT and compare results.
    
    Supports multi-layer ablation: groups selected heads by layer and runs
    ablation for each layer. Shows cumulative effect on model output.
    """
    if not n_clicks or not selected_heads or not activation_data:
        return no_update
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation='eager')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.eval()
        
        # Always analyze the original prompt
        sequence_text = activation_data.get('prompt', prompt)
        
        config = {
            'attention_modules': activation_data.get('attention_modules', []),
            'block_modules': activation_data.get('block_modules', []),
            'norm_parameters': activation_data.get('norm_parameters', [])
        }
        
        # Get original output (from analysis of original prompt)
        original_output = activation_data.get('actual_output', {})
        original_token = original_output.get('token', '')
        original_prob = original_output.get('probability', 0)
        
        # Group selected heads by layer
        heads_by_layer = {}
        for item in selected_heads:
            if isinstance(item, dict):
                layer = item.get('layer')
                head = item.get('head')
                if layer is not None and head is not None:
                    if layer not in heads_by_layer:
                        heads_by_layer[layer] = []
                    heads_by_layer[layer].append(head)
        
        if not heads_by_layer:
            return html.Div("No valid heads selected.", style={'color': '#dc3545'})
        
        # Run ablation for all layers simultaneously in a single forward pass
        ablated_data = execute_forward_pass_with_multi_layer_head_ablation(
            model, tokenizer, sequence_text, config, heads_by_layer
        )
        ablated_output = ablated_data.get('actual_output', {})
        ablated_token = ablated_output.get('token', '')
        ablated_prob = ablated_output.get('probability', 0)
        
        # Format selected heads for display
        all_heads_formatted = [f"L{item['layer']}-H{item['head']}" for item in selected_heads if isinstance(item, dict)]
        
        # Build results display
        results = []
        
        # Summary of what was ablated
        results.append(html.Div([
            html.H5("Ablation Results", style={'color': '#495057', 'marginBottom': '16px'}),
            html.Div([
                html.Span("Ablated heads: ", style={'color': '#6c757d'}),
                html.Span(', '.join(all_heads_formatted),
                         style={'fontWeight': '500', 'color': '#667eea', 'fontFamily': 'monospace'})
            ], style={'marginBottom': '16px'})
        ]))
        
        # Before/After comparison
        output_changed = original_token != ablated_token
        prob_delta = ablated_prob - original_prob
        
        results.append(html.Div([
            # Original
            html.Div([
                html.Div("Original", style={'fontSize': '12px', 'color': '#6c757d', 'marginBottom': '6px'}),
                html.Div(original_token, style={
                    'padding': '10px 16px',
                    'backgroundColor': '#e8f5e9',
                    'border': '2px solid #4caf50',
                    'borderRadius': '6px',
                    'fontFamily': 'monospace',
                    'fontWeight': '600',
                    'textAlign': 'center'
                }),
                html.Div(f"{original_prob:.1%}", style={
                    'fontSize': '12px',
                    'color': '#6c757d',
                    'marginTop': '4px',
                    'textAlign': 'center'
                })
            ], style={'flex': '1'}),
            
            # Arrow
            html.Div([
                html.I(className='fas fa-arrow-right', style={
                    'fontSize': '24px',
                    'color': '#dc3545' if output_changed else '#6c757d'
                })
            ], style={'display': 'flex', 'alignItems': 'center', 'padding': '0 20px'}),
            
            # Ablated
            html.Div([
                html.Div("After Ablation", style={'fontSize': '12px', 'color': '#6c757d', 'marginBottom': '6px'}),
                html.Div(ablated_token, style={
                    'padding': '10px 16px',
                    'backgroundColor': '#ffebee' if output_changed else '#e8f5e9',
                    'border': f'2px solid {"#dc3545" if output_changed else "#4caf50"}',
                    'borderRadius': '6px',
                    'fontFamily': 'monospace',
                    'fontWeight': '600',
                    'textAlign': 'center'
                }),
                html.Div(f"{ablated_prob:.1%} ({prob_delta:+.1%})", style={
                    'fontSize': '12px',
                    'color': '#dc3545' if prob_delta < 0 else '#4caf50' if prob_delta > 0 else '#6c757d',
                    'marginTop': '4px',
                    'textAlign': 'center'
                })
            ], style={'flex': '1'})
            
        ], style={
            'display': 'flex',
            'alignItems': 'stretch',
            'padding': '16px',
            'backgroundColor': 'white',
            'borderRadius': '8px',
            'border': '1px solid #e2e8f0'
        }))
        
        # Interpretation
        results.append(html.Div([
            html.I(className='fas fa-lightbulb', style={'color': '#ffc107', 'marginRight': '8px'}),
            html.Span(
                "The prediction changed! These heads are important for this input."
                if output_changed else
                "The prediction stayed the same. These heads may not be critical for this specific input.",
                style={'color': '#6c757d', 'fontSize': '13px'}
            )
        ], style={'marginTop': '16px', 'padding': '12px', 'backgroundColor': '#fff8e1', 'borderRadius': '6px'}))
        
        # If a beam was selected, show context for comparison
        if selected_beam and selected_beam.get('text'):
            results.append(html.Div([
                html.Hr(style={'margin': '15px 0', 'borderColor': '#dee2e6'}),
                html.Div([
                    html.I(className='fas fa-info-circle', style={'color': '#6c757d', 'marginRight': '8px'}),
                    html.Span("Selected generation for reference: ", style={'fontWeight': '500', 'color': '#495057'}),
                    html.Span(selected_beam['text'], style={'fontFamily': 'monospace', 'backgroundColor': '#f8f9fa', 
                                                           'padding': '4px 8px', 'borderRadius': '4px'})
                ], style={'fontSize': '13px'})
            ]))
        
        return html.Div(results, style={
            'padding': '20px',
            'backgroundColor': '#fafbfc',
            'borderRadius': '8px',
            'border': '1px solid #e2e8f0'
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return html.Div(f"Ablation error: {str(e)}", style={'color': '#dc3545'})


# ============================================================================
# CALLBACKS: Investigation Panel - Attribution
# ============================================================================

@app.callback(
    Output('attribution-target-dropdown', 'options'),
    [Input('session-activation-store', 'data')]
)
def update_attribution_target_options(activation_data):
    if not activation_data:
        return []
    
    global_top5 = activation_data.get('global_top5_tokens', [])
    options = []
    for t in global_top5:
        if isinstance(t, dict):
            options.append({'label': f"{t['token']} ({t['probability']:.1%})", 'value': t['token']})
        else:
            options.append({'label': t[0], 'value': t[0]})
    return options


@app.callback(
    Output('attribution-results-container', 'children'),
    [Input('run-attribution-btn', 'n_clicks')],
    [State('attribution-method-radio', 'value'),
     State('attribution-target-dropdown', 'value'),
     State('session-activation-store', 'data'),
     State('model-dropdown', 'value'),
     State('prompt-input', 'value')],
    prevent_initial_call=True
)
def run_attribution_experiment(n_clicks, method, target_token, activation_data, model_name, prompt):
    if not n_clicks or not activation_data:
        return no_update
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation='eager')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.eval()
        
        sequence_text = activation_data.get('prompt', prompt)
        
        # Get target token ID if specified
        target_token_id = None
        if target_token:
            target_ids = tokenizer.encode(target_token, add_special_tokens=False)
            if target_ids:
                target_token_id = target_ids[-1]
        
        # Run attribution
        if method == 'integrated':
            attribution_result = compute_integrated_gradients(
                model, tokenizer, sequence_text, target_token_id, n_steps=30
            )
        else:
            attribution_result = compute_simple_gradient_attribution(
                model, tokenizer, sequence_text, target_token_id
            )
        
        return create_attribution_results_display(
            attribution_result, attribution_result['target_token']
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return html.Div(f"Attribution error: {str(e)}", style={'color': '#dc3545'})


# ============================================================================
# CALLBACKS: Reset Ablation
# ============================================================================

@app.callback(
    Output('reset-ablation-container', 'style'),
    Input('session-activation-store', 'data'),
    prevent_initial_call=False
)
def toggle_reset_ablation_button(activation_data):
    if activation_data and activation_data.get('ablated', False):
        return {'display': 'block'}
    return {'display': 'none'}


@app.callback(
    [Output('session-activation-store', 'data', allow_duplicate=True),
     Output('session-activation-store-original', 'data'),
     Output('model-status', 'children', allow_duplicate=True)],
    Input('reset-ablation-btn', 'n_clicks'),
    [State('session-activation-store-original', 'data')],
    prevent_initial_call=True
)
def reset_ablation(n_clicks, original_data):
    if not n_clicks or not original_data:
        return no_update, no_update, no_update
    
    success_message = html.Div([
        html.I(className="fas fa-undo", style={'marginRight': '8px', 'color': '#28a745'}),
        "Ablation reset"
    ], className="status-success")
    
    return original_data, {}, success_message


if __name__ == '__main__':
    app.run(debug=True, port=8050)
