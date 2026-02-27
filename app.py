"""
Transformer Explanation Dashboard

A dashboard focused on explaining how transformer models process input and arrive at predictions.
Uses a linear pipeline visualization with expandable stages.
"""

# Disable TensorFlow before any imports (fixes transformers/TF version incompatibility)
import os
os.environ["USE_TF"] = "0"

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import dash
from dash import html, dcc, Input, Output, State, callback, no_update, ALL, MATCH
import json
import torch
from utils import (load_model_and_get_patterns, execute_forward_pass, extract_layer_data,
                   perform_beam_search, execute_forward_pass_with_multi_layer_head_ablation)
from utils.head_detection import get_active_head_summary
from utils.model_config import get_auto_selections
from utils.token_attribution import compute_integrated_gradients, compute_simple_gradient_attribution

# Import modular components
from components.sidebar import create_sidebar
from components.model_selector import create_model_selector
from components.glossary import create_glossary_modal
from components.pipeline import (create_pipeline_container, create_tokenization_content,
                                  create_embedding_content, create_attention_content,
                                  create_mlp_content, create_output_content,
                                  _build_token_display, _build_top5_chart)
from components.investigation_panel import create_investigation_panel, create_attribution_results_display
from components.ablation_panel import create_selected_heads_display, create_ablation_results_display
from components.chatbot import create_chatbot_container, render_messages

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
                                html.Label("Options to Generate:", className="input-label"),
                                dcc.Slider(
                                    id='beam-width-slider',
                                    min=1, max=5, step=1, value=1,
                                    marks={1: '1', 3: '3', 5: '5'},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                ),
                                html.P("Generate multiple possible completions to compare",
                                       style={'color': '#6c757d', 'fontSize': '11px', 'marginTop': '4px', 'marginBottom': '0'})
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
    ], className="app-container"),
    
    # AI Chatbot
    create_chatbot_container()
], className="app-wrapper")


# ============================================================================
# CALLBACKS: Glossary
# ============================================================================

@app.callback(
    [Output("glossary-overlay-bg", "className"),
     Output("glossary-drawer-content", "className")],
    [Input("open-glossary-btn", "n_clicks"),
     Input("close-glossary-btn", "n_clicks"),
     Input("glossary-overlay-bg", "n_clicks")],
    prevent_initial_call=True
)
def toggle_glossary(open_clicks, close_clicks, overlay_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id == "open-glossary-btn":
        return "glossary-overlay open", "glossary-drawer open"
    else:
        return "glossary-overlay", "glossary-drawer"


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
     Output('session-activation-store-original', 'data', allow_duplicate=True),
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
                    results, {'display': 'none'}, {'display': 'none'}, {}, {}, original_prompt_data, {})

        # For single-token generation, analyze the full output (prompt + generated token)
        # so attention covers the entire sequence. For multi-token, start with prompt only;
        # the full-sequence analysis runs when the user selects a beam.
        if max_new_tokens == 1:
            full_text = results[0]['text']
            # Pass original_prompt so per-position top-5 is computed for the scrubber
            activation_data = execute_forward_pass(model, tokenizer, full_text, config,
                                                   original_prompt=prompt)
        else:
            full_text = prompt
            activation_data = execute_forward_pass(model, tokenizer, full_text, config)
        
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
                    activation_data, activation_data, original_prompt_data, {})
            
        else:
            # Single token generation - store the result as selected beam
            selected_beam_data = {'text': results[0]['text'], 'score': results[0].get('score', 0)}
            return (results_ui, results, {'display': 'block'}, {'display': 'block'}, 
                    activation_data, activation_data, original_prompt_data, selected_beam_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return (html.Div(f"Error: {e}", style={'color': 'red'}), [], 
                {'display': 'none'}, {'display': 'none'}, {}, {}, {}, {})


@app.callback(
    [Output('session-selected-beam-store', 'data', allow_duplicate=True),
     Output('generation-results-container', 'children', allow_duplicate=True),
     Output('session-activation-store', 'data', allow_duplicate=True),
     Output('session-activation-store-original', 'data', allow_duplicate=True)],
    Input({'type': 'result-item', 'index': ALL}, 'n_clicks'),
    [State('generation-results-store', 'data'),
     State('session-activation-store', 'data'),
     State('session-original-prompt-store', 'data')],
    prevent_initial_call=True
)
def store_selected_beam(n_clicks_list, results_data, existing_activation_data, original_prompt_data):
    """
    Store selected beam and re-run forward pass on the full sequence.
    
    When a beam is selected, this re-runs execute_forward_pass on the complete
    generated text (prompt + output) so the attention visualization covers
    the entire chosen output, not just the input.
    """
    if not any(n_clicks_list) or not results_data:
        return no_update, no_update, no_update
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update
        
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
    
    # Re-run forward pass on the full beam text so attention covers entire output
    new_activation_data = no_update
    if existing_activation_data:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model_name = existing_activation_data['model']
            config = {
                'attention_modules': existing_activation_data['attention_modules'],
                'block_modules': existing_activation_data['block_modules'],
                'norm_parameters': existing_activation_data.get('norm_parameters', [])
            }
            model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation='eager')
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model.eval()
            # Pass original_prompt so per-position top-5 data is computed for scrubber
            orig_prompt = original_prompt_data.get('prompt', '') if original_prompt_data else ''
            new_activation_data = execute_forward_pass(
                model, tokenizer, result['text'], config,
                original_prompt=orig_prompt
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Warning: Could not re-run forward pass for full sequence: {e}")
    
    # Store the selected beam for comparison after experiments
    return ({'text': result['text'], 'score': result.get('score', 0), 'index': index}, 
            selected_ui, new_activation_data, new_activation_data)


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
        
        # Generate BertViz HTML
        from utils import generate_bertviz_html
        attention_html = None
        try:
            attention_html = generate_bertviz_html(activation_data, 0, 'full')
        except:
            pass
        
        # Get head categorization from pre-computed JSON + runtime verification
        head_categories = None
        try:
            from utils.head_detection import get_active_head_summary
            head_categories = get_active_head_summary(activation_data, model_name)
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
        # Per-position data for the scrubber (populated when original_prompt was given)
        per_position_data = activation_data.get('per_position_top5', [])
        generated_tokens = activation_data.get('generated_tokens', [])
        scrubber_prompt = activation_data.get('original_prompt', original_prompt)

        outputs.append(f"→ {predicted_token}")
        outputs.append(create_output_content(
            top_tokens, predicted_token, predicted_prob,
            original_prompt=original_prompt,
            per_position_data=per_position_data,
            generated_tokens=generated_tokens,
            prompt_text=scrubber_prompt
        ))
        
        return tuple(outputs)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return tuple(empty_outputs)


# ============================================================================
# CALLBACKS: Output Scrubber
# ============================================================================

@app.callback(
    [Output('output-token-display', 'children'),
     Output('output-top5-chart', 'children')],
    [Input('output-scrubber-slider', 'value')],
    [State('session-activation-store', 'data')],
    prevent_initial_call=True
)
def update_output_scrubber(position, activation_data):
    """Update the token display and top-5 chart when the scrubber slider moves."""
    if activation_data is None or position is None:
        return no_update, no_update

    per_position_data = activation_data.get('per_position_top5', [])
    generated_tokens = activation_data.get('generated_tokens', [])
    prompt_text = activation_data.get('original_prompt', activation_data.get('prompt', ''))

    if not per_position_data or not generated_tokens:
        return no_update, no_update

    # Clamp position to valid range
    position = max(0, min(position, len(per_position_data) - 1))
    pos_data = per_position_data[position]

    token_display = _build_token_display(
        prompt_text, generated_tokens, position, pos_data['actual_prob']
    )
    top5_chart = _build_top5_chart(pos_data['top5'], pos_data.get('actual_token'))

    return token_display, top5_chart


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
# CALLBACKS: Investigation Panel - Ablation (Updated for New UI)
# ============================================================================

@app.callback(
    [Output('ablation-layer-select', 'options'),
     Output('ablation-head-select', 'options')],
    [Input('session-activation-store', 'data'),
     Input('ablation-layer-select', 'value')],
    [State('model-dropdown', 'value')]
)
def update_ablation_selectors(activation_data, selected_layer, model_name):
    """Update options for layer and head dropdowns."""
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    if not activation_data or not model_name:
        return [], []
    
    # Get model config
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
    except:
        return [], []
    
    # Update layer options (only if data changed or init)
    layer_options = [{'label': f'Layer {i}', 'value': i} for i in range(num_layers)]
    
    # Update head options based on selected layer
    head_options = []
    if selected_layer is not None:
        head_options = [{'label': f'Head {i}', 'value': i} for i in range(num_heads)]
    
    # If only layer changed, return no_update for layer options to avoid flickering
    if trigger_id == 'ablation-layer-select':
        return no_update, head_options
        
    return layer_options, head_options


@app.callback(
    [Output('ablation-selected-heads', 'data'),
     Output('run-ablation-btn', 'disabled'),
     Output('ablation-selected-display', 'children'),
     Output('ablation-head-select', 'value')],
    [Input('ablation-add-head-btn', 'n_clicks'),
     Input('clear-ablation-btn', 'n_clicks'),
     Input({'type': 'ablation-remove-btn', 'layer': ALL, 'head': ALL}, 'n_clicks')],
    [State('ablation-layer-select', 'value'),
     State('ablation-head-select', 'value'),
     State('ablation-selected-heads', 'data')],
    prevent_initial_call=True
)
def manage_ablation_heads(add_clicks, clear_clicks, remove_clicks, 
                         layer_val, head_val, selected_heads):
    """
    Manage the list of selected heads (Add, Clear, Remove).
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update, no_update
    
    triggered_id = ctx.triggered[0]['prop_id']
    
    # Initialize selected_heads
    if selected_heads is None:
        selected_heads = []
    
    # Handle "Add"
    if 'ablation-add-head-btn' in triggered_id:
        if layer_val is not None and head_val is not None:
            # Check for duplicate
            exists = any(item['layer'] == layer_val and item['head'] == head_val 
                        for item in selected_heads if isinstance(item, dict))
            if not exists:
                selected_heads.append({'layer': layer_val, 'head': head_val})
                # Sort for cleaner display
                selected_heads.sort(key=lambda x: (x['layer'], x['head']))
        # Clear head selection after add
        return selected_heads, len(selected_heads) == 0, create_selected_heads_display(selected_heads), None
        
    # Handle "Clear"
    elif 'clear-ablation-btn' in triggered_id:
        return [], True, create_selected_heads_display([]), no_update
        
    # Handle "Remove" (pattern matching)
    elif 'ablation-remove-btn' in triggered_id:
        # Determine which button was clicked
        # Note: remove_clicks is a list, but we need the id from context
        prop_id = json.loads(triggered_id.split('.')[0])
        rm_layer = prop_id['layer']
        rm_head = prop_id['head']
        
        selected_heads = [
            item for item in selected_heads 
            if not (isinstance(item, dict) and item.get('layer') == rm_layer and item.get('head') == rm_head)
        ]
        return selected_heads, len(selected_heads) == 0, create_selected_heads_display(selected_heads), no_update
    
    return no_update, no_update, no_update, no_update


@app.callback(
    [Output('ablation-results-container', 'children'),
     Output('session-activation-store', 'data', allow_duplicate=True),
     Output('session-activation-store-original', 'data', allow_duplicate=True)],
    [Input('run-ablation-btn', 'n_clicks')],
    [State('ablation-selected-heads', 'data'),
     State('session-activation-store', 'data'),
     State('model-dropdown', 'value'),
     State('prompt-input', 'value'),
     State('session-selected-beam-store', 'data'),
     State('max-new-tokens-slider', 'value'),
     State('beam-width-slider', 'value')],
    prevent_initial_call=True
)
def run_ablation_experiment(n_clicks, selected_heads, activation_data, model_name, prompt, selected_beam, max_new_tokens, beam_width):
    """Run ablation on ORIGINAL PROMPT and compare results, including beam generation."""
    if not n_clicks or not selected_heads or not activation_data:
        return no_update, no_update, no_update
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation='eager')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.eval()
        
        sequence_text = prompt
        
        config = {
            'attention_modules': activation_data.get('attention_modules', []),
            'block_modules': activation_data.get('block_modules', []),
            'norm_parameters': activation_data.get('norm_parameters', [])
        }
        
        # Original output
        original_output = activation_data.get('actual_output', {})
        original_token = original_output.get('token', '')
        original_prob = original_output.get('probability', 0)
        
        # Group heads by layer
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
            return html.Div("No valid heads selected.", style={'color': '#dc3545'}), no_update, no_update
        
        # Run ablation for generation
        ablated_beam = None
        analysis_text = sequence_text
        try:
            # Always perform beam search during ablation to show comparison
            beam_results = perform_beam_search(
                model, tokenizer, sequence_text, 
                beam_width=beam_width, 
                max_new_tokens=max_new_tokens,
                ablation_config=heads_by_layer
            )
            if beam_results:
                # Select the top beam
                ablated_beam = {'text': beam_results[0]['text'], 'score': beam_results[0].get('score', 0)}
                analysis_text = ablated_beam['text']
        except Exception as e:
            print(f"Error during ablated generation: {e}")
            
        # Run ablation for analysis (single pass) on the final generated text
        ablated_data = execute_forward_pass_with_multi_layer_head_ablation(
            model, tokenizer, analysis_text, config, heads_by_layer, original_prompt=prompt
        )
        
        # Mark as ablated so UI knows
        ablated_data['ablated'] = True
        
        ablated_output = ablated_data.get('actual_output', {})
        ablated_token = ablated_output.get('token', '')
        ablated_prob = ablated_output.get('probability', 0)
        
        results_display = create_ablation_results_display(
            activation_data, ablated_data,
            selected_heads, selected_beam, ablated_beam
        )
        
        return results_display, ablated_data, activation_data
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return html.Div(f"Ablation error: {str(e)}", style={'color': '#dc3545'}), no_update, no_update


@app.callback(
    [Output('ablation-original-token-map', 'children'),
     Output('ablation-original-text-box', 'children'),
     Output('ablation-original-top5-chart', 'figure'),
     Output('ablation-ablated-token-map', 'children'),
     Output('ablation-ablated-text-box', 'children'),
     Output('ablation-ablated-top5-chart', 'figure'),
     Output('ablation-divergence-indicator', 'children')],
    [Input('ablation-scrubber-slider', 'value')],
    [State('session-activation-store-original', 'data'),
     State('session-activation-store', 'data')],
    prevent_initial_call=True
)
def update_ablation_scrubber(position, original_data, ablated_data):
    if position is None or not original_data or not ablated_data:
        import dash
        return dash.no_update
        
    orig_pos_data = original_data.get('per_position_top5', [])
    abl_pos_data = ablated_data.get('per_position_top5', [])
    
    orig_tokens = original_data.get('generated_tokens', [])
    abl_tokens = ablated_data.get('generated_tokens', [])
    
    # Helper to build token map
    def build_token_map(tokens, current_pos, changed_indices):
        from dash import html
        elements = []
        for i, token in enumerate(tokens):
            if i > 0: elements.append(html.Span(" → ", style={'color': '#ced4da', 'margin': '0 4px'}))
            
            is_current = i == current_pos
            is_changed = i in changed_indices
            
            style = {'fontWeight': 'bold' if is_current else 'normal'}
            if is_current:
                style['color'] = '#ffffff'
                style['backgroundColor'] = '#dc3545' if is_changed else '#28a745'
                style['padding'] = '2px 6px'
                style['borderRadius'] = '4px'
            elif is_changed:
                style['color'] = '#dc3545'
                
            elements.append(html.Span(f"T{i} ({token.strip()})", style=style))
        return elements

    # Helper to build text box
    def build_text_box(prompt_text, tokens, current_pos, changed_indices):
        from dash import html
        elements = [html.Span(prompt_text, style={'color': '#6c757d'})]
        for i, token in enumerate(tokens):
            is_current = i == current_pos
            is_changed = i in changed_indices
            
            style = {}
            if is_current:
                style['backgroundColor'] = '#ffc107' if is_changed else '#0dcaf0'
                style['color'] = '#000'
                style['borderRadius'] = '3px'
                style['padding'] = '0 2px'
                style['fontWeight'] = 'bold'
                
            elements.append(html.Span(token, style=style))
        return elements
        
    def build_chart(pos_data, actual_token, main_color):
        import plotly.graph_objs as go
        if not pos_data: return go.Figure().update_layout(margin=dict(l=0, r=0, t=0, b=0), height=200)
        
        top5 = pos_data.get('top5', [])
        tokens = [t['token'] for t in reversed(top5)]
        probs = [t['probability'] for t in reversed(top5)]
        
        colors = []
        for t in tokens:
            if t == actual_token:
                colors.append(main_color)
            else:
                colors.append('#e2e8f0' if main_color == '#4c51bf' else '#f8d7da')
                
        fig = go.Figure(go.Bar(
            x=probs, y=tokens, orientation='h', marker_color=colors,
            text=[f"{p:.1%}" for p in probs], textposition='auto'
        ))
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0), height=200,
            xaxis=dict(visible=False, range=[0, 1]),
            yaxis=dict(tickfont=dict(size=12)),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        return fig

    # Find changed indices
    changed_indices = set()
    for i in range(max(len(orig_tokens), len(abl_tokens))):
        if i >= len(orig_tokens) or i >= len(abl_tokens) or orig_tokens[i] != abl_tokens[i]:
            changed_indices.add(i)
            
    prompt_text = original_data.get('original_prompt', original_data.get('prompt', ''))
    
    orig_map = build_token_map(orig_tokens, position, set()) # original doesn't show red 'changed' state
    abl_map = build_token_map(abl_tokens, position, changed_indices)
    
    orig_text_box = build_text_box(prompt_text, orig_tokens, position, set())
    abl_text_box = build_text_box(prompt_text, abl_tokens, position, changed_indices)
    
    orig_chart = []
    abl_chart = []
    
    from dash import html
    divergence_indicator = html.Div()
    
    if position < len(orig_pos_data):
        orig_chart = build_chart(orig_pos_data[position], orig_pos_data[position].get('actual_token'), '#4c51bf')
    if position < len(abl_pos_data):
        abl_chart = build_chart(abl_pos_data[position], abl_pos_data[position].get('actual_token'), '#e53e3e')
        
    is_diverged = position in changed_indices
    if is_diverged:
        divergence_indicator = html.Div([
            html.I(className="fas fa-exclamation-circle", style={'color': '#dc3545', 'fontSize': '32px', 'backgroundColor': '#fff5f5', 'borderRadius': '50%', 'padding': '10px', 'boxShadow': '0 0 15px rgba(220,53,69,0.4)'})
        ])
    else:
        divergence_indicator = html.Div([
            html.I(className="fas fa-check-circle", style={'color': '#28a745', 'fontSize': '32px', 'backgroundColor': '#f0fdf4', 'borderRadius': '50%', 'padding': '10px'})
        ])
        
    return orig_map, orig_text_box, orig_chart, abl_map, abl_text_box, abl_chart, divergence_indicator


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


# ============================================================================
# CALLBACKS: AI Chatbot
# ============================================================================
import threading
import queue

_chat_stream_queue = queue.Queue()
_chat_stream_active = False
_chat_stream_content = ""

def _background_generate(user_input, chat_history, rag_context, dashboard_context):
    global _chat_stream_active
    try:
        from utils.openrouter_client import generate_stream
        for chunk in generate_stream(user_input, chat_history, rag_context, dashboard_context):
            _chat_stream_queue.put(chunk)
    except Exception as e:
        _chat_stream_queue.put(f"\n[Error: {str(e)}]")
    finally:
        _chat_stream_active = False


@app.callback(
    [Output('chat-window', 'style'),
     Output('chat-open-store', 'data'),
     Output('chat-toggle-btn', 'style')],
    [Input('chat-toggle-btn', 'n_clicks'),
     Input('chat-close-btn', 'n_clicks')],
    [State('chat-open-store', 'data')],
    prevent_initial_call=True
)
def toggle_chat_window(toggle_clicks, close_clicks, is_open):
    """Toggle the chat window open/closed."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Default toggle button style (visible)
    toggle_visible = {}
    # Hidden toggle button style
    toggle_hidden = {'display': 'none'}
    
    if trigger_id == 'chat-close-btn':
        return {'display': 'none'}, False, toggle_visible
    
    # Toggle button clicked
    new_state = not is_open
    if new_state:
        # Open chat, hide toggle button
        return {'display': 'flex'}, True, toggle_hidden
    else:
        # Close chat, show toggle button
        return {'display': 'none'}, False, toggle_visible


@app.callback(
    Output('chat-history-store', 'data', allow_duplicate=True),
    Input('chat-clear-btn', 'n_clicks'),
    prevent_initial_call=True
)
def clear_chat_history(n_clicks):
    """Clear all chat history."""
    if not n_clicks:
        return no_update
    return []


@app.callback(
    [Output('chat-messages-list', 'children'),
     Output('chat-history-store', 'data', allow_duplicate=True),
     Output('chat-input', 'value'),
     Output('chat-typing-indicator', 'style'),
     Output('chat-response-interval', 'disabled')],
    [Input('chat-send-btn', 'n_clicks')],
    [State('chat-input', 'value'),
     State('chat-history-store', 'data'),
     State('model-dropdown', 'value'),
     State('prompt-input', 'value'),
     State('session-activation-store', 'data'),
     State('ablation-selected-heads', 'data')],
    prevent_initial_call=True
)
def send_chat_message(send_clicks, user_input, chat_history, 
                      model_name, prompt, activation_data, ablated_heads):
    """Handle sending a chat message and start stream."""
    global _chat_stream_active, _chat_stream_content
    from utils.rag_utils import build_rag_context
    
    if not user_input or not user_input.strip():
        return no_update, no_update, no_update, no_update, no_update
    
    user_input = user_input.strip()
    
    if chat_history is None:
        chat_history = []
    
    chat_history.append({'role': 'user', 'content': user_input})
    chat_history.append({'role': 'assistant', 'content': ''})
    
    dashboard_context = {}
    if model_name: dashboard_context['model'] = model_name
    if prompt: dashboard_context['prompt'] = prompt
    if activation_data:
        actual_output = activation_data.get('actual_output', {})
        if actual_output:
            dashboard_context['predicted_token'] = actual_output.get('token', '')
            dashboard_context['predicted_probability'] = actual_output.get('probability', 0)
        top5 = activation_data.get('global_top5_tokens', [])
        if top5: dashboard_context['top_predictions'] = top5
    if ablated_heads: dashboard_context['ablated_heads'] = ablated_heads
    
    rag_context = build_rag_context(user_input, top_k=3)
    
    _chat_stream_content = ""
    while not _chat_stream_queue.empty():
        try: _chat_stream_queue.get_nowait()
        except: pass
    
    _chat_stream_active = True
    threading.Thread(
        target=_background_generate, 
        args=(user_input, chat_history[:-2], rag_context, dashboard_context), 
        daemon=True
    ).start()
    
    messages_ui = render_messages(chat_history)
    
    return messages_ui, chat_history, '', {'display': 'flex'}, False


@app.callback(
    [Output('chat-messages-list', 'children', allow_duplicate=True),
     Output('chat-history-store', 'data', allow_duplicate=True),
     Output('chat-response-interval', 'disabled', allow_duplicate=True),
     Output('chat-typing-indicator', 'style', allow_duplicate=True)],
    [Input('chat-response-interval', 'n_intervals')],
    [State('chat-history-store', 'data')],
    prevent_initial_call=True
)
def update_stream(n_intervals, chat_history):
    global _chat_stream_active, _chat_stream_content
    
    if not chat_history:
        return no_update, no_update, True, no_update
        
    chunks = []
    while not _chat_stream_queue.empty():
        try:
            chunks.append(_chat_stream_queue.get_nowait())
        except queue.Empty:
            break
            
    if chunks:
        _chat_stream_content += "".join(chunks)
        if chat_history[-1]['role'] == 'assistant':
            chat_history[-1]['content'] = _chat_stream_content
            
        messages_ui = render_messages(chat_history)
        return messages_ui, chat_history, False, {'display': 'none'}
        
    elif not _chat_stream_active:
        return no_update, no_update, True, {'display': 'none'}
        
    return no_update, no_update, False, {'display': 'flex'}


@app.callback(
    [Output('chat-messages-list', 'children', allow_duplicate=True),
     Output('chat-history-store', 'data', allow_duplicate=True)],
    Input('chat-history-store', 'data'),
    prevent_initial_call='initial_duplicate'  # Allows initial call with allow_duplicate
)
def update_messages_from_store(chat_history):
    """Update message display when history changes (e.g., on page load from localStorage)."""
    from components.chatbot import GREETING_MESSAGE
    
    # If history is empty (localStorage empty/cleared), inject the greeting
    if not chat_history:
        greeting_history = [{'role': 'assistant', 'content': GREETING_MESSAGE}]
        return render_messages(greeting_history), greeting_history
    
    return render_messages(chat_history), no_update


# Client-side callback for scroll down
app.clientside_callback(
    """
    function(children) {
        setTimeout(() => {
            const container = document.getElementById('chat-messages-container');
            if (container) {
                container.scrollTop = container.scrollHeight;
            }
        }, 50);
        return window.dash_clientside.no_update;
    }
    """,
    Output('chat-messages-container', 'className'),
    Input('chat-messages-list', 'children'),
    prevent_initial_call=True
)

# Client-side callback for copy functionality
app.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) return window.dash_clientside.no_update;
        
        // Find the clicked button and get its data-content attribute
        const triggered = window.dash_clientside.callback_context.triggered;
        if (!triggered || triggered.length === 0) return window.dash_clientside.no_update;
        
        const propId = triggered[0].prop_id;
        const match = propId.match(/{"index":(\d+),"type":"copy-message-btn"}/);
        if (!match) return window.dash_clientside.no_update;
        
        const btn = document.querySelector(`[id='{"index":${match[1]},"type":"copy-message-btn"}']`);
        if (!btn) return window.dash_clientside.no_update;
        
        const content = btn.getAttribute('data-content');
        if (content) {
            navigator.clipboard.writeText(content).then(() => {
                btn.classList.add('copied');
                setTimeout(() => btn.classList.remove('copied'), 2000);
            });
        }
        
        return window.dash_clientside.no_update;
    }
    """,
    Output('chat-messages-list', 'className'),
    Input({'type': 'copy-message-btn', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True
)


# Client-side callback for Enter key to send message
app.clientside_callback(
    """
    function(id) {
        // Set up the event listener only once
        const textarea = document.getElementById('chat-input');
        const sendBtn = document.getElementById('chat-send-btn');
        
        if (textarea && sendBtn && !textarea._enterListenerAdded) {
            textarea.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendBtn.click();
                }
            });
            textarea._enterListenerAdded = true;
        }
        
        return window.dash_clientside.no_update;
    }
    """,
    Output('chat-input', 'className'),
    Input('chat-open-store', 'data'),
    prevent_initial_call=True
)


if __name__ == '__main__':
    # Use 0.0.0.0:7860 for Hugging Face Spaces, fallback to localhost:8050 for local dev
    import os
    port = int(os.environ.get("PORT", 7860))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    app.run(host='0.0.0.0', port=port, debug=debug)
