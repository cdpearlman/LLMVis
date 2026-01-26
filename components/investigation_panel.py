"""
Investigation Panel component for exploring model behavior.

Consolidates two main investigation tools:
1. Ablation - Remove attention heads/layers to see their impact
2. Token Attribution - Identify which input tokens influenced the output
"""

from dash import html, dcc
import plotly.graph_objs as go


def create_investigation_panel():
    """
    Create the main investigation panel with tabs for different tools.
    """
    return html.Div([
        # Section header
        html.Div([
            html.H3("Investigate the Results", className="section-title"),
            html.P("Use these tools to understand why the model made its prediction.",
                   style={'color': '#6c757d', 'fontSize': '14px', 'marginBottom': '1.5rem'})
        ]),
        
        # Tab buttons
        html.Div([
            html.Button([
                html.I(className='fas fa-cut', style={'marginRight': '8px'}),
                "Ablation"
            ], id='investigation-tab-ablation', className='investigation-tab active',
               n_clicks=0, style=get_tab_style(True)),
            
            html.Button([
                html.I(className='fas fa-highlighter', style={'marginRight': '8px'}),
                "Token Attribution"
            ], id='investigation-tab-attribution', className='investigation-tab',
               n_clicks=0, style=get_tab_style(False))
        ], className='investigation-tabs', style={
            'display': 'flex',
            'gap': '8px',
            'marginBottom': '1.5rem'
        }),
        
        # Store for active tab
        dcc.Store(id='investigation-active-tab', data='ablation'),
        
        # Tab content containers
        html.Div([
            # Ablation tab content
            html.Div(
                id='investigation-ablation-content',
                children=create_ablation_content(),
                style={'display': 'block'}
            ),
            
            # Attribution tab content
            html.Div(
                id='investigation-attribution-content',
                children=create_attribution_content(),
                style={'display': 'none'}
            )
        ], className='investigation-content')
        
    ], id='investigation-panel', className='investigation-panel', style={'display': 'none'})


def get_tab_style(is_active):
    """Get style for tab button based on active state."""
    base_style = {
        'padding': '10px 20px',
        'border': 'none',
        'borderRadius': '6px',
        'cursor': 'pointer',
        'fontSize': '14px',
        'fontWeight': '500',
        'transition': 'all 0.2s'
    }
    
    if is_active:
        base_style.update({
            'backgroundColor': '#667eea',
            'color': 'white'
        })
    else:
        base_style.update({
            'backgroundColor': '#f8f9fa',
            'color': '#495057'
        })
    
    return base_style


def create_ablation_content():
    """Create the ablation tool content."""
    return html.Div([
        # Explanation
        html.Div([
            html.H5("What is Ablation?", style={'color': '#495057', 'marginBottom': '8px'}),
            html.P([
                "Ablation lets you ", html.Strong("remove specific attention heads"),
                " to see how they affect the model's output. If removing a head changes the prediction significantly, ",
                "that head was important for this particular input."
            ], style={'color': '#6c757d', 'fontSize': '14px', 'marginBottom': '16px'})
        ]),
        
        # Layer selector
        html.Div([
            html.Label("Select Layer:", className="input-label", style={'marginBottom': '8px', 'display': 'block'}),
            dcc.Dropdown(
                id='ablation-layer-dropdown',
                options=[],  # Populated by callback
                value=None,
                placeholder="Choose a layer to view heads...",
                className="module-dropdown"
            )
        ], style={'marginBottom': '16px'}),
        
        # Head selector grid
        html.Div([
            html.Label("Select Heads to Ablate:", className="input-label", style={'marginBottom': '8px', 'display': 'block'}),
            html.Div(id='ablation-head-grid', children=[
                html.P("Select a layer first.", style={'color': '#6c757d', 'fontStyle': 'italic'})
            ], style={
                'padding': '16px',
                'backgroundColor': '#f8f9fa',
                'borderRadius': '8px',
                'border': '1px solid #e2e8f0'
            })
        ], style={'marginBottom': '16px'}),
        
        # Selected heads display (chips with remove buttons)
        html.Div([
            html.Label("Selected Heads:", className="input-label", style={'marginBottom': '8px', 'display': 'block'}),
            html.Div(id='ablation-selected-display', children=[
                html.Span("No heads selected yet", style={'color': '#6c757d', 'fontSize': '13px', 'fontStyle': 'italic'})
            ], style={
                'padding': '12px',
                'backgroundColor': '#f8f9fa',
                'borderRadius': '8px',
                'border': '1px solid #e2e8f0',
                'minHeight': '40px'
            })
        ], style={'marginBottom': '16px'}),
        
        # Run ablation button
        html.Button([
            html.I(className='fas fa-play', style={'marginRight': '8px'}),
            "Run Ablation"
        ], id='run-ablation-btn', className='action-button primary-button',
           disabled=True, style={'width': '100%', 'marginBottom': '16px'}),
        
        # Results container
        html.Div(id='ablation-results-container', children=[
            # Will be populated by callback
        ])
        
    ], className='ablation-tool')


def create_attribution_content():
    """Create the token attribution tool content."""
    return html.Div([
        # Explanation
        html.Div([
            html.H5("What is Token Attribution?", style={'color': '#495057', 'marginBottom': '8px'}),
            html.P([
                "Token attribution uses ", html.Strong("gradient analysis"),
                " to identify which input tokens had the most influence on the model's prediction. ",
                "Tokens with higher attribution contributed more to the final output."
            ], style={'color': '#6c757d', 'fontSize': '14px', 'marginBottom': '16px'})
        ]),
        
        # Method selector
        html.Div([
            html.Label("Attribution Method:", className="input-label", style={'marginBottom': '8px', 'display': 'block'}),
            dcc.RadioItems(
                id='attribution-method-radio',
                options=[
                    {'label': ' Integrated Gradients (more accurate, slower)', 'value': 'integrated'},
                    {'label': ' Simple Gradient (faster, less accurate)', 'value': 'simple'}
                ],
                value='simple',
                style={'display': 'flex', 'flexDirection': 'column', 'gap': '8px'}
            )
        ], style={'marginBottom': '16px'}),
        
        # Target token selector
        html.Div([
            html.Label("Target Token:", className="input-label", style={'marginBottom': '8px', 'display': 'block'}),
            dcc.Dropdown(
                id='attribution-target-dropdown',
                options=[],  # Populated by callback with top-5 predictions
                value=None,
                placeholder="Use top predicted token (default)",
                className="module-dropdown"
            ),
            html.P("Leave empty to compute attribution for the top predicted token.",
                  style={'color': '#6c757d', 'fontSize': '12px', 'marginTop': '4px'})
        ], style={'marginBottom': '16px'}),
        
        # Run attribution button
        html.Button([
            html.I(className='fas fa-highlighter', style={'marginRight': '8px'}),
            "Compute Attribution"
        ], id='run-attribution-btn', className='action-button primary-button',
           style={'width': '100%', 'marginBottom': '16px'}),
        
        # Results container
        html.Div(id='attribution-results-container', children=[
            # Will be populated by callback
        ])
        
    ], className='attribution-tool')


def create_ablation_head_buttons(num_heads, layer_num, selected_heads=None):
    """
    Create grid of head selection buttons for ablation.
    
    Args:
        num_heads: Number of attention heads in the layer
        layer_num: Layer number for button IDs
        selected_heads: List of currently selected head dicts [{layer: N, head: M}, ...] 
                       or list of head indices for backward compatibility
    """
    if selected_heads is None:
        selected_heads = []
    
    # Get head indices selected for THIS layer
    current_layer_heads = []
    for item in selected_heads:
        if isinstance(item, dict):
            if item.get('layer') == layer_num:
                current_layer_heads.append(item.get('head'))
        else:
            # Backward compatibility: assume it's just a head index
            current_layer_heads.append(item)
    
    buttons = []
    for h in range(num_heads):
        is_selected = h in current_layer_heads
        buttons.append(
            html.Button(
                f"H{h}",
                id={'type': 'ablation-head-btn', 'layer': layer_num, 'head': h},
                n_clicks=0,
                style={
                    'padding': '8px 14px',
                    'margin': '4px',
                    'border': '2px solid #667eea' if is_selected else '1px solid #dee2e6',
                    'borderRadius': '6px',
                    'backgroundColor': '#667eea' if is_selected else 'white',
                    'color': 'white' if is_selected else '#495057',
                    'cursor': 'pointer',
                    'fontWeight': '600' if is_selected else '400',
                    'fontSize': '13px',
                    'transition': 'all 0.2s'
                }
            )
        )
    
    return html.Div(buttons, style={
        'display': 'flex',
        'flexWrap': 'wrap',
        'gap': '4px'
    })


def create_selected_heads_display(selected_heads):
    """
    Create display of selected heads as chips with remove buttons.
    
    Args:
        selected_heads: List of {layer: N, head: M} dicts
    
    Returns:
        Dash HTML component showing selected heads as removable chips
    """
    if not selected_heads:
        return html.Div(
            "No heads selected yet",
            style={'color': '#6c757d', 'fontSize': '13px', 'fontStyle': 'italic', 'padding': '8px 0'}
        )
    
    chips = []
    for item in selected_heads:
        layer = item.get('layer')
        head = item.get('head')
        label = f"L{layer}-H{head}"
        
        chips.append(
            html.Span([
                html.Span(label, style={'marginRight': '6px'}),
                html.Button(
                    '×',
                    id={'type': 'ablation-remove-btn', 'layer': layer, 'head': head},
                    n_clicks=0,
                    style={
                        'background': 'none',
                        'border': 'none',
                        'color': '#667eea',
                        'cursor': 'pointer',
                        'fontSize': '16px',
                        'fontWeight': 'bold',
                        'padding': '0',
                        'lineHeight': '1',
                        'verticalAlign': 'middle'
                    }
                )
            ], style={
                'display': 'inline-flex',
                'alignItems': 'center',
                'padding': '6px 10px',
                'margin': '4px',
                'backgroundColor': '#667eea20',
                'border': '1px solid #667eea40',
                'borderRadius': '16px',
                'fontSize': '12px',
                'fontFamily': 'monospace',
                'fontWeight': '500'
            })
        )
    
    return html.Div(chips, style={
        'display': 'flex',
        'flexWrap': 'wrap',
        'gap': '4px',
        'padding': '8px 0'
    })


def create_ablation_results_display(original_output, ablated_output, original_prob, ablated_prob,
                                    ablated_layer, ablated_heads):
    """
    Create the ablation results display.
    
    Args:
        original_output: Original predicted token
        ablated_output: Predicted token after ablation
        original_prob: Original prediction probability
        ablated_prob: Ablated prediction probability
        ablated_layer: Layer number that was ablated
        ablated_heads: List of head indices that were ablated
    """
    output_changed = original_output != ablated_output
    prob_delta = ablated_prob - original_prob
    
    return html.Div([
        html.H5("Ablation Results", style={'color': '#495057', 'marginBottom': '16px'}),
        
        # Summary
        html.Div([
            html.Span("Ablated: ", style={'color': '#6c757d'}),
            html.Span(f"Layer {ablated_layer}, Heads {', '.join(f'H{h}' for h in ablated_heads)}",
                     style={'fontWeight': '500', 'color': '#667eea'})
        ], style={'marginBottom': '16px'}),
        
        # Before/After comparison
        html.Div([
            # Original
            html.Div([
                html.Div("Original", style={'fontSize': '12px', 'color': '#6c757d', 'marginBottom': '6px'}),
                html.Div(original_output, style={
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
                html.Div(ablated_output, style={
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
        }),
        
        # Interpretation
        html.Div([
            html.I(className='fas fa-lightbulb', style={'color': '#ffc107', 'marginRight': '8px'}),
            html.Span(
                "The prediction changed! These heads are important for this input."
                if output_changed else
                "The prediction stayed the same. These heads may not be critical for this specific input.",
                style={'color': '#6c757d', 'fontSize': '13px'}
            )
        ], style={'marginTop': '16px', 'padding': '12px', 'backgroundColor': '#fff8e1', 'borderRadius': '6px'})
        
    ], style={
        'padding': '20px',
        'backgroundColor': '#fafbfc',
        'borderRadius': '8px',
        'border': '1px solid #e2e8f0'
    })


def create_attribution_results_display(attribution_data, target_token):
    """
    Create the token attribution results display.
    
    Args:
        attribution_data: Output from token_attribution.py functions
        target_token: The token that attribution was computed for
    """
    tokens = attribution_data['tokens']
    normalized = attribution_data['normalized_attributions']
    
    # Create token chips with color intensity
    token_chips = []
    for tok, norm in zip(tokens, normalized):
        # Color gradient from light blue (low) to dark blue (high)
        intensity = int(255 * (1 - norm * 0.7))
        bg_color = f'rgb({intensity}, {intensity}, 255)'
        text_color = 'white' if norm > 0.5 else '#333'
        
        token_chips.append(
            html.Span(tok, style={
                'display': 'inline-block',
                'padding': '6px 12px',
                'margin': '3px',
                'backgroundColor': bg_color,
                'color': text_color,
                'borderRadius': '4px',
                'fontFamily': 'monospace',
                'fontSize': '13px',
                'fontWeight': '500' if norm > 0.3 else '400'
            }, title=f"Attribution: {norm:.2f}")
        )
    
    # Create bar chart
    fig = go.Figure(go.Bar(
        x=normalized,
        y=tokens,
        orientation='h',
        marker_color=[f'rgba(102, 126, 234, {0.3 + n * 0.7})' for n in normalized],
        text=[f"{n:.2f}" for n in normalized],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Attribution Scores by Token",
        xaxis_title="Attribution (normalized)",
        yaxis_title="Input Token",
        height=max(200, len(tokens) * 30),
        margin=dict(l=20, r=60, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(autorange='reversed')
    )
    
    return html.Div([
        html.H5("Token Attribution Results", style={'color': '#495057', 'marginBottom': '8px'}),
        html.P([
            "Attribution for predicting: ",
            html.Span(target_token, style={
                'padding': '4px 10px',
                'backgroundColor': '#667eea',
                'color': 'white',
                'borderRadius': '4px',
                'fontFamily': 'monospace',
                'fontWeight': '600'
            })
        ], style={'marginBottom': '16px'}),
        
        # Token chips visualization
        html.Div([
            html.H6("Input tokens (darker = more important):", style={'color': '#6c757d', 'marginBottom': '8px'}),
            html.Div(token_chips, style={'lineHeight': '2'})
        ], style={
            'padding': '16px',
            'backgroundColor': 'white',
            'borderRadius': '8px',
            'border': '1px solid #e2e8f0',
            'marginBottom': '16px'
        }),
        
        # Bar chart
        html.Div([
            dcc.Graph(figure=fig, config={'displayModeBar': False})
        ], style={
            'backgroundColor': 'white',
            'borderRadius': '8px',
            'border': '1px solid #e2e8f0'
        }),
        
        # Interpretation
        html.Div([
            html.I(className='fas fa-info-circle', style={'color': '#667eea', 'marginRight': '8px'}),
            html.Span(
                "Tokens with higher attribution scores contributed more to the model's prediction. "
                "This helps identify which parts of the input were most influential.",
                style={'color': '#6c757d', 'fontSize': '13px'}
            )
        ], style={'marginTop': '16px', 'padding': '12px', 'backgroundColor': '#e8eaf6', 'borderRadius': '6px'})
        
    ], style={
        'padding': '20px',
        'backgroundColor': '#fafbfc',
        'borderRadius': '8px',
        'border': '1px solid #e2e8f0'
    })

