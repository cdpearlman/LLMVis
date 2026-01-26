"""
Pipeline visualization component for explaining transformer model data flow.

Provides a linear, expandable visualization of:
1. Tokenization (reuses tokenization_panel)
2. Embedding
3. Attention
4. MLP/Feed-Forward
5. Output Selection
"""

from dash import html, dcc
import plotly.graph_objs as go


def create_pipeline_container():
    """
    Create the main pipeline visualization container.
    
    Each stage is a clickable, expandable section that shows:
    - Collapsed: Icon + title + summary metric
    - Expanded: Detailed explanation + visualization
    """
    return html.Div([
        # Pipeline header
        html.Div([
            html.H3("How the Model Processes Your Input", className="section-title"),
            html.P("Click any stage to expand and see details about how your text flows through the model.",
                   style={'color': '#6c757d', 'fontSize': '14px', 'marginBottom': '1.5rem'})
        ]),
        
        # Visual flow indicator (always visible)
        create_flow_indicator(),
        
        # Stage containers (populated by callbacks)
        html.Div([
            # Stage 1: Tokenization
            create_stage_container(
                stage_id='tokenization',
                stage_num=1,
                title='Tokenization',
                icon='fa-puzzle-piece',
                color='#667eea',
                summary_id='stage-1-summary',
                content_id='stage-1-content'
            ),
            
            # Stage 2: Embedding
            create_stage_container(
                stage_id='embedding',
                stage_num=2,
                title='Embedding',
                icon='fa-cube',
                color='#764ba2',
                summary_id='stage-2-summary',
                content_id='stage-2-content'
            ),
            
            # Stage 3: Attention
            create_stage_container(
                stage_id='attention',
                stage_num=3,
                title='Attention',
                icon='fa-eye',
                color='#f093fb',
                summary_id='stage-3-summary',
                content_id='stage-3-content'
            ),
            
            # Stage 4: MLP/Feed-Forward
            create_stage_container(
                stage_id='mlp',
                stage_num=4,
                title='MLP (Feed-Forward)',
                icon='fa-network-wired',
                color='#4facfe',
                summary_id='stage-4-summary',
                content_id='stage-4-content'
            ),
            
            # Stage 5: Output Selection
            create_stage_container(
                stage_id='output',
                stage_num=5,
                title='Output Selection',
                icon='fa-bullseye',
                color='#00f2fe',
                summary_id='stage-5-summary',
                content_id='stage-5-content'
            ),
        ], id='pipeline-stages', className='pipeline-stages')
        
    ], id='pipeline-container', className='pipeline-container', style={'display': 'none'})


def create_flow_indicator():
    """Create the horizontal flow indicator showing all stages."""
    stages = [
        ('Input', '#6c757d'),
        ('Tokens', '#667eea'),
        ('Embed', '#764ba2'),
        ('Attention', '#f093fb'),
        ('MLP', '#4facfe'),
        ('Output', '#00f2fe'),
    ]
    
    flow_items = []
    for i, (label, color) in enumerate(stages):
        flow_items.append(
            html.Div(label, className='flow-stage-chip', style={
                'backgroundColor': color,
                'color': 'white',
                'padding': '6px 14px',
                'borderRadius': '16px',
                'fontSize': '12px',
                'fontWeight': '500',
                'whiteSpace': 'nowrap'
            })
        )
        if i < len(stages) - 1:
            flow_items.append(
                html.Span('→', style={'color': '#adb5bd', 'fontSize': '18px', 'margin': '0 4px'})
            )
    
    return html.Div(flow_items, className='flow-indicator', style={
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'center',
        'flexWrap': 'wrap',
        'gap': '4px',
        'padding': '1rem',
        'backgroundColor': '#f8f9fa',
        'borderRadius': '8px',
        'marginBottom': '1.5rem',
        'border': '1px solid #dee2e6'
    })


def create_stage_container(stage_id, stage_num, title, icon, color, summary_id, content_id):
    """
    Create an expandable stage container.
    
    Args:
        stage_id: Unique identifier for the stage
        stage_num: Stage number (1-5)
        title: Display title
        icon: FontAwesome icon class (without 'fas' prefix)
        color: Theme color for the stage
        summary_id: ID for the summary element (populated by callback)
        content_id: ID for the content element (populated by callback)
    """
    return html.Details([
        html.Summary([
            # Stage number badge
            html.Span(str(stage_num), className='stage-number', style={
                'display': 'inline-flex',
                'alignItems': 'center',
                'justifyContent': 'center',
                'width': '28px',
                'height': '28px',
                'backgroundColor': color,
                'color': 'white',
                'borderRadius': '50%',
                'fontSize': '14px',
                'fontWeight': '600',
                'marginRight': '12px'
            }),
            
            # Icon
            html.I(className=f'fas {icon}', style={
                'color': color,
                'marginRight': '10px',
                'fontSize': '16px'
            }),
            
            # Title
            html.Span(title, style={
                'fontWeight': '500',
                'color': '#2d3748',
                'fontSize': '15px'
            }),
            
            # Summary text (populated by callback)
            html.Span(id=summary_id, style={
                'marginLeft': 'auto',
                'color': '#6c757d',
                'fontSize': '13px',
                'fontStyle': 'italic'
            }),
            
            # Expand indicator
            html.Span('▸', className='expand-indicator', style={
                'marginLeft': '12px',
                'color': '#adb5bd',
                'transition': 'transform 0.2s'
            })
            
        ], className='stage-summary', style={
            'display': 'flex',
            'alignItems': 'center',
            'padding': '14px 18px',
            'backgroundColor': '#ffffff',
            'cursor': 'pointer',
            'userSelect': 'none',
            'borderRadius': '8px',
            'border': f'2px solid {color}20',
            'transition': 'all 0.2s'
        }),
        
        # Content area (populated by callback)
        html.Div(id=content_id, className='stage-content', style={
            'padding': '1.5rem',
            'backgroundColor': '#fafbfc',
            'borderRadius': '0 0 8px 8px',
            'borderTop': 'none',
            'marginTop': '-8px'
        })
        
    ], className=f'stage-container stage-{stage_id}', open=False, style={
        'marginBottom': '12px'
    })


# ============================================================================
# Stage Content Generators (called by callbacks in app.py)
# ============================================================================

def create_tokenization_content(tokens, token_ids, model_name=None):
    """
    Create content for the tokenization stage.
    
    Args:
        tokens: List of token strings
        token_ids: List of token IDs
        model_name: Optional model name for context
    """
    if not tokens:
        return html.P("Run analysis to see tokenization details.", 
                     style={'color': '#6c757d', 'fontStyle': 'italic'})
    
    # Create token chips showing the mapping
    token_chips = []
    for i, (tok, tid) in enumerate(zip(tokens, token_ids)):
        token_chips.append(
            html.Div([
                html.Span(tok, style={
                    'padding': '4px 10px',
                    'backgroundColor': '#667eea20',
                    'borderRadius': '4px',
                    'fontFamily': 'monospace',
                    'fontSize': '13px',
                    'border': '1px solid #667eea40'
                }),
                html.Span('→', style={'color': '#adb5bd', 'margin': '0 6px'}),
                html.Span(str(tid), style={
                    'padding': '4px 8px',
                    'backgroundColor': '#764ba220',
                    'borderRadius': '4px',
                    'fontFamily': 'monospace',
                    'fontSize': '12px'
                })
            ], style={'display': 'inline-flex', 'alignItems': 'center', 'marginRight': '16px', 'marginBottom': '8px'})
        )
    
    return html.Div([
        html.Div([
            html.H5("What happens here:", style={'color': '#495057', 'marginBottom': '8px'}),
            html.P([
                "Your text is split into ", 
                html.Strong(f"{len(tokens)} tokens"),
                " - small pieces that the model can understand. Each token is assigned a unique ID from the model's vocabulary."
            ], style={'color': '#6c757d', 'fontSize': '14px', 'marginBottom': '16px'})
        ]),
        
        html.Div([
            html.H5("Your tokens:", style={'color': '#495057', 'marginBottom': '12px'}),
            html.Div(token_chips, style={'display': 'flex', 'flexWrap': 'wrap'})
        ], style={'padding': '16px', 'backgroundColor': 'white', 'borderRadius': '8px', 'border': '1px solid #e2e8f0'})
    ])


def create_embedding_content(hidden_dim=None, num_tokens=None):
    """
    Create content for the embedding stage.
    
    Args:
        hidden_dim: Embedding dimension (e.g., 768)
        num_tokens: Number of tokens being processed
    """
    dim_text = f"{hidden_dim}-dimensional" if hidden_dim else "high-dimensional"
    
    return html.Div([
        html.Div([
            html.H5("What happens here:", style={'color': '#495057', 'marginBottom': '8px'}),
            html.P([
                "Each token ID is used to look up a ", html.Strong(dim_text), " vector from a ",
                html.Strong("pre-learned embedding table"), ". Think of it like a dictionary: the model has already ",
                "memorized a numeric representation for every word in its vocabulary during training."
            ], style={'color': '#6c757d', 'fontSize': '14px', 'marginBottom': '12px'}),
            html.P([
                "These embeddings capture semantic meaning - words with similar meanings (like 'happy' and 'joyful') ",
                "have similar vectors, allowing the model to understand relationships between words."
            ], style={'color': '#6c757d', 'fontSize': '14px', 'marginBottom': '16px'})
        ]),
        
        # Visual representation
        html.Div([
            html.Div([
                html.Span("Token ID", style={
                    'padding': '8px 16px',
                    'backgroundColor': '#764ba2',
                    'color': 'white',
                    'borderRadius': '6px',
                    'fontWeight': '500'
                }),
                html.Span('→', style={'margin': '0 16px', 'fontSize': '24px', 'color': '#adb5bd'}),
                html.Div([
                    html.I(className='fas fa-table', style={'color': '#764ba2', 'marginRight': '8px', 'fontSize': '18px'}),
                    html.Span("Lookup Table", style={'fontWeight': '500', 'color': '#495057', 'marginRight': '12px'})
                ], style={'display': 'inline-flex', 'alignItems': 'center'}),
                html.Span('→', style={'margin': '0 16px', 'fontSize': '24px', 'color': '#adb5bd'}),
                html.Div([
                    html.Span('[', style={'fontSize': '20px', 'color': '#495057'}),
                    html.Span(f' {dim_text} vector ', style={
                        'padding': '4px 12px',
                        'backgroundColor': '#e5d4ff',
                        'borderRadius': '4px',
                        'fontFamily': 'monospace',
                        'fontSize': '12px'
                    }),
                    html.Span(']', style={'fontSize': '20px', 'color': '#495057'})
                ], style={'display': 'inline-flex', 'alignItems': 'center'})
            ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'flexWrap': 'wrap', 'gap': '8px'})
        ], style={'padding': '24px', 'backgroundColor': 'white', 'borderRadius': '8px', 'border': '1px solid #e2e8f0'}),
        
        # Info box explaining the lookup table concept
        html.Div([
            html.I(className='fas fa-lightbulb', style={'color': '#ffc107', 'marginRight': '8px'}),
            html.Span([
                html.Strong("How the lookup table was created: "),
                "During training on billions of text examples, the model learned which numbers best represent each token. ",
                "This table is frozen after training - every time you use the model, the same token always maps to the same vector."
            ], style={'color': '#6c757d', 'fontSize': '13px'})
        ], style={'marginTop': '16px', 'padding': '12px', 'backgroundColor': '#fff8e1', 'borderRadius': '6px'})
    ])


def create_attention_content(attention_html=None, top_attended=None, layer_info=None, head_categories=None):
    """
    Create content for the attention stage.
    
    Agent G: Removed "Most attended tokens" section (deprecated). Now shows head categorization
    to help users understand what different attention heads are doing.
    
    Args:
        attention_html: BertViz HTML string for attention visualization
        top_attended: DEPRECATED - no longer used, kept for backward compatibility
        layer_info: Optional layer information for context
        head_categories: Optional dict mapping category names to counts (from get_head_category_counts)
    """
    content_items = [
        html.Div([
            html.H5("What happens here:", style={'color': '#495057', 'marginBottom': '8px'}),
            html.P([
                "The model looks at ", html.Strong("all tokens at once"), 
                " and figures out which ones are related to each other. This is called 'attention' - ",
                "each token 'attends to' other tokens to gather context for its prediction."
            ], style={'color': '#6c757d', 'fontSize': '14px', 'marginBottom': '12px'}),
            html.P([
                "Attention has multiple ", html.Strong("heads"), " - each head learns to look for different types of relationships. ",
                "For example, one head might track subject-verb agreement, while another tracks pronouns and their referents."
            ], style={'color': '#6c757d', 'fontSize': '14px', 'marginBottom': '16px'})
        ])
    ]
    
    # Agent G: Head Categorization Summary (replaces deprecated "Most attended tokens")
    if head_categories:
        category_labels = {
            'previous_token': ('Previous-Token', '#667eea', 'Heads that attend to the immediately preceding token'),
            'first_token': ('First/Positional', '#764ba2', 'Heads that focus on the first token or positional patterns'),
            'bow': ('Bag-of-Words', '#f093fb', 'Heads with diffuse attention across many tokens'),
            'syntactic': ('Syntactic', '#4facfe', 'Heads that capture grammatical relationships'),
            'other': ('Other', '#6c757d', 'Heads with mixed or specialized patterns')
        }
        
        category_chips = []
        for cat_key, count in head_categories.items():
            if count > 0 and cat_key in category_labels:
                label, color, tooltip = category_labels[cat_key]
                category_chips.append(
                    html.Span([
                        html.Span(label, style={'fontWeight': '500'}),
                        html.Span(f" ({count})", style={'marginLeft': '4px'})
                    ], style={
                        'padding': '6px 12px',
                        'backgroundColor': f'{color}20',
                        'border': f'1px solid {color}40',
                        'borderRadius': '16px',
                        'marginRight': '8px',
                        'display': 'inline-block',
                        'marginBottom': '6px',
                        'cursor': 'help'
                    }, title=tooltip)
                )
        
        if category_chips:
            content_items.append(
                html.Div([
                    html.H5("Attention Head Categories:", style={'color': '#495057', 'marginBottom': '12px'}),
                    html.Div(category_chips),
                    html.P([
                        html.I(className='fas fa-info-circle', style={'color': '#6c757d', 'marginRight': '6px'}),
                        "Heads are automatically categorized based on their attention patterns. Hover for descriptions."
                    ], style={'color': '#6c757d', 'fontSize': '12px', 'marginTop': '8px'})
                ], style={'marginBottom': '16px'})
            )
    
    # BertViz visualization with navigation instructions
    if attention_html:
        # Agent G: Enhanced navigation instructions for head view
        content_items.append(
            html.Div([
                html.H5("How to Navigate the Attention Visualization:", style={'color': '#495057', 'marginBottom': '12px'}),
                html.Div([
                    html.Div([
                        html.I(className='fas fa-mouse-pointer', style={'color': '#f093fb', 'marginRight': '8px'}),
                        html.Strong("Select heads: "),
                        html.Span("Click on layer/head numbers at the top to view specific attention heads. ",
                                 style={'color': '#6c757d'})
                    ], style={'marginBottom': '8px'}),
                    html.Div([
                        html.I(className='fas fa-arrows-alt-h', style={'color': '#f093fb', 'marginRight': '8px'}),
                        html.Strong("Lines show attention: "),
                        html.Span("Each line connects a token (left) to tokens it attends to (right). ",
                                 style={'color': '#6c757d'})
                    ], style={'marginBottom': '8px'}),
                    html.Div([
                        html.I(className='fas fa-paint-brush', style={'color': '#f093fb', 'marginRight': '8px'}),
                        html.Strong("Line thickness = attention strength: "),
                        html.Span("Thicker, darker lines mean stronger attention.",
                                 style={'color': '#6c757d'})
                    ], style={'marginBottom': '8px'}),
                    html.Div([
                        html.I(className='fas fa-search-plus', style={'color': '#f093fb', 'marginRight': '8px'}),
                        html.Strong("Hover for details: "),
                        html.Span("Hover over tokens or lines to see exact attention weights.",
                                 style={'color': '#6c757d'})
                    ])
                ], style={'padding': '12px', 'backgroundColor': '#fdf4ff', 'borderRadius': '6px', 'marginBottom': '16px'})
            ])
        )
        
        content_items.append(
            html.Div([
                html.H5("Attention Visualization:", style={'color': '#495057', 'marginBottom': '12px'}),
                html.Iframe(
                    srcDoc=attention_html,
                    style={'width': '100%', 'height': '400px', 'border': '1px solid #e2e8f0', 'borderRadius': '8px'}
                )
            ])
        )
    else:
        content_items.append(
            html.Div([
                html.P("Attention visualization will appear after analysis.", 
                      style={'color': '#6c757d', 'fontStyle': 'italic', 'padding': '24px', 'textAlign': 'center',
                             'backgroundColor': 'white', 'borderRadius': '8px', 'border': '1px dashed #dee2e6'})
            ])
        )
    
    return html.Div(content_items)


def create_mlp_content(layer_count=None, hidden_dim=None, intermediate_dim=None):
    """
    Create content for the MLP/Feed-Forward stage.
    
    Args:
        layer_count: Number of transformer layers
        hidden_dim: Model hidden dimension
        intermediate_dim: MLP intermediate dimension (usually 4x hidden_dim)
    """
    return html.Div([
        html.Div([
            html.H5("What happens here:", style={'color': '#495057', 'marginBottom': '8px'}),
            html.P([
                "After attention gathers context, each token's representation passes through a ", 
                html.Strong("Feed-Forward Network (MLP)"),
                ". This is where the model's ", html.Strong("factual knowledge"), " is stored."
            ], style={'color': '#6c757d', 'fontSize': '14px', 'marginBottom': '12px'}),
            html.P([
                "During training, the MLP weights learned to encode facts and patterns from the training data. ",
                "For example, when processing 'The capital of France is', the MLP layers help recall that 'Paris' is the answer. ",
                "Researchers have found that specific facts are often stored in specific MLP neurons."
            ], style={'color': '#6c757d', 'fontSize': '14px', 'marginBottom': '16px'})
        ]),
        
        # Visual representation of MLP
        html.Div([
            html.Div([
                # Input
                html.Div([
                    html.Div("Input", style={'fontSize': '11px', 'color': '#6c757d', 'marginBottom': '4px'}),
                    html.Div(f"{hidden_dim or '768'}d" if hidden_dim else "hidden", style={
                        'padding': '12px 20px',
                        'backgroundColor': '#4facfe',
                        'color': 'white',
                        'borderRadius': '6px',
                        'fontWeight': '500',
                        'fontSize': '14px'
                    })
                ], style={'textAlign': 'center'}),
                
                html.Span('→', style={'margin': '0 12px', 'fontSize': '20px', 'color': '#adb5bd'}),
                
                # Expand (Up-projection)
                html.Div([
                    html.Div("Expand", style={'fontSize': '11px', 'color': '#6c757d', 'marginBottom': '4px'}),
                    html.Div(f"{intermediate_dim or '3072'}d" if intermediate_dim else "4x larger", style={
                        'padding': '16px 24px',
                        'backgroundColor': '#00c9ff',
                        'color': 'white',
                        'borderRadius': '6px',
                        'fontWeight': '500',
                        'fontSize': '14px'
                    })
                ], style={'textAlign': 'center'}),
                
                html.Span('→', style={'margin': '0 12px', 'fontSize': '20px', 'color': '#adb5bd'}),
                
                # Compress (Down-projection)
                html.Div([
                    html.Div("Compress", style={'fontSize': '11px', 'color': '#6c757d', 'marginBottom': '4px'}),
                    html.Div(f"{hidden_dim or '768'}d" if hidden_dim else "hidden", style={
                        'padding': '12px 20px',
                        'backgroundColor': '#4facfe',
                        'color': 'white',
                        'borderRadius': '6px',
                        'fontWeight': '500',
                        'fontSize': '14px'
                    })
                ], style={'textAlign': 'center'})
                
            ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
        ], style={'padding': '24px', 'backgroundColor': 'white', 'borderRadius': '8px', 'border': '1px solid #e2e8f0'}),
        
        # Knowledge storage explanation
        html.Div([
            html.I(className='fas fa-brain', style={'color': '#9c27b0', 'marginRight': '8px'}),
            html.Span([
                html.Strong("Why expand then compress? "),
                "The expansion to a larger dimension creates space for the model to represent complex patterns. ",
                "Each neuron in the expanded layer can activate for specific concepts or facts. ",
                "The compression then combines these activations into a refined representation."
            ], style={'color': '#6c757d', 'fontSize': '13px'})
        ], style={'marginTop': '16px', 'padding': '12px', 'backgroundColor': '#f3e5f5', 'borderRadius': '6px'}),
        
        # Layer info
        html.Div([
            html.I(className='fas fa-layer-group', style={'color': '#4facfe', 'marginRight': '8px'}),
            html.Span([
                f"This happens in each of the model's ",
                html.Strong(f"{layer_count} layers" if layer_count else "transformer layers"),
                ", with attention and MLP working together - attention gathers context, MLP retrieves knowledge."
            ], style={'color': '#6c757d', 'fontSize': '13px'})
        ], style={'marginTop': '12px', 'padding': '12px', 'backgroundColor': '#e3f2fd', 'borderRadius': '6px'})
    ])


def create_output_content(top_tokens=None, predicted_token=None, predicted_prob=None, 
                          top5_chart=None, original_prompt=None):
    """
    Create content for the output selection stage.
    
    Args:
        top_tokens: List of (token, probability) tuples for top predictions
        predicted_token: The final predicted token
        predicted_prob: Probability of the predicted token
        top5_chart: Optional Plotly figure for top-5 visualization
        original_prompt: Original input prompt to show context with prediction
    """
    content_items = [
        html.Div([
            html.H5("What happens here:", style={'color': '#495057', 'marginBottom': '8px'}),
            html.P([
                "The model converts the final hidden state into a ", html.Strong("probability distribution"),
                " over all possible next tokens. The token with the highest probability is typically chosen as the output."
            ], style={'color': '#6c757d', 'fontSize': '14px', 'marginBottom': '16px'})
        ])
    ]
    
    # Predicted token display with full prompt context
    if predicted_token:
        # Build the full prompt + predicted token display
        prompt_display = original_prompt if original_prompt else ""
        
        content_items.append(
            html.Div([
                html.Div([
                    html.Span("Model prediction:", style={'color': '#495057', 'marginBottom': '12px', 'display': 'block', 'fontWeight': '500'}),
                    html.Div([
                        # Original prompt (dimmed)
                        html.Span(prompt_display, style={
                            'color': '#6c757d',
                            'fontFamily': 'monospace',
                            'fontSize': '15px'
                        }),
                        # Predicted token (highlighted)
                        html.Span(predicted_token, style={
                            'padding': '4px 8px',
                            'backgroundColor': '#00f2fe',
                            'color': '#1a1a2e',
                            'borderRadius': '4px',
                            'fontFamily': 'monospace',
                            'fontWeight': '600',
                            'fontSize': '15px',
                            'marginLeft': '2px'
                        })
                    ], style={'display': 'inline'}),
                    # Confidence indicator
                    html.Div([
                        html.Span(f"{predicted_prob:.1%} confidence" if predicted_prob else "", style={
                            'color': '#6c757d',
                            'fontSize': '13px',
                            'marginTop': '8px',
                            'display': 'block'
                        })
                    ])
                ], style={'textAlign': 'center'})
            ], style={'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '8px', 
                      'border': '2px solid #00f2fe', 'marginBottom': '16px'})
        )
    
    # Top-5 bar chart with improved hover formatting
    if top_tokens:
        tokens = [t[0] for t in top_tokens[:5]]
        probs = [t[1] for t in top_tokens[:5]]
        
        fig = go.Figure(go.Bar(
            x=probs,
            y=tokens,
            orientation='h',
            marker_color=['#00f2fe' if i == 0 else '#4facfe' for i in range(len(tokens))],
            text=[f"{p:.1%}" for p in probs],
            textposition='outside',
            # Format hover to show "Token (X%)" instead of long decimals
            hovertemplate='%{y} (%{x:.1%})<extra></extra>'
        ))
        
        fig.update_layout(
            title="Top 5 Predictions",
            xaxis_title="Probability",
            yaxis_title="Token",
            height=250,
            margin=dict(l=20, r=60, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(autorange='reversed')
        )
        
        content_items.append(
            html.Div([
                dcc.Graph(figure=fig, config={'displayModeBar': False})
            ], style={'backgroundColor': 'white', 'borderRadius': '8px', 'border': '1px solid #e2e8f0'})
        )
    elif top5_chart:
        content_items.append(
            html.Div([
                dcc.Graph(figure=top5_chart, config={'displayModeBar': False})
            ], style={'backgroundColor': 'white', 'borderRadius': '8px', 'border': '1px solid #e2e8f0'})
        )
    
    return html.Div(content_items)


def create_stage_summary(stage_id, activation_data=None, model_config=None):
    """
    Generate summary text for a stage (shown when collapsed).
    
    Args:
        stage_id: Stage identifier ('tokenization', 'embedding', etc.)
        activation_data: Optional activation data from forward pass
        model_config: Optional model configuration
    """
    if not activation_data:
        return "Awaiting input..."
    
    summaries = {
        'tokenization': lambda: f"{len(activation_data.get('input_ids', [[]])[0])} tokens",
        'embedding': lambda: f"{model_config.hidden_size if model_config else 768}-dim vectors" if model_config else "Vectors ready",
        'attention': lambda: f"{model_config.num_attention_heads if model_config else 12} heads" if model_config else "Context gathered",
        'mlp': lambda: f"{model_config.num_hidden_layers if model_config else 12} layers" if model_config else "Transformations applied",
        'output': lambda: f"→ {activation_data.get('actual_output', {}).get('token', '?')}" if activation_data.get('actual_output') else "Output computed"
    }
    
    return summaries.get(stage_id, lambda: "")()

