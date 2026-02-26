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
    
    Displays tokens in vertical rows: each row shows [token] → [ID]
    
    Args:
        tokens: List of token strings
        token_ids: List of token IDs
        model_name: Optional model name for context
    """
    if not tokens:
        return html.P("Run analysis to see tokenization details.", 
                     style={'color': '#6c757d', 'fontStyle': 'italic'})
    
    # Create vertical token rows - each token is its own row
    token_rows = []
    for i, (tok, tid) in enumerate(zip(tokens, token_ids)):
        token_rows.append(
            html.Div([
                # Token box
                html.Span(tok, style={
                    'display': 'inline-block',
                    'minWidth': '80px',
                    'padding': '6px 12px',
                    'backgroundColor': '#667eea20',
                    'borderRadius': '6px',
                    'fontFamily': 'monospace',
                    'fontSize': '13px',
                    'border': '1px solid #667eea40',
                    'textAlign': 'center'
                }),
                # Arrow
                html.Span('→', style={
                    'color': '#adb5bd',
                    'margin': '0 12px',
                    'fontSize': '16px'
                }),
                # ID box
                html.Span(str(tid), style={
                    'display': 'inline-block',
                    'minWidth': '60px',
                    'padding': '6px 12px',
                    'backgroundColor': '#764ba220',
                    'borderRadius': '6px',
                    'fontFamily': 'monospace',
                    'fontSize': '13px',
                    'border': '1px solid #764ba240',
                    'textAlign': 'center'
                })
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'marginBottom': '8px',
                'padding': '4px 0'
            })
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
            # Header row
            html.Div([
                html.Span("Token", style={
                    'display': 'inline-block',
                    'minWidth': '80px',
                    'fontWeight': '600',
                    'color': '#495057',
                    'fontSize': '12px',
                    'textAlign': 'center'
                }),
                html.Span('', style={'margin': '0 12px', 'width': '16px'}),
                html.Span("ID", style={
                    'display': 'inline-block',
                    'minWidth': '60px',
                    'fontWeight': '600',
                    'color': '#495057',
                    'fontSize': '12px',
                    'textAlign': 'center'
                })
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'marginBottom': '12px',
                'paddingBottom': '8px',
                'borderBottom': '1px solid #e9ecef'
            }),
            # Token rows (vertical layout)
            html.Div(token_rows, style={
                'display': 'flex',
                'flexDirection': 'column'
            })
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
    
    Displays head categorization with active/inactive states, activation bars,
    suggested prompts, and guided interpretation.
    
    Args:
        attention_html: BertViz HTML string for attention visualization
        top_attended: DEPRECATED - no longer used
        layer_info: Optional layer information for context
        head_categories: Output from get_active_head_summary() — dict with 'categories' key
                        containing per-category data with activation scores.
                        Falls back gracefully if None or old format.
    """
    content_items = [
        html.Div([
            html.H5("What happens here:", style={'color': '#495057', 'marginBottom': '8px'}),
            html.P([
                "The model looks at ", html.Strong("all tokens at once"), 
                " and figures out which ones are related to each other. This is called 'attention' — ",
                "each token 'attends to' other tokens to gather context for its prediction."
            ], style={'color': '#6c757d', 'fontSize': '14px', 'marginBottom': '12px'}),
            html.P([
                "Attention has multiple ", html.Strong("heads"), " — each head learns to look for different types of relationships. ",
                "Below you can see what role each head plays and whether it's active on your current input."
            ], style={'color': '#6c757d', 'fontSize': '14px', 'marginBottom': '16px'})
        ])
    ]
    
    # New: Head Roles Panel using get_active_head_summary() output
    if head_categories and isinstance(head_categories, dict) and 'categories' in head_categories:
        categories = head_categories['categories']
        
        # Color scheme per category
        category_colors = {
            'previous_token': '#667eea',
            'induction': '#e67e22',
            'duplicate_token': '#9b59b6',
            'positional': '#2ecc71',
            'diffuse': '#3498db',
            'other': '#95a5a6'
        }
        
        # Find the top recommended head for guided interpretation
        guided_head = None
        guided_cat = None
        for cat_key in ['previous_token', 'induction', 'positional']:
            cat_data = categories.get(cat_key, {})
            heads = cat_data.get('heads', [])
            active_heads = [h for h in heads if h.get('is_active')]
            if active_heads:
                best = max(active_heads, key=lambda h: h['activation_score'])
                if guided_head is None or best['activation_score'] > guided_head['activation_score']:
                    guided_head = best
                    guided_cat = cat_data.get('display_name', cat_key)
        
        # Guided interpretation recommendation
        if guided_head:
            content_items.append(
                html.Div([
                    html.I(className='fas fa-lightbulb', style={'color': '#f39c12', 'marginRight': '8px', 'fontSize': '16px'}),
                    html.Span([
                        html.Strong("Try this: "),
                        f"Select Layer {guided_head['layer']}, Head {guided_head['head']} in the visualization below — ",
                        f"this is a {guided_cat} head ",
                        f"(activation: {guided_head['activation_score']:.0%} on your input)."
                    ], style={'color': '#495057', 'fontSize': '13px'})
                ], style={
                    'padding': '12px 16px', 'backgroundColor': '#fef9e7', 'borderRadius': '8px',
                    'border': '1px solid #f9e79f', 'marginBottom': '16px', 'display': 'flex', 'alignItems': 'center'
                })
            )
        
        # Build category sections
        category_sections = []
        category_order = ['previous_token', 'induction', 'duplicate_token', 'positional', 'diffuse', 'other']
        
        for cat_key in category_order:
            cat_data = categories.get(cat_key, {})
            if not cat_data:
                continue
            
            display_name = cat_data.get('display_name', cat_key)
            description = cat_data.get('description', '')
            educational_text = cat_data.get('educational_text', '')
            icon_name = cat_data.get('icon', 'circle')
            is_applicable = cat_data.get('is_applicable', True)
            suggested_prompt = cat_data.get('suggested_prompt')
            heads = cat_data.get('heads', [])
            color = category_colors.get(cat_key, '#95a5a6')
            
            # Active vs inactive indicator
            has_active_heads = any(h.get('is_active') for h in heads)
            status_icon = '●' if (is_applicable and has_active_heads) else '○'
            status_color = color if (is_applicable and has_active_heads) else '#ccc'
            
            # Skip "other" if no heads (which is the normal case)
            if cat_key == 'other' and not heads:
                continue
            
            # Build head items with activation bars
            head_items = []
            if heads:
                for head_info in heads:
                    activation = head_info.get('activation_score', 0.0)
                    is_active = head_info.get('is_active', False)
                    label = head_info.get('label', f"L{head_info['layer']}-H{head_info['head']}")
                    
                    # Activation bar
                    bar_width = max(activation * 100, 2)  # Min 2% for visibility
                    bar_color = color if is_active else '#ddd'
                    
                    head_items.append(
                        html.Div([
                            # Head label
                            html.Span(label, style={
                                'fontFamily': 'monospace', 'fontSize': '12px', 'fontWeight': '500',
                                'minWidth': '60px', 'color': '#495057' if is_active else '#aaa',
                            }, title=f"See Layer {head_info['layer']}, Head {head_info['head']} in the visualization below"),
                            # Activation bar
                            html.Div([
                                html.Div(style={
                                    'width': f'{bar_width}%', 'height': '100%',
                                    'backgroundColor': bar_color, 'borderRadius': '3px',
                                    'transition': 'width 0.3s ease'
                                })
                            ], style={
                                'flex': '1', 'height': '12px', 'backgroundColor': '#f0f0f0',
                                'borderRadius': '3px', 'margin': '0 8px', 'overflow': 'hidden'
                            }),
                            # Score label
                            html.Span(f"{activation:.2f}", style={
                                'fontSize': '11px', 'fontFamily': 'monospace',
                                'color': '#495057' if is_active else '#bbb', 'minWidth': '32px'
                            })
                        ], style={
                            'display': 'flex', 'alignItems': 'center', 'marginBottom': '4px',
                            'opacity': '1' if is_active else '0.5'
                        })
                    )
            
            # Build the category section
            # Header content
            summary_children = [
                html.Span(status_icon, style={
                    'color': status_color, 'fontSize': '16px', 'marginRight': '8px'
                }),
                html.Span(display_name, style={'fontWeight': '500', 'color': '#495057'}),
            ]
            
            if heads:
                active_count = sum(1 for h in heads if h.get('is_active'))
                summary_children.append(
                    html.Span(f" ({active_count}/{len(heads)} active)", style={
                        'marginLeft': '6px', 'color': '#6c757d', 'fontSize': '12px'
                    })
                )
            
            if not is_applicable:
                summary_children.append(
                    html.Span(" — not triggered on this input", style={
                        'marginLeft': '6px', 'color': '#aaa', 'fontSize': '12px', 'fontStyle': 'italic'
                    })
                )
            
            # Expanded content
            expanded_children = []
            
            # Educational explanation
            if educational_text:
                expanded_children.append(
                    html.P(educational_text, style={
                        'color': '#6c757d', 'fontSize': '13px', 'marginBottom': '10px',
                        'fontStyle': 'italic', 'lineHeight': '1.5'
                    })
                )
            
            # Suggested prompt (for grayed-out categories)
            if not is_applicable and suggested_prompt:
                expanded_children.append(
                    html.Div([
                        html.I(className='fas fa-flask', style={'color': '#e67e22', 'marginRight': '6px'}),
                        html.Span(suggested_prompt, style={'color': '#e67e22', 'fontSize': '12px'})
                    ], style={
                        'padding': '8px 12px', 'backgroundColor': '#fef5e7',
                        'borderRadius': '6px', 'marginBottom': '10px', 'border': '1px solid #fde8c8'
                    })
                )
            
            # Head activation bars
            if head_items:
                expanded_children.append(html.Div(head_items))
            
            category_sections.append(
                html.Details([
                    html.Summary(summary_children, style={
                        'padding': '10px 14px',
                        'backgroundColor': f'{color}08' if is_applicable else '#fafafa',
                        'border': f'1px solid {color}25' if is_applicable else '1px solid #eee',
                        'borderRadius': '8px', 'cursor': 'pointer', 'userSelect': 'none',
                        'listStyle': 'none', 'display': 'flex', 'alignItems': 'center'
                    }),
                    html.Div(expanded_children, style={
                        'padding': '12px 14px', 'backgroundColor': '#fafbfc',
                        'borderRadius': '0 0 8px 8px', 'marginTop': '-1px',
                        'border': f'1px solid {color}25' if is_applicable else '1px solid #eee',
                        'borderTop': 'none'
                    })
                ], style={'marginBottom': '8px'}, open=(cat_key == 'previous_token'))  # Default-open first category
            )
        
        if category_sections:
            # Legend
            legend = html.Div([
                html.Span("● = active on your input", style={
                    'color': '#495057', 'fontSize': '11px', 'marginRight': '16px'
                }),
                html.Span("○ = role exists but not triggered", style={
                    'color': '#aaa', 'fontSize': '11px'
                })
            ], style={'marginBottom': '10px'})
            
            content_items.append(
                html.Div([
                    html.H5("Attention Head Roles:", style={'color': '#495057', 'marginBottom': '8px'}),
                    html.P([
                        "Each category represents a type of behavior we detected in this model's attention heads. ",
                        "Click a category to see individual heads and how strongly they're activated on your input."
                    ], style={'color': '#6c757d', 'fontSize': '12px', 'marginBottom': '12px'}),
                    legend,
                    html.Div(category_sections),
                    # Accuracy caveat
                    html.Div([
                        html.I(className='fas fa-info-circle', style={'color': '#6c757d', 'marginRight': '6px', 'fontSize': '11px'}),
                        html.Span(
                            "These categories are simplified labels based on each head's dominant behavior. "
                            "In reality, heads can serve multiple roles and may behave differently on different inputs.",
                            style={'color': '#999', 'fontSize': '11px'}
                        )
                    ], style={'marginTop': '12px', 'padding': '8px 12px', 'backgroundColor': '#f8f9fa', 'borderRadius': '6px'})
                ], style={'marginBottom': '16px'})
            )
    elif head_categories is None:
        # Model not analyzed — show fallback message
        content_items.append(
            html.Div([
                html.I(className='fas fa-info-circle', style={'color': '#6c757d', 'marginRight': '8px'}),
                html.Span(
                    "Head categorization is not available for this model. "
                    "The attention visualization below still shows the full attention patterns.",
                    style={'color': '#6c757d', 'fontSize': '13px'}
                )
            ], style={
                'padding': '12px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px',
                'border': '1px solid #dee2e6', 'marginBottom': '16px'
            })
        )
    
    # BertViz visualization with navigation instructions
    if attention_html:
        content_items.append(
            html.Div([
                html.H5("How to Navigate the Attention Visualization:", style={'color': '#495057', 'marginBottom': '12px'}),
                html.Div([
                    html.Div([
                        html.I(className='fas fa-mouse-pointer', style={'color': '#f093fb', 'marginRight': '8px'}),
                        html.Strong("Select heads: "),
                        html.Span("Click on layer/head numbers at the top to view specific attention heads.",
                                 style={'color': '#6c757d'})
                    ], style={'marginBottom': '4px'}),
                    html.Div([
                        html.Span("• ", style={'color': '#f093fb', 'fontWeight': 'bold'}),
                        html.Strong("Single click ", style={'color': '#495057'}),
                        html.Span("on a colored head square: selects or deselects that head",
                                 style={'color': '#6c757d'})
                    ], style={'marginLeft': '28px', 'marginBottom': '4px', 'fontSize': '13px'}),
                    html.Div([
                        html.Span("• ", style={'color': '#f093fb', 'fontWeight': 'bold'}),
                        html.Strong("Double click ", style={'color': '#495057'}),
                        html.Span("on a colored head square: selects only that head (deselects all others)",
                                 style={'color': '#6c757d'})
                    ], style={'marginLeft': '28px', 'marginBottom': '12px', 'fontSize': '13px'}),
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


def _build_token_display(prompt_text, generated_tokens, position, actual_prob):
    """Build the token display for a given scrubber position.
    
    Args:
        prompt_text: Original prompt string.
        generated_tokens: List of generated token strings.
        position: Current slider position (0-indexed into generated_tokens).
        actual_prob: Probability of the highlighted token at this position.
    """
    # Context = prompt + all generated tokens before the current position
    context_parts = [
        html.Span(prompt_text, style={
            'color': '#6c757d',
            'fontFamily': 'monospace',
            'fontSize': '15px'
        })
    ]
    for j in range(position):
        context_parts.append(
            html.Span(generated_tokens[j], style={
                'color': '#6c757d',
                'fontFamily': 'monospace',
                'fontSize': '15px'
            })
        )

    # Highlighted token at the current position
    highlighted = html.Span(generated_tokens[position], style={
        'padding': '4px 8px',
        'backgroundColor': '#00f2fe',
        'color': '#1a1a2e',
        'borderRadius': '4px',
        'fontFamily': 'monospace',
        'fontWeight': '600',
        'fontSize': '15px',
        'marginLeft': '2px'
    })

    confidence = html.Div([
        html.Span(
            f"{actual_prob:.1%} confidence" if actual_prob else "",
            style={'color': '#6c757d', 'fontSize': '13px', 'marginTop': '8px', 'display': 'block'}
        )
    ])

    return html.Div([
        html.Div([
            html.Span(
                f"Token {position + 1} of {len(generated_tokens)}:",
                style={'color': '#495057', 'marginBottom': '12px', 'display': 'block', 'fontWeight': '500'}
            ),
            html.Div(context_parts + [highlighted], style={'display': 'inline'}),
            confidence
        ], style={'textAlign': 'center'})
    ], style={
        'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '8px',
        'border': '2px solid #00f2fe', 'marginBottom': '16px'
    })


def _build_top5_chart(top5_data, actual_token=None):
    """Build the top-5 bar chart for a single scrubber position.
    
    Args:
        top5_data: List of {'token': str, 'probability': float}.
        actual_token: The token that was actually generated (highlighted if present).
    """
    tokens = [entry['token'] for entry in top5_data]
    probs = [entry['probability'] for entry in top5_data]

    # Highlight the actual chosen token if it appears in the top 5
    colors = []
    actual_in_top5 = False
    for t in tokens:
        if actual_token and t.strip() == actual_token.strip():
            colors.append('#00f2fe')
            actual_in_top5 = True
        else:
            colors.append('#4facfe')

    fig = go.Figure(go.Bar(
        x=probs,
        y=tokens,
        orientation='h',
        marker_color=colors,
        text=[f"{p:.1%}" for p in probs],
        textposition='outside',
        hovertemplate='%{y} (%{x:.1%})<extra></extra>'
    ))

    fig.update_layout(
        title="Top 5 Next-Token Predictions",
        xaxis_title="Probability",
        yaxis_title="Token",
        height=250,
        margin=dict(l=20, r=60, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(autorange='reversed')
    )

    children = [dcc.Graph(figure=fig, config={'displayModeBar': False})]

    # If the actual token is not in the top 5, add a note below
    if actual_token and not actual_in_top5:
        children.append(html.Div([
            html.I(className='fas fa-info-circle', style={'color': '#6c757d', 'marginRight': '6px'}),
            html.Span([
                "The actual token \"", html.Strong(actual_token.strip()),
                "\" was not in the top 5 predictions at this position."
            ], style={'color': '#6c757d', 'fontSize': '13px'})
        ], style={'padding': '8px 12px'}))

    return html.Div(children, style={
        'backgroundColor': 'white', 'borderRadius': '8px', 'border': '1px solid #e2e8f0'
    })


def create_output_content(top_tokens=None, predicted_token=None, predicted_prob=None,
                          top5_chart=None, original_prompt=None,
                          per_position_data=None, generated_tokens=None,
                          prompt_text=None):
    """
    Create content for the output selection stage.

    When per_position_data is available the output is an interactive scrubber
    that lets the user step through each generated-token position.  Otherwise
    falls back to the previous static display.
    
    Args:
        top_tokens: List of (token, probability) tuples for top predictions (static mode).
        predicted_token: The final predicted token (static mode).
        predicted_prob: Probability of the predicted token (static mode).
        top5_chart: Optional Plotly figure for top-5 visualization (static mode).
        original_prompt: Original input prompt to show context with prediction (static mode).
        per_position_data: List of per-position dicts from compute_per_position_top5 (scrubber mode).
        generated_tokens: List of generated token strings (scrubber mode).
        prompt_text: Original prompt text for context display (scrubber mode).
    """
    # --- Scrubber mode ---
    if per_position_data and generated_tokens:
        num_positions = len(per_position_data)
        prompt_display = prompt_text or original_prompt or ""

        content_items = [
            html.Div([
                html.H5("What happens here:", style={'color': '#495057', 'marginBottom': '8px'}),
                html.P([
                    "The model converts the final hidden state into a ",
                    html.Strong("probability distribution"),
                    " over all possible next tokens. Use the slider below to step through "
                    "each generated token and see the model's top predictions at that point."
                ], style={'color': '#6c757d', 'fontSize': '14px', 'marginBottom': '16px'})
            ])
        ]

        # Slider / scrubber
        slider_marks = {i: {'label': generated_tokens[i].strip() or repr(generated_tokens[i])}
                        for i in range(num_positions)}
        content_items.append(
            html.Div([
                html.Span("Step through generated tokens:",
                           style={'color': '#495057', 'fontWeight': '500', 'display': 'block',
                                  'marginBottom': '8px'}),
                dcc.Slider(
                    id='output-scrubber-slider',
                    min=0,
                    max=max(num_positions - 1, 0),
                    step=1,
                    value=0,
                    marks=slider_marks,
                    included=False,
                )
            ], style={'marginBottom': '20px', 'padding': '12px 16px',
                      'backgroundColor': '#f8f9fa', 'borderRadius': '8px',
                      'border': '1px solid #dee2e6'})
        )

        # Initial render at position 0
        pos0 = per_position_data[0]
        content_items.append(
            html.Div(
                _build_token_display(prompt_display, generated_tokens, 0, pos0['actual_prob']),
                id='output-token-display'
            )
        )
        content_items.append(
            html.Div(
                _build_top5_chart(pos0['top5'], pos0.get('actual_token')),
                id='output-top5-chart'
            )
        )

        # Disclaimer
        content_items.append(
            html.Div([
                html.I(className='fas fa-info-circle', style={'color': '#6c757d', 'marginRight': '8px'}),
                html.Span([
                    html.Strong("Note on Token Selection: "),
                    "While the probabilities above show the model's raw preference for the immediate next token, the final choice ",
                    "can be influenced by other factors. Techniques like ", html.Strong("Beam Search"),
                    " look ahead at multiple possible sequences to find the best overall result, rather than just the single most likely token at each step. ",
                    "Additionally, architectures like ", html.Strong("Mixture of Experts (MoE)"),
                    " might route processing through different specialized internal networks which can impact the final output distribution."
                ], style={'color': '#6c757d', 'fontSize': '13px'})
            ], style={'marginTop': '16px', 'padding': '12px', 'backgroundColor': '#f8f9fa',
                      'borderRadius': '6px', 'border': '1px solid #dee2e6'})
        )

        return html.Div(content_items)

    # --- Static fallback (prompt-only analysis, no generated tokens yet) ---
    content_items = [
        html.Div([
            html.H5("What happens here:", style={'color': '#495057', 'marginBottom': '8px'}),
            html.P([
                "The model converts the final hidden state into a ", html.Strong("probability distribution"),
                " over all possible next tokens. The token with the highest probability is typically chosen as the output."
            ], style={'color': '#6c757d', 'fontSize': '14px', 'marginBottom': '16px'})
        ])
    ]
    
    if predicted_token:
        prompt_display = original_prompt if original_prompt else ""
        content_items.append(
            html.Div([
                html.Div([
                    html.Span("Model prediction:", style={'color': '#495057', 'marginBottom': '12px', 'display': 'block', 'fontWeight': '500'}),
                    html.Div([
                        html.Span(prompt_display, style={
                            'color': '#6c757d', 'fontFamily': 'monospace', 'fontSize': '15px'
                        }),
                        html.Span(predicted_token, style={
                            'padding': '4px 8px', 'backgroundColor': '#00f2fe',
                            'color': '#1a1a2e', 'borderRadius': '4px',
                            'fontFamily': 'monospace', 'fontWeight': '600',
                            'fontSize': '15px', 'marginLeft': '2px'
                        })
                    ], style={'display': 'inline'}),
                    html.Div([
                        html.Span(f"{predicted_prob:.1%} confidence" if predicted_prob else "", style={
                            'color': '#6c757d', 'fontSize': '13px', 'marginTop': '8px', 'display': 'block'
                        })
                    ])
                ], style={'textAlign': 'center'})
            ], style={'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '8px',
                      'border': '2px solid #00f2fe', 'marginBottom': '16px'})
        )
    
    if top_tokens:
        tokens = [t[0] for t in top_tokens[:5]]
        probs = [t[1] for t in top_tokens[:5]]
        
        fig = go.Figure(go.Bar(
            x=probs, y=tokens, orientation='h',
            marker_color=['#00f2fe' if i == 0 else '#4facfe' for i in range(len(tokens))],
            text=[f"{p:.1%}" for p in probs], textposition='outside',
            hovertemplate='%{y} (%{x:.1%})<extra></extra>'
        ))
        fig.update_layout(
            title="Top 5 Predictions", xaxis_title="Probability", yaxis_title="Token",
            height=250, margin=dict(l=20, r=60, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
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
    
    content_items.append(
        html.Div([
            html.I(className='fas fa-info-circle', style={'color': '#6c757d', 'marginRight': '8px'}),
            html.Span([
                html.Strong("Note on Token Selection: "),
                "While the probabilities above show the model's raw preference for the immediate next token, the final choice ",
                "can be influenced by other factors. Techniques like ", html.Strong("Beam Search"), 
                " look ahead at multiple possible sequences to find the best overall result, rather than just the single most likely token at each step. ",
                "Additionally, architectures like ", html.Strong("Mixture of Experts (MoE)"),
                " might route processing through different specialized internal networks which can impact the final output distribution."
            ], style={'color': '#6c757d', 'fontSize': '13px'})
        ], style={'marginTop': '16px', 'padding': '12px', 'backgroundColor': '#f8f9fa', 'borderRadius': '6px', 'border': '1px solid #dee2e6'})
    )
    
    return html.Div(content_items)

