"""
Investigation Panel component for exploring model behavior.

Consolidates two main investigation tools:
1. Ablation - Remove attention heads/layers to see their impact
2. Token Attribution - Identify which input tokens influenced the output
"""

from dash import html, dcc
import plotly.graph_objs as go
from components.ablation_panel import create_ablation_panel


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
                children=create_ablation_panel(),
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
                value='integrated',
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
            )
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

