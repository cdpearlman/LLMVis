"""
Ablation Panel component for interactive model experimentation.

Provides a grid-based interface for selecting attention heads to ablate
and viewing the resulting changes in model output.
"""

from dash import html, dcc
import plotly.graph_objs as go
import json

def create_ablation_panel():
    """Create the main ablation tool content."""
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
        
        # Head Selector Interface
        html.Div([
            html.Label("Add Head to Ablation List:", className="input-label", style={'marginBottom': '8px', 'display': 'block'}),
            html.Div([
                # Layer Select
                html.Div([
                    dcc.Dropdown(
                        id='ablation-layer-select',
                        placeholder="Layer",
                        options=[], # Populated by callback
                        style={'fontSize': '14px'}
                    )
                ], style={'flex': '1', 'marginRight': '8px'}),
                
                # Head Select
                html.Div([
                    dcc.Dropdown(
                        id='ablation-head-select',
                        placeholder="Head",
                        options=[], # Populated by callback
                        style={'fontSize': '14px'}
                    )
                ], style={'flex': '1', 'marginRight': '8px'}),
                
                # Add Button
                html.Button([
                    html.I(className='fas fa-plus'),
                ], id='ablation-add-head-btn', className='action-button secondary-button',
                   title="Add Head", style={'padding': '8px 12px'})
                
            ], style={'display': 'flex', 'alignItems': 'center'})
        ], style={'marginBottom': '16px', 'padding': '16px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px', 'border': '1px solid #e2e8f0'}),
        
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
        
        # Reset button
        html.Button([
            html.I(className='fas fa-trash-alt', style={'marginRight': '8px'}),
            "Clear Selected Heads"
        ], id='clear-ablation-btn', className='action-button secondary-button',
           style={'width': '100%', 'marginBottom': '8px'}),

        # Run ablation button
        html.Button([
            html.I(className='fas fa-play', style={'marginRight': '8px'}),
            "Run Ablation Experiment"
        ], id='run-ablation-btn', className='action-button primary-button',
           disabled=True, title="Add at least one head above to run the experiment",
           style={'width': '100%', 'marginBottom': '16px'}),
        
        # Results container
        dcc.Loading(
            id="ablation-loading",
            type="default",
            children=[
                html.Div(id='ablation-results-container', children=[
                    # Will be populated by callback
                ])
            ]
        )
        
    ], className='ablation-tool')


def create_selected_heads_display(selected_heads):
    """
    Create display of selected heads as chips with remove buttons.
    
    Args:
        selected_heads: List of {layer: N, head: M} dicts
    """
    if not selected_heads:
        return html.Div(
            "No heads selected yet",
            style={'color': '#6c757d', 'fontSize': '13px', 'fontStyle': 'italic', 'padding': '8px 0'}
        )
    
    chips = []
    for item in selected_heads:
        if not isinstance(item, dict):
            continue
            
        layer = item.get('layer')
        head = item.get('head')
        
        if layer is None or head is None:
            continue
            
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


def create_ablation_results_display(original_data, ablated_data, selected_heads, selected_beam=None, ablated_beam=None):
    """
    Create the ablation results display focusing on full generation comparison,
    including an interactive scrubber and metrics summary.
    """
    # Format selected heads for display
    all_heads_formatted = [f"L{item['layer']}-H{item['head']}" for item in selected_heads if isinstance(item, dict)]

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
    
    # Extract data for metrics and UI
    orig_positions = original_data.get('per_position_top5', []) if original_data else []
    abl_positions = ablated_data.get('per_position_top5', []) if ablated_data else []
    orig_tokens = original_data.get('generated_tokens', []) if original_data else []
    abl_tokens = ablated_data.get('generated_tokens', []) if ablated_data else []
    
    # Fallback strings if missing
    orig_text = selected_beam['text'] if selected_beam else ""
    abl_text = ablated_beam['text'] if ablated_beam else ""
    
    max_len = max(len(orig_tokens), len(abl_tokens))
    if max_len == 0:
        results.append(html.Div("No generated tokens to compare.", style={'padding': '20px', 'color': '#6c757d'}))
        return html.Div(results)
        
    tokens_changed = sum(1 for i in range(max_len) if i >= len(orig_tokens) or i >= len(abl_tokens) or orig_tokens[i] != abl_tokens[i])
    percent_changed = (tokens_changed / max_len * 100) if max_len > 0 else 0
    
    # Calculate probability shifts and unchanged spans
    prob_shifts = []
    prob_chart_data = [] # For the mini sparkline
    longest_unchanged = 0
    current_unchanged = 0
    current_start = 0
    longest_span_start = 0
    longest_span_end = 0
    
    for i in range(len(orig_positions)):
        orig_token = orig_positions[i]['actual_token']
        orig_prob = orig_positions[i]['actual_prob']
        
        # Find the probability of orig_token in ablated top5
        abl_prob = 0.0
        if i < len(abl_positions):
            if abl_positions[i]['actual_token'] == orig_token:
                abl_prob = abl_positions[i]['actual_prob']
            else:
                for t in abl_positions[i]['top5']:
                    if t['token'] == orig_token:
                        abl_prob = t['probability']
                        break
                        
        shift = abl_prob - orig_prob
        prob_shifts.append(shift)
        prob_chart_data.append(orig_prob)
        
        if i < len(orig_tokens) and i < len(abl_tokens) and orig_tokens[i] == abl_tokens[i]:
            if current_unchanged == 0:
                current_start = i
            current_unchanged += 1
            if current_unchanged > longest_unchanged:
                longest_unchanged = current_unchanged
                longest_span_start = current_start
                longest_span_end = i
        else:
            current_unchanged = 0

    avg_prob_shift = sum(prob_shifts) / len(prob_shifts) if prob_shifts else 0.0
    
    # --- Top Panel: Interactive Scrubber ---
    
    # Shared Slider
    slider_div = html.Div([
        dcc.Slider(
            id='ablation-scrubber-slider',
            min=0,
            max=max_len - 1,
            step=1,
            value=0,
            marks={i: {'label': orig_tokens[i].strip()[:6] if i < len(orig_tokens) else str(i), 'style': {'fontSize': '10px', 'whiteSpace': 'nowrap'}} for i in range(max_len)},
            tooltip={"placement": "bottom", "always_visible": False},
            updatemode='drag'
        )
    ], style={'padding': '0 20px 20px 20px'})
    
    # Comparison Grid
    comparison_grid = html.Div([
        # Original Output Column (Green Theme)
        html.Div([
            html.Div("ORIGINAL OUTPUT", style={
                'backgroundColor': '#28a745', 'color': 'white', 'padding': '4px 16px', 
                'borderRadius': '16px', 'fontWeight': 'bold', 'fontSize': '12px',
                'display': 'inline-block', 'marginBottom': '15px'
            }),
            html.Div(id='ablation-original-token-map', style={'fontSize': '12px', 'color': '#6c757d', 'marginBottom': '10px', 'minHeight': '40px', 'textAlign': 'left', 'lineHeight': '1.5'}),
            html.Div(id='ablation-original-text-box', style={
                'backgroundColor': '#f8f9fa', 'border': '1px solid #dee2e6', 'borderRadius': '8px', 
                'padding': '15px', 'fontFamily': 'monospace', 'fontSize': '14px', 'minHeight': '80px', 'marginBottom': '15px', 'textAlign': 'left', 'whiteSpace': 'pre-wrap'
            }),
            html.Div("TOP 5 PREDICTIONS", style={'textAlign': 'center', 'fontWeight': 'bold', 'color': '#495057', 'fontSize': '12px', 'marginBottom': '10px'}),
            dcc.Graph(id='ablation-original-top5-chart', config={'displayModeBar': False}, style={'height': '200px'})
        ], style={
            'flex': '1', 'border': '2px solid #28a745', 'borderRadius': '12px', 
            'padding': '20px', 'textAlign': 'center', 'backgroundColor': 'white', 'width': '45%'
        }),
        
        # Center Divergence Indicator
        html.Div(id='ablation-divergence-indicator', style={
            'width': '60px', 'display': 'flex', 'flexDirection': 'column', 
            'alignItems': 'center', 'justifyContent': 'center'
        }),
        
        # Ablated Output Column (Red Theme)
        html.Div([
            html.Div("ABLATED OUTPUT", style={
                'backgroundColor': '#dc3545', 'color': 'white', 'padding': '4px 16px', 
                'borderRadius': '16px', 'fontWeight': 'bold', 'fontSize': '12px',
                'display': 'inline-block', 'marginBottom': '15px'
            }),
            html.Div(id='ablation-ablated-token-map', style={'fontSize': '12px', 'color': '#6c757d', 'marginBottom': '10px', 'minHeight': '40px', 'textAlign': 'left', 'lineHeight': '1.5'}),
            html.Div(id='ablation-ablated-text-box', style={
                'backgroundColor': '#f8f9fa', 'border': '1px solid #dee2e6', 'borderRadius': '8px', 
                'padding': '15px', 'fontFamily': 'monospace', 'fontSize': '14px', 'minHeight': '80px', 'marginBottom': '15px', 'textAlign': 'left', 'whiteSpace': 'pre-wrap'
            }),
            html.Div("TOP 5 PREDICTIONS", style={'textAlign': 'center', 'fontWeight': 'bold', 'color': '#495057', 'fontSize': '12px', 'marginBottom': '10px'}),
            dcc.Graph(id='ablation-ablated-top5-chart', config={'displayModeBar': False}, style={'height': '200px'})
        ], style={
            'flex': '1', 'border': '2px solid #dc3545', 'borderRadius': '12px', 
            'padding': '20px', 'textAlign': 'center', 'backgroundColor': 'white', 'width': '45%'
        })
    ], style={'display': 'flex', 'gap': '15px', 'marginBottom': '25px', 'justifyContent': 'center'})
    
    results.append(html.Div([slider_div, comparison_grid], style={
        'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '12px', 'border': '1px solid #e2e8f0'
    }))
    
    # --- Bottom Panel: Impact Summary ---
    
    sparkline_fig = go.Figure(go.Scatter(y=prob_chart_data, mode='lines+markers', line=dict(color='#667eea', width=2), marker=dict(size=6, color='#667eea')))
    sparkline_fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=10), height=60, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False), yaxis=dict(visible=False)
    )
    
    summary_panel = html.Div([
        html.Div("POSITION-BY-POSITION IMPACT", style={'fontWeight': 'bold', 'color': '#667eea', 'fontSize': '13px', 'marginBottom': '15px', 'textTransform': 'uppercase'}),
        html.Div([
            # Tokens Changed
            html.Div([
                html.Div("TOKENS CHANGED:", style={'fontSize': '11px', 'fontWeight': 'bold', 'color': '#495057'}),
                html.Div(f"{tokens_changed}/{max_len}", style={'fontSize': '28px', 'fontWeight': 'bold', 'color': '#212529', 'lineHeight': '1.2'}),
                html.Div(f"{percent_changed:.1f}% of sequence modified", style={'fontSize': '11px', 'color': '#6c757d'})
            ], style={'flex': '1', 'borderRight': '1px solid #dee2e6', 'paddingRight': '15px'}),
            
            # Avg Prob Shift
            html.Div([
                html.Div("AVERAGE PROBABILITY SHIFT:", style={'fontSize': '11px', 'fontWeight': 'bold', 'color': '#495057'}),
                html.Div([
                    html.Span(f"{avg_prob_shift*100:+.1f}%", style={'color': '#dc3545' if avg_prob_shift < 0 else '#28a745', 'marginRight': '5px'}),
                    html.I(className=f"fas {'fa-arrow-down' if avg_prob_shift < 0 else 'fa-arrow-up'}", style={'color': '#dc3545' if avg_prob_shift < 0 else '#28a745'})
                ], style={'fontSize': '28px', 'fontWeight': 'bold', 'lineHeight': '1.2'}),
                html.Div("Average shift in confidence", style={'fontSize': '11px', 'color': '#6c757d'})
            ], style={'flex': '1', 'padding': '0 15px', 'textAlign': 'center'})
            
        ], style={'display': 'flex', 'alignItems': 'center'})
    ], style={
        'border': '1px solid #667eea40', 'borderRadius': '12px', 'padding': '20px', 'backgroundColor': '#f8faff'
    })
    
    results.append(summary_panel)
        
    return html.Div(results)
