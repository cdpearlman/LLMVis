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
           disabled=True, style={'width': '100%', 'marginBottom': '16px'}),
        
        # Results container
        html.Div(id='ablation-results-container', children=[
            # Will be populated by callback
        ])
        
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


def create_ablation_results_display(original_token, ablated_token, original_prob, ablated_prob,
                                    selected_heads, selected_beam=None, ablated_beam=None):
    """
    Create the ablation results display focusing on full generation comparison.
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
    
    # Generation Comparison (Main Display)
    if selected_beam and selected_beam.get('text') and ablated_beam and ablated_beam.get('text'):
        gen_changed = selected_beam['text'] != ablated_beam['text']
        
        results.append(html.Div([
            html.H6("Full Generation Comparison", style={'color': '#495057', 'marginBottom': '15px'}),
            
            # Comparison Grid
            html.Div([
                # Original Generation
                html.Div([
                    html.Div("Original Generation", style={'fontSize': '12px', 'fontWeight': 'bold', 'color': '#28a745', 'marginBottom': '8px'}),
                    html.Div(selected_beam['text'], style={
                        'fontFamily': 'monospace',
                        'fontSize': '13px',
                        'padding': '12px',
                        'backgroundColor': '#f8f9fa',
                        'border': '1px solid #dee2e6', 
                        'borderRadius': '6px',
                        'whiteSpace': 'pre-wrap',
                        'height': '100%'
                    })
                ], style={'flex': '1', 'marginRight': '10px'}),
                
                # Ablated Generation
                html.Div([
                    html.Div("Ablated Generation", style={'fontSize': '12px', 'fontWeight': 'bold', 'color': '#dc3545', 'marginBottom': '8px'}),
                    html.Div(ablated_beam['text'], style={
                        'fontFamily': 'monospace',
                        'fontSize': '13px',
                        'padding': '12px',
                        'backgroundColor': '#fff5f5' if gen_changed else '#f8f9fa',
                        'border': '1px solid #dee2e6',
                        'borderRadius': '6px',
                        'whiteSpace': 'pre-wrap',
                        'height': '100%'
                    })
                ], style={'flex': '1', 'marginLeft': '10px'})
            ], style={'display': 'flex', 'alignItems': 'stretch'}),
            
            # Generation Change Indicator
            html.Div([
                html.I(className=f"fas {'fa-exclamation-circle' if gen_changed else 'fa-check-circle'}", 
                       style={'color': '#dc3545' if gen_changed else '#28a745', 'marginRight': '8px'}),
                html.Span(
                    "The generated sequence changed significantly after ablation."
                    if gen_changed else
                    "The generated sequence remained identical.",
                    style={'fontWeight': '500', 'color': '#495057'}
                )
            ], style={'marginTop': '15px', 'fontSize': '13px'}),

            # Probability info as secondary context
            html.Div([
                html.Hr(style={'margin': '15px 0', 'borderTop': '1px dotted #dee2e6'}),
                html.Span("Immediate next-token probability: ", style={'color': '#6c757d', 'fontSize': '12px'}),
                html.Span(f"{original_prob:.1%} → {ablated_prob:.1%} ", 
                         style={'fontSize': '12px', 'fontWeight': 'bold', 'color': '#495057'}),
                html.Span(f"({ablated_prob - original_prob:+.1%})", 
                         style={'fontSize': '12px', 'color': '#dc3545' if ablated_prob < original_prob else '#28a745'})
            ], style={'marginTop': '10px'})
            
        ]))
    else:
        # Fallback if beam data is missing
        results.append(html.Div([
            html.I(className='fas fa-info-circle', style={'color': '#6c757d', 'marginRight': '8px'}),
            html.Span("Run a full analysis first to select a generation for comparison.", 
                     style={'color': '#6c757d', 'fontSize': '14px'})
        ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px', 'border': '1px solid #dee2e6'}))
        
    return html.Div(results, style={
        'padding': '20px',
        'backgroundColor': '#fafbfc',
        'borderRadius': '8px',
        'border': '1px solid #e2e8f0'
    })
