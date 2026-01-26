"""
Tokenization panel component for visualizing the tokenization process.

Displays tokens in vertical rows: [token] → [ID] → [embedding] per token.
"""

from dash import html, dcc


def create_static_tokenization_diagram():
    """Create static HTML/CSS diagram showing example tokenization flow."""
    return html.Div([
        html.H4("Example: How text becomes model input", 
                style={'marginBottom': '1rem', 'color': '#495057', 'fontSize': '16px'}),
        
        # Example flow: text -> tokens -> IDs -> embeddings
        html.Div([
            # Input text
            html.Div([
                html.Div('"Hello world"', 
                        className='example-text',
                        style={'padding': '8px 12px', 'backgroundColor': '#e9ecef', 
                               'borderRadius': '4px', 'fontFamily': 'monospace'})
            ], style={'flex': '1', 'textAlign': 'center'}),
            
            html.Div('→', style={'padding': '0 10px', 'fontSize': '20px', 'color': '#6c757d'}),
            
            # Tokens
            html.Div([
                html.Div([
                    html.Span('["Hello"', style={'fontFamily': 'monospace'}),
                    html.Span(', " world"]', style={'fontFamily': 'monospace'})
                ], style={'padding': '8px 12px', 'backgroundColor': '#d4edff', 
                         'borderRadius': '4px'})
            ], style={'flex': '1', 'textAlign': 'center'}),
            
            html.Div('→', style={'padding': '0 10px', 'fontSize': '20px', 'color': '#6c757d'}),
            
            # IDs
            html.Div([
                html.Div('[1234, 5678]', 
                        style={'padding': '8px 12px', 'backgroundColor': '#ffe5d4', 
                               'borderRadius': '4px', 'fontFamily': 'monospace'})
            ], style={'flex': '1', 'textAlign': 'center'}),
            
            html.Div('→', style={'padding': '0 10px', 'fontSize': '20px', 'color': '#6c757d'}),
            
            # Embeddings
            html.Div([
                html.Div('[[ ... ], [ ... ]]', 
                        style={'padding': '8px 12px', 'backgroundColor': '#e5d4ff', 
                               'borderRadius': '4px', 'fontFamily': 'monospace'})
            ], style={'flex': '1', 'textAlign': 'center'})
            
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center',
                 'padding': '1rem', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px',
                 'border': '1px solid #dee2e6'})
    ], style={'marginBottom': '2rem'})


def create_tokenization_panel():
    """Create the tokenization visualization panel with three columns."""
    return html.Div([
        # Section title and subtitle
        html.Div([
            html.H3("Step 1: Tokenization & Embedding", 
                   className="section-title",
                   style={'marginBottom': '0.5rem'}),
            html.P("This is the first step in processing text through a transformer model. "
                  "The input text is broken into tokens, converted to IDs, and embedded as vectors.",
                  style={'color': '#6c757d', 'fontSize': '14px', 'marginBottom': '1.5rem'})
        ]),
        
        # Static example diagram (always visible)
        create_static_tokenization_diagram(),
        
        # Dynamic tokenization display container (populated by callback)
        html.Div(id='tokenization-display-container', children=[
            # This will be populated after analysis runs
        ])
        
    ], id='tokenization-panel', style={'display': 'none'}, className='tokenization-section')


def create_tokenization_display(tokens_list, token_ids_list, color_palette=None):
    """
    Create a vertical tokenization display showing each token's flow.
    
    Args:
        tokens_list: List of token strings
        token_ids_list: List of token IDs
        color_palette: Optional list of colors for each token (auto-generated if None)
    
    Returns:
        Dash HTML component with vertical token rows: [token] → [ID] → [embedding]
    """
    if color_palette is None:
        # Generate distinct colors for each token
        color_palette = generate_token_colors(len(tokens_list))
    
    preview_token = tokens_list[0] if tokens_list else ""
    preview_id = token_ids_list[0] if token_ids_list else ""
    preview_color = color_palette[0] if color_palette else '#f8f9fa'
    
    return html.Details([
        html.Summary(
            html.Div([
                html.Span("Tokenization preview:", style={'color': '#6c757d', 'fontSize': '13px'}),
                html.Span(preview_token, style={
                    'padding': '4px 8px',
                    'backgroundColor': preview_color,
                    'borderRadius': '4px',
                    'fontFamily': 'monospace',
                    'fontSize': '12px'
                }),
                html.Span('→', style={'color': '#6c757d'}),
                html.Span(str(preview_id), style={
                    'padding': '4px 8px',
                    'backgroundColor': '#ffe5d4',
                    'borderRadius': '4px',
                    'fontFamily': 'monospace',
                    'fontSize': '12px'
                }),
                html.Span('→', style={'color': '#6c757d'}),
                html.Span('[ ... ]', style={
                    'padding': '4px 8px',
                    'backgroundColor': '#e5d4ff',
                    'borderRadius': '4px',
                    'fontFamily': 'monospace',
                    'fontSize': '12px'
                }),
                html.Span('...', style={'color': '#6c757d'}),
                html.Span("Expand", style={'marginLeft': 'auto', 'color': '#667eea', 'fontWeight': '500'})
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '8px', 'flexWrap': 'wrap'})
        ),
        
        html.Div([
            html.H4("Full Tokenization:", 
                   style={'marginTop': '1.5rem', 'marginBottom': '1rem', 
                          'color': '#495057', 'fontSize': '16px'}),
            
            # Column headers row
            html.Div([
                html.Span("Token", className='token-header', 
                         style={'flex': '1', 'fontWeight': '600', 'color': '#495057', 'fontSize': '13px'}),
                html.Span("", style={'width': '32px'}),  # Arrow spacer
                html.Span("ID", className='token-header',
                         style={'flex': '1', 'fontWeight': '600', 'color': '#495057', 'fontSize': '13px'}),
                html.Span("", style={'width': '32px'}),  # Arrow spacer
                html.Span("Embedding", className='token-header',
                         style={'flex': '1', 'fontWeight': '600', 'color': '#495057', 'fontSize': '13px'})
            ], className='tokenization-header-row',
               style={'display': 'flex', 'alignItems': 'center', 'gap': '4px', 
                      'marginBottom': '0.75rem', 'paddingBottom': '0.5rem',
                      'borderBottom': '1px solid #e9ecef'}),
            
            # Vertical token rows - each row shows [token] → [ID] → [embedding]
            html.Div([
                create_token_row(token, token_id, color, idx)
                for idx, (token, token_id, color) in enumerate(zip(tokens_list, token_ids_list, color_palette))
            ], className='tokenization-rows')
            
        ], style={'padding': '1rem', 'backgroundColor': '#ffffff', 
                  'borderRadius': '8px', 'border': '1px solid #dee2e6'})
        
    ], open=False, style={'marginTop': '1rem'})


def create_token_row(token, token_id, color, idx):
    """
    Create a single horizontal row showing: [token] → [ID] → [embedding].
    
    Args:
        token: Token string
        token_id: Token ID number
        color: Background color for the token
        idx: Index of the token (for key uniqueness)
    
    Returns:
        Dash HTML component for a single token row
    """
    # Tooltip text for educational purposes
    tooltips = {
        'token': "The text is broken into 'tokens' - small pieces like words or parts of words. "
                "This is how the model reads text. Breaking words into smaller pieces lets the model "
                "understand new words by combining pieces it already knows.",
        'id': "Each token gets a unique number (ID) from the model's dictionary. "
             "Think of it like a phonebook - every token has its own number. "
             "The model uses these numbers instead of the actual text.",
        'embedding': "Each token number is turned into a list of numbers called an 'embedding.' "
                    "These numbers capture the token's meaning. Similar words get similar numbers. "
                    "This list of numbers is what actually goes into the model's layers."
    }
    
    return html.Div([
        # Token box
        html.Div(
            token,
            className='token-row-box token-row-token',
            style={
                'flex': '1',
                'padding': '8px 12px',
                'backgroundColor': color,
                'borderRadius': '6px',
                'border': f'2px solid {darken_color(color)}',
                'fontFamily': 'monospace',
                'fontSize': '13px',
                'textAlign': 'center',
                'wordBreak': 'break-word',
                'minWidth': '60px'
            },
            title=tooltips['token']
        ),
        
        # Arrow
        html.Span('→', className='token-row-arrow',
                  style={'color': '#6c757d', 'fontSize': '16px', 'padding': '0 8px'}),
        
        # ID box
        html.Div(
            str(token_id),
            className='token-row-box token-row-id',
            style={
                'flex': '1',
                'padding': '8px 12px',
                'backgroundColor': '#ffe5d4',
                'borderRadius': '6px',
                'border': '2px solid #e6cfc0',
                'fontFamily': 'monospace',
                'fontSize': '13px',
                'textAlign': 'center',
                'minWidth': '60px'
            },
            title=tooltips['id']
        ),
        
        # Arrow
        html.Span('→', className='token-row-arrow',
                  style={'color': '#6c757d', 'fontSize': '16px', 'padding': '0 8px'}),
        
        # Embedding box
        html.Div(
            '[ ... ]',
            className='token-row-box token-row-embedding',
            style={
                'flex': '1',
                'padding': '8px 12px',
                'backgroundColor': '#e5d4ff',
                'borderRadius': '6px',
                'border': '2px solid #cfbfe6',
                'fontFamily': 'monospace',
                'fontSize': '13px',
                'textAlign': 'center',
                'minWidth': '60px'
            },
            title=tooltips['embedding']
        )
        
    ], className='token-row',
       style={'display': 'flex', 'alignItems': 'center', 'gap': '4px', 'marginBottom': '8px'})


def generate_token_colors(num_tokens):
    """Generate a list of distinct colors for tokens."""
    # Predefined pleasant color palette
    base_colors = [
        '#ffcccb',  # Light red
        '#add8e6',  # Light blue
        '#90ee90',  # Light green
        '#ffb6c1',  # Light pink
        '#ffd700',  # Gold
        '#dda0dd',  # Plum
        '#f0e68c',  # Khaki
        '#ff6347',  # Tomato
        '#98fb98',  # Pale green
        '#87ceeb',  # Sky blue
        '#ffa07a',  # Light salmon
        '#da70d6',  # Orchid
    ]
    
    # Cycle through colors if we have more tokens than colors
    colors = []
    for i in range(num_tokens):
        colors.append(base_colors[i % len(base_colors)])
    
    return colors


def darken_color(hex_color, factor=0.8):
    """Darken a hex color by a factor."""
    # Remove '#' if present
    hex_color = hex_color.lstrip('#')
    
    # Convert to RGB
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    
    # Darken
    r, g, b = int(r * factor), int(g * factor), int(b * factor)
    
    # Convert back to hex
    return f'#{r:02x}{g:02x}{b:02x}'

