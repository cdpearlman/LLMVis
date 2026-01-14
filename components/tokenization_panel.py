"""
Tokenization panel component for visualizing the tokenization process.

Displays three columns: Tokens | IDs | Embeddings with colored connectors.
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
    Create the three-column tokenization display for actual prompt data.
    
    Args:
        tokens_list: List of token strings
        token_ids_list: List of token IDs
        color_palette: Optional list of colors for each token (auto-generated if None)
    
    Returns:
        Dash HTML component with three-column layout
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
            
            # Three-column grid
            html.Div([
                # Column 1: Tokens
                html.Div([
                    html.H5("Tokens", style={'marginBottom': '1rem', 'color': '#495057'}),
                    html.Div([
                        create_token_box(token, color, idx, 'token')
                        for idx, (token, color) in enumerate(zip(tokens_list, color_palette))
                    ], className='token-column')
                ], className='tokenization-col', style={'flex': '1'}),
                
                # Column 2: Token IDs
                html.Div([
                    html.H5("Token IDs", style={'marginBottom': '1rem', 'color': '#495057'}),
                    html.Div([
                        create_token_box(str(token_id), color, idx, 'id')
                        for idx, (token_id, color) in enumerate(zip(token_ids_list, color_palette))
                    ], className='token-column')
                ], className='tokenization-col', style={'flex': '1'}),
                
                # Column 3: Embeddings
                html.Div([
                    html.H5("Embeddings", style={'marginBottom': '1rem', 'color': '#495057'}),
                    html.Div([
                        create_token_box("[ ... ]", color, idx, 'embedding')
                        for idx, color in enumerate(color_palette)
                    ], className='token-column')
                ], className='tokenization-col', style={'flex': '1'})
                
            ], className='tokenization-grid', 
               style={'display': 'flex', 'gap': '2rem', 'alignItems': 'flex-start'})
            
        ], style={'padding': '1rem', 'backgroundColor': '#ffffff', 
                  'borderRadius': '8px', 'border': '1px solid #dee2e6'})
        
    ], open=False, style={'marginTop': '1rem'})


def create_token_box(content, color, idx, box_type):
    """
    Create a single token box with colored border and connecting line.
    
    Args:
        content: Text to display in the box
        color: Background color for the box
        idx: Index of the token (for connector lines)
        box_type: Type of box ('token', 'id', or 'embedding') for tooltip
    
    Returns:
        Dash HTML component for the token box
    """
    # Tooltip text based on box type (written for high school education level)
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
        html.Div(
            content,
            className=f'token-box token-box-{box_type}',
            style={
                'padding': '8px 12px',
                'margin': '8px 0',
                'backgroundColor': color,
                'borderRadius': '6px',
                'border': f'2px solid {darken_color(color)}',
                'fontFamily': 'monospace',
                'fontSize': '13px',
                'textAlign': 'center',
                'position': 'relative',
                'wordBreak': 'break-word'
            },
            title=tooltips[box_type]  # Native HTML tooltip
        )
    ], style={'position': 'relative'})


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

