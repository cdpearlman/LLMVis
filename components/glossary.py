"""
Glossary component providing educational definitions for key terms.
"""

from dash import html, dcc

def create_glossary_modal():
    """
    Create the hidden glossary modal that appears when the Help button is clicked.
    """
    return html.Div([
        html.Div([
            html.Div([
                html.H2("Transformer Concept Glossary", style={'marginTop': '0', 'color': '#2d3748'}),
                html.Button('×', id='close-glossary-btn', className='close-button', 
                           style={'position': 'absolute', 'right': '15px', 'top': '15px', 
                                  'background': 'none', 'border': 'none', 'fontSize': '24px', 'cursor': 'pointer'})
            ], style={'position': 'relative', 'borderBottom': '1px solid #e2e8f0', 'paddingBottom': '15px', 'marginBottom': '15px'}),
            
            html.Div([
                _create_term_entry(
                    "Tokenization", 
                    "Breaking text into pieces", 
                    "Models don't read words like we do. They break text into small chunks called 'tokens'. A token can be a whole word (like 'apple'), part of a word (like 'ing' in 'playing'), or even a space."
                ),
                _create_term_entry(
                    "Embedding", 
                    "Converting tokens to numbers",
                    "Once text is tokenized, each token is converted into a list of numbers (a vector). This vector represents the meaning of the token. Words with similar meanings (like 'dog' and 'puppy') have similar vectors."
                ),
                _create_term_entry(
                    "Attention", 
                    "Context Lookup",
                    "This is how the model understands context. When processing a word (like 'it'), the model 'pays attention' to other words in the sentence (like 'the cat') to figure out what 'it' refers to. It's like a spotlight shining on relevant past information."
                ),
                _create_term_entry(
                    "Residual Stream", 
                    "The Information Highway",
                    "Think of this as a conveyor belt carrying the model's current understanding of the sentence. As it passes through each layer, the layer adds new information to it (via addition), refining the prediction step-by-step."
                ),
                _create_term_entry(
                    "Logits / Log-Probs", 
                    "Prediction Scores",
                    "The raw scores the model assigns to every possible next token. Higher scores mean the model thinks that token is more likely to come next."
                ),
                _create_term_entry(
                    "Beam Search", 
                    "Exploring Multiple Paths",
                    "Instead of just picking the single best next word, Beam Search explores several likely future paths simultaneously (like parallel universes) and picks the one that makes the most sense overall."
                )
            ], className="glossary-content", style={'maxHeight': '60vh', 'overflowY': 'auto', 'padding': '0 20px 10px 10px'})
            
        ], id='glossary-modal-content', className="modal-content", style={
            'backgroundColor': 'white',
            'padding': '30px',
            'borderRadius': '8px',
            'maxWidth': '600px',
            'width': '90%',
            'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
            'position': 'relative',
            'display': 'block' 
        })
    ], id='glossary-modal-overlay', style={
        'position': 'fixed',
        'top': '0',
        'left': '0',
        'width': '100%',
        'height': '100%',
        'backgroundColor': 'rgba(0,0,0,0.5)',
        'zIndex': '1000',
        'display': 'none',
        'alignItems': 'center',
        'justifyContent': 'center'
    })

def _create_term_entry(term, analogy, definition):
    return html.Div([
        html.Div([
            html.H4(term, style={'margin': '0', 'color': '#4a5568'}),
            html.Span(analogy, style={'fontSize': '12px', 'backgroundColor': '#ebf8ff', 'color': '#2b6cb0', 'padding': '2px 8px', 'borderRadius': '12px', 'marginLeft': '10px'})
        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '5px'}),
        html.P(definition, style={'color': '#718096', 'fontSize': '14px', 'lineHeight': '1.5', 'marginTop': '0'})
    ], style={'marginBottom': '20px', 'borderBottom': '1px solid #f7fafc', 'paddingBottom': '15px'})
