"""
Glossary component providing educational definitions for key terms.
"""

from dash import html, dcc

def create_glossary_modal():
    """
    Create the hidden glossary modal that appears when the Help button is clicked.
    """
    return html.Div([
        html.Div(id='glossary-overlay-bg', className='glossary-overlay'),
        
        html.Div([
            html.Div([
                html.H2("Transformer Concept Glossary"),
                html.Button('×', id='close-glossary-btn', className='close-button', 
                           style={'background': 'none', 'border': 'none', 'fontSize': '28px', 'cursor': 'pointer', 'color': '#a0aec0'})
            ], className='glossary-header'),
            
            html.Div([
                _create_term_entry(
                    "Tokenization", 
                    "Breaking text into pieces", 
                    "Models don't read words like we do. They break text into small chunks called 'tokens'. A token can be a whole word (like 'apple'), part of a word (like 'ing' in 'playing'), or even a space.",
                    "https://www.youtube.com/embed/wjZofJX0v4M?start=0"
                ),
                _create_term_entry(
                    "Embedding", 
                    "Converting tokens to numbers",
                    "Once text is tokenized, each token is converted into a list of numbers (a vector). This vector represents the meaning of the token. Words with similar meanings (like 'dog' and 'puppy') have similar vectors.",
                    "https://www.youtube.com/embed/wjZofJX0v4M?start=195"
                ),
                _create_term_entry(
                    "Attention", 
                    "Context Lookup",
                    "This is how the model understands context. When processing a word (like 'it'), the model 'pays attention' to other words in the sentence (like 'the cat') to figure out what 'it' refers to. It's like a spotlight shining on relevant past information.",
                    "https://www.youtube.com/embed/eMlx5fFNoYc?start=0"
                ),
                _create_term_entry(
                    "Attention Heads",
                    "Parallel Context Searchers",
                    "Instead of having just one attention mechanism, models use multiple 'heads' in parallel. Each head can learn to look for different types of relationships (e.g., one head might look for adjectives, while another tracks pronouns).",
                    "https://www.youtube.com/embed/eMlx5fFNoYc?start=420"
                ),
                _create_term_entry(
                    "Residual Stream", 
                    "The Information Highway",
                    "Think of this as a conveyor belt carrying the model's current understanding of the sentence. As it passes through each layer, the layer adds new information to it (via addition), refining the prediction step-by-step.",
                    "https://www.youtube.com/embed/wjZofJX0v4M?start=1173"
                ),
                _create_term_entry(
                    "Logits / Log-Probs", 
                    "Prediction Scores",
                    "The raw scores the model assigns to every possible next token. Higher scores mean the model thinks that token is more likely to come next.",
                    "https://www.youtube.com/embed/wjZofJX0v4M?start=850"
                ),
                _create_term_entry(
                    "Beam Search", 
                    "Exploring Multiple Paths",
                    "Instead of just picking the single best next word, Beam Search explores several likely future paths simultaneously (like parallel universes) and picks the one that makes the most sense overall. The 'Number of Generation Choices' setting controls how many paths are explored at once."
                ),
                _create_term_entry(
                    "Ablation (Activation Patching)", 
                    "Digital Brain Surgery",
                    "A technique used to understand which parts of a model are responsible for certain behaviors. By artificially modifying or 'turning off' specific attention heads or activations, we can measure how much the model's output changes, revealing the importance of those components."
                )
            ], className="glossary-content-area"),
            
        ], id='glossary-drawer-content', className="glossary-drawer")
    ], id='glossary-container')

def _create_term_entry(term, analogy, definition, video_url=None):
    content = [
        html.Div([
            html.H4(term, style={'margin': '0', 'color': '#4a5568'}),
            html.Span(analogy, style={'fontSize': '12px', 'backgroundColor': '#ebf8ff', 'color': '#2b6cb0', 'padding': '2px 8px', 'borderRadius': '12px', 'marginLeft': '10px'})
        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
        html.P(definition, style={'color': '#718096', 'fontSize': '14px', 'lineHeight': '1.5', 'marginTop': '0', 'marginBottom': '15px' if video_url else '0'})
    ]
    
    if video_url:
        content.append(html.Iframe(
            src=video_url,
            style={
                'width': '100%',
                'height': '350px',
                'border': 'none',
                'borderRadius': '8px',
                'marginTop': '10px'
            },
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; fullscreen"
        ))

    return html.Div(content, style={'marginBottom': '30px', 'borderBottom': '1px solid #f7fafc', 'paddingBottom': '20px'})
