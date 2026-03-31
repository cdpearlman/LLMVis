"""
Glossary component providing educational definitions for key terms,
grouped by related 3Blue1Brown video explanations.
"""

from dash import html

# --- Glossary data: terms grouped by explanatory video ---

VIDEO_GROUPS = [
    {
        "title": "The Transformer Pipeline",
        "video_id": "wjZofJX0v4M",
        "terms": [
            (
                "Text Splitting (Tokenization)",
                "Breaking text into pieces",
                "Models don't read words like we do. They break text into small "
                "chunks called 'tokens'. A token can be a whole word (like "
                "'apple'), part of a word (like 'ing' in 'playing'), or even a space."
            ),
            (
                "Meaning Encoding (Embedding)",
                "Converting words to numbers",
                "Once text is split into pieces, each piece is converted into a "
                "list of numbers. This list represents the meaning of the piece. "
                "Words with similar meanings (like 'dog' and 'puppy') get similar "
                "numbers."
            ),
            (
                "Confidence Scores (Logits)",
                "Prediction Scores",
                "The raw scores the model assigns to every possible next word. "
                "Higher scores mean the model thinks that word is more likely to "
                "come next."
            ),
            (
                "Softmax",
                "Turning Scores into Probabilities",
                "The function that converts raw prediction scores (logits) into "
                "probabilities — positive numbers that add up to 1.0. This lets "
                "us interpret the model's output as 'how likely is each next word?'"
            ),
            (
                "Residual Stream",
                "The Information Highway",
                "Think of this as a conveyor belt carrying the model's current "
                "understanding of the sentence. As it passes through each layer, "
                "the layer adds new information to it (via addition), refining "
                "the prediction step-by-step."
            ),
        ],
    },
    {
        "title": "The Attention Mechanism",
        "video_id": "eMlx5fFNoYc",
        "terms": [
            (
                "Attention",
                "Context Lookup",
                "This is how the model understands context. When processing a "
                "word (like 'it'), the model 'pays attention' to other words in "
                "the sentence (like 'the cat') to figure out what 'it' refers "
                "to. It's like a spotlight shining on relevant past information."
            ),
            (
                "Attention Detectors (Heads)",
                "Parallel Context Searchers",
                "Instead of having just one attention mechanism, models use "
                "multiple 'detectors' (called 'heads') in parallel. Each "
                "detector can learn to look for different types of relationships "
                "(e.g., one might look for adjectives, while another tracks "
                "pronouns)."
            ),
        ],
    },
    {
        "title": "The Big Picture",
        "video_id": "LPZh9BOjkQs",
        "terms": [
            (
                "Probability Distribution",
                "The Full Prediction Spread",
                "The complete set of probabilities over all possible next words. "
                "Instead of one answer, the model gives a probability for every "
                "word in its vocabulary. The dashboard shows the top 5 most "
                "likely predictions."
            ),
            (
                "Knowledge Retrieval (MLP / Feed-Forward Network)",
                "The Model's Memory Banks",
                "The component in each layer that processes tokens individually. "
                "It acts like a memory lookup — retrieving stored factual "
                "knowledge about the world. It expands the information, applies "
                "learned patterns, then compresses it back down."
            ),
            (
                "Parameters / Weights",
                "The Model's Learned Numbers",
                "The learnable numbers inside the model. These are adjusted "
                "during training to improve predictions. GPT-2 has about 124 "
                "million of them. Every decision the model makes flows through "
                "these numbers."
            ),
            (
                "Training",
                "Learning from Examples",
                "The process of showing the model billions of text examples and "
                "adjusting its parameters to get better at predicting what comes "
                "next. The model doesn't memorize text — it learns patterns and "
                "relationships between words."
            ),
        ],
    },
    {
        "title": "How Models Store Knowledge",
        "video_id": "9-Jl0dxWQs8",
        "terms": [
            (
                "Hidden Dimension",
                "The Width of the Conveyor Belt",
                "The size of the number list representing each token internally. "
                "For GPT-2, each token is represented by 768 numbers at each "
                "layer. A wider conveyor belt means the model can carry more "
                "nuanced information about each word."
            ),
            (
                "Neurons, Weights, and Biases",
                "The Building Blocks",
                "Neurons are the individual units that activate (or don't) based "
                "on their input. Weights control how strongly inputs influence "
                "each neuron. Biases set the threshold for when a neuron 'fires'. "
                "Together, they form the basic machinery of every layer."
            ),
        ],
    },
    {
        "title": "How Models Learn",
        "video_id": "IHZwWFHWa-w",
        "terms": [
            (
                "Gradient",
                "Measuring Influence",
                "A measure of how much each part of the model contributed to its "
                "prediction. In the dashboard's 'Word Influence' tool, gradients "
                "reveal which input words had the biggest effect on the model's "
                "output — like tracing cause and effect."
            ),
            (
                "Loss",
                "The Model's Error Score",
                "A number measuring how wrong the model's predictions are. "
                "During training, the goal is to make this number as small as "
                "possible. Lower loss means the model is making better "
                "predictions."
            ),
        ],
    },
    {
        "title": "Dashboard Tools",
        "video_id": None,
        "terms": [
            (
                "Layer",
                "One Processing Step",
                "One complete processing step in the transformer, containing "
                "both an attention mechanism and a knowledge retrieval (MLP) "
                "component. GPT-2 has 12 layers stacked on top of each other, "
                "each refining the model's understanding."
            ),
            (
                "Beam Search",
                "Exploring Multiple Paths",
                "Instead of just picking the single best next word, Beam Search "
                "explores several likely future paths simultaneously (like "
                "parallel universes) and picks the one that makes the most sense "
                "overall. The 'Options to Generate' setting controls how many "
                "paths are explored at once."
            ),
            (
                "Test by Removing (Ablation)",
                "Digital Brain Surgery",
                "A technique used to understand which parts of a model are "
                "responsible for certain behaviors. By artificially 'turning off' "
                "specific attention detectors, we can measure how much the "
                "model's output changes, revealing the importance of those "
                "components."
            ),
            (
                "Word Influence (Token Attribution)",
                "Tracing What Mattered",
                "A way to measure how much each input word affected the model's "
                "final prediction. Uses gradients to trace the flow of influence "
                "backwards through the model, highlighting which words pushed "
                "the prediction in a particular direction."
            ),
            (
                "Vocabulary",
                "The Model's Dictionary",
                "The complete set of tokens the model knows. GPT-2 has a "
                "vocabulary of about 50,257 tokens. The model can only predict "
                "words that are in its vocabulary — it picks from this fixed set "
                "every time."
            ),
            (
                "Inference / Forward Pass",
                "Running the Model",
                "Using the trained model to make predictions on new text. This "
                "is what happens when you click 'Analyze' in the dashboard — no "
                "learning occurs, the model just processes your input through "
                "all its layers to produce a prediction."
            ),
            (
                "Temperature",
                "Controlling Randomness",
                "A setting that controls how spread out the model's predictions "
                "are. Low temperature makes the model more confident and "
                "predictable (it strongly favors its top pick). High temperature "
                "makes predictions more spread out and creative."
            ),
        ],
    },
]


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
                _create_video_group(group) for group in VIDEO_GROUPS
            ], className="glossary-content-area"),

        ], id='glossary-drawer-content', className="glossary-drawer")
    ], id='glossary-container')


def _create_video_group(group):
    """Render a section with an optional video followed by its related terms."""
    children = [
        html.H3(group["title"], style={
            'color': '#2d3748',
            'fontSize': '16px',
            'fontWeight': '600',
            'marginBottom': '15px',
            'paddingBottom': '8px',
            'borderBottom': '2px solid #e2e8f0',
        })
    ]

    if group["video_id"]:
        children.append(html.Iframe(
            src=f"https://www.youtube.com/embed/{group['video_id']}?rel=0",
            style={
                'width': '100%',
                'height': '350px',
                'border': 'none',
                'borderRadius': '8px',
                'marginBottom': '20px',
            },
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; fullscreen"
        ))

    for term, analogy, definition in group["terms"]:
        children.append(_create_term_entry(term, analogy, definition))

    return html.Div(children, style={'marginBottom': '35px'})


def _create_term_entry(term, analogy, definition):
    """Render a single glossary term with its friendly analogy and definition."""
    return html.Div([
        html.Div([
            html.H4(term, style={'margin': '0', 'color': '#4a5568'}),
            html.Span(analogy, style={'fontSize': '12px', 'backgroundColor': '#ebf8ff', 'color': '#2b6cb0', 'padding': '2px 8px', 'borderRadius': '12px', 'marginLeft': '10px'})
        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
        html.P(definition, style={'color': '#718096', 'fontSize': '14px', 'lineHeight': '1.5', 'marginTop': '0', 'marginBottom': '0'})
    ], style={'marginBottom': '20px', 'paddingBottom': '15px', 'borderBottom': '1px solid #f7fafc'})
