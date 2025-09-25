"""
Top bar component with model selection, prompt input, and module discovery button.
"""

import dash
from dash import dcc, html
import logging

from ..constants import SUPPORTED_MODELS, COMPONENT_IDS, MAX_PROMPT_WORDS, CSS_CLASSES

logger = logging.getLogger(__name__)

def validate_prompt_length(raw_prompt: str, max_words: int = MAX_PROMPT_WORDS) -> tuple[str, bool, str]:
    """
    Validate and potentially truncate prompt to word limit.
    
    Args:
        raw_prompt: Raw input prompt
        max_words: Maximum number of words allowed
        
    Returns:
        Tuple of (processed_prompt, was_truncated, message)
    """
    words = raw_prompt.strip().split()
    
    if len(words) <= max_words:
        return raw_prompt.strip(), False, ""
    
    # Truncate to max words
    truncated_words = words[:max_words]
    truncated_prompt = " ".join(truncated_words)
    
    message = f"Prompt truncated to {max_words} words"
    logger.info(f"Prompt truncated: {len(words)} -> {max_words} words")
    
    return truncated_prompt, True, message

def create_top_bar() -> html.Div:
    """
    Create the top bar component with model selection and prompt input.
    
    Returns:
        Dash HTML div containing the top bar components
    """
    return html.Div([
        html.Div([
            # Model selection
            html.Div([
                html.Label("Model:", className="input-label"),
                dcc.Dropdown(
                    id=COMPONENT_IDS["model_dropdown"],
                    options=[{"label": model, "value": model} for model in SUPPORTED_MODELS],
                    value=SUPPORTED_MODELS[0],  # Default to first model
                    clearable=False,
                    className="model-dropdown"
                )
            ], className="input-group"),
            
            # Prompt input
            html.Div([
                html.Label("Prompt:", className="input-label"),
                dcc.Textarea(
                    id=COMPONENT_IDS["prompt_input"],
                    placeholder=f"Enter your prompt here (max {MAX_PROMPT_WORDS} words)...",
                    value="The quick brown fox jumps over the lazy dog",  # Default prompt
                    rows=3,
                    className="prompt-input"
                )
            ], className="input-group"),
            
            # Find modules button
            html.Div([
                html.Button(
                    "Find Module Names",
                    id=COMPONENT_IDS["find_modules_btn"],
                    n_clicks=0,
                    className="btn btn-primary",
                    disabled=False
                )
            ], className="button-group"),
            
        ], className="top-bar-controls"),
        
        # Status message area
        html.Div(
            id=COMPONENT_IDS["prompt_status"],
            className="status-message"
        )
        
    ], className=CSS_CLASSES["top_bar"])

# Callback for prompt validation will be defined in main app.py

def get_processed_prompt(raw_prompt: str) -> str:
    """
    Get the processed (potentially truncated) prompt.
    
    Args:
        raw_prompt: Raw prompt input
        
    Returns:
        Processed prompt string
    """
    processed_prompt, _, _ = validate_prompt_length(raw_prompt)
    return processed_prompt
