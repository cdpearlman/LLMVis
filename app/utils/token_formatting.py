"""
Token formatting utilities for hover labels and display.
Consistent with the formatting used in logit_lens_analysis.py.
"""

import logging
from typing import List, Tuple
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

def format_token_probability(token: str, probability: float) -> str:
    """
    Format a single token and its probability for display.
    Uses the same formatting as logit_lens_analysis.py:
    - prob < 0.001 => 6 decimals
    - else => 3 decimals
    
    Args:
        token: Token string
        probability: Token probability (0.0 to 1.0)
        
    Returns:
        Formatted string like "token(0.123)" or "▁the(0.000045)"
    """
    if probability < 0.001:
        return f'{token}({probability:.6f})'
    else:
        return f'{token}({probability:.3f})'

def format_topk_for_hover(layer_results: List[Tuple[str, float]]) -> List[str]:
    """
    Format top-k token results for hover display.
    
    Args:
        layer_results: List of (token, probability) tuples
        
    Returns:
        List of formatted strings
    """
    return [format_token_probability(token, prob) for token, prob in layer_results]

def format_topk_for_display(layer_results: List[Tuple[str, float]]) -> str:
    """
    Format top-k token results for display as a single string.
    
    Args:
        layer_results: List of (token, probability) tuples
        
    Returns:
        Comma-separated formatted string
    """
    formatted = format_topk_for_hover(layer_results)
    return ', '.join(formatted)

def get_printable_tokens(tokenizer: AutoTokenizer, input_ids: List[int]) -> List[str]:
    """
    Convert input IDs to printable token strings for visualization.
    Preserves formatting consistent with BertViz expectations.
    
    Args:
        tokenizer: Tokenizer instance
        input_ids: List of token IDs
        
    Returns:
        List of printable token strings
    """
    # Convert IDs to tokens
    raw_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # Clean tokens: replace BPE space markers with proper spacing
    # This handles different tokenizer conventions (GPT-2 uses Ġ, others may vary)
    cleaned_tokens = []
    for token in raw_tokens:
        if token.startswith('Ġ'):
            # GPT-2 style: Ġ prefix indicates space
            cleaned_token = ' ' + token[1:]
        elif token.startswith('▁'):
            # SentencePiece style: ▁ prefix indicates space
            cleaned_token = ' ' + token[1:]
        else:
            cleaned_token = token
        cleaned_tokens.append(cleaned_token)
    
    logger.info(f"Converted {len(input_ids)} tokens to printable format")
    return cleaned_tokens

def compute_edge_opacity(prob: float, top1_prob: float, min_opacity: float = 0.15) -> float:
    """
    Map probability to opacity for edge visualization.
    
    Args:
        prob: Current token probability
        top1_prob: Highest probability for this layer (for normalization)
        min_opacity: Minimum opacity to ensure visibility
        
    Returns:
        Opacity value between min_opacity and 1.0
    """
    if top1_prob <= 0:
        return min_opacity
    
    # Scale probability relative to top-1, then map to opacity range
    relative_prob = prob / top1_prob
    opacity = min_opacity + (1.0 - min_opacity) * relative_prob
    
    return max(min_opacity, min(1.0, opacity))

def create_hover_text(token: str, probability: float, layer_idx: int, rank: int) -> str:
    """
    Create hover text for edge lines.
    
    Args:
        token: Token string
        probability: Token probability
        layer_idx: Layer index (0-based)
        rank: Rank of this token (1=top, 2=second, 3=third)
        
    Returns:
        Formatted hover text
    """
    formatted_token = format_token_probability(token, probability)
    return f"Layer {layer_idx} → {layer_idx + 1}<br>Rank {rank}: {formatted_token}"

def truncate_token_display(token: str, max_length: int = 15) -> str:
    """
    Truncate long tokens for display purposes.
    
    Args:
        token: Token string
        max_length: Maximum display length
        
    Returns:
        Truncated token with ellipsis if needed
    """
    if len(token) <= max_length:
        return token
    return token[:max_length-3] + "..."

def validate_probability_range(probabilities: List[float]) -> bool:
    """
    Validate that probabilities are in valid range [0, 1].
    
    Args:
        probabilities: List of probability values
        
    Returns:
        True if all probabilities are valid
    """
    for prob in probabilities:
        if not (0.0 <= prob <= 1.0):
            logger.warning(f"Invalid probability: {prob}")
            return False
    return True

def normalize_probabilities(probabilities: List[float]) -> List[float]:
    """
    Normalize probabilities to sum to 1.0 if needed.
    
    Args:
        probabilities: List of probability values
        
    Returns:
        Normalized probabilities
    """
    total = sum(probabilities)
    if total <= 0:
        logger.warning("Invalid probability sum, returning equal distribution")
        return [1.0 / len(probabilities)] * len(probabilities)
    
    if abs(total - 1.0) > 1e-6:  # Allow small floating point errors
        logger.info(f"Normalizing probabilities (sum={total:.6f})")
        return [p / total for p in probabilities]
    
    return probabilities
