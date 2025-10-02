"""
Attention Head Detection and Categorization

Implements heuristics to categorize attention heads into:
- Previous-Token Heads: high attention on previous token
- First/Positional Heads: high attention on first token or positional patterns
- Bag-of-Words Heads: diffuse attention on content tokens
- Syntactic Heads: dependency-like patterns
- Other: heads that don't fit the above categories
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import re


class HeadCategorizationConfig:
    """Configuration for attention head categorization heuristics."""
    
    def __init__(self):
        # Previous-token head thresholds
        self.prev_token_threshold = 0.5  # Minimum avg attention to prev token
        self.prev_token_diagonal_offset = 1  # Check i → i-1 pattern
        
        # First/Positional head thresholds
        self.first_token_threshold = 0.3  # Minimum avg attention to first token
        self.positional_pattern_threshold = 0.4  # For detecting positional patterns
        
        # Bag-of-words head thresholds
        self.bow_entropy_threshold = 0.7  # Minimum entropy (normalized)
        self.bow_max_attention_threshold = 0.3  # Maximum attention to any single token
        
        # Syntactic head thresholds
        self.syntactic_distance_pattern_threshold = 0.4  # For detecting distance patterns
        
        # General thresholds
        self.min_seq_len = 3  # Minimum sequence length for reliable detection


def compute_attention_entropy(attention_weights: torch.Tensor) -> float:
    """
    Compute normalized entropy of attention distribution.
    
    Args:
        attention_weights: [seq_len] tensor of attention weights for a position
    
    Returns:
        Normalized entropy (0 to 1)
    """
    # Avoid log(0) by adding small epsilon
    epsilon = 1e-10
    weights = attention_weights + epsilon
    
    # Compute entropy: -sum(p * log(p))
    entropy = -torch.sum(weights * torch.log(weights))
    
    # Normalize by max entropy (log(n) where n is sequence length)
    max_entropy = np.log(len(weights))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    return normalized_entropy.item()


def detect_previous_token_head(attention_matrix: torch.Tensor, config: HeadCategorizationConfig) -> Tuple[bool, float]:
    """
    Detect if head shows strong previous-token pattern (i → i-1).
    
    Args:
        attention_matrix: [seq_len, seq_len] attention weights
        config: Configuration object
    
    Returns:
        (is_prev_token_head, score) where score is avg attention to previous token
    """
    seq_len = attention_matrix.shape[0]
    
    if seq_len < config.min_seq_len:
        return False, 0.0
    
    # Extract the diagonal offset by 1 (i → i-1 pattern)
    # For each position i > 0, check attention to position i-1
    prev_token_attentions = []
    for i in range(1, seq_len):
        prev_token_attentions.append(attention_matrix[i, i-1].item())
    
    avg_prev_attention = np.mean(prev_token_attentions)
    is_prev_token_head = avg_prev_attention >= config.prev_token_threshold
    
    return is_prev_token_head, avg_prev_attention


def detect_first_token_head(attention_matrix: torch.Tensor, config: HeadCategorizationConfig) -> Tuple[bool, float]:
    """
    Detect if head shows strong attention to first token(s) or positional patterns.
    
    Args:
        attention_matrix: [seq_len, seq_len] attention weights
        config: Configuration object
    
    Returns:
        (is_first_token_head, score) where score is avg attention to first token
    """
    seq_len = attention_matrix.shape[0]
    
    if seq_len < config.min_seq_len:
        return False, 0.0
    
    # Check average attention to first token across all positions
    first_token_attention = attention_matrix[:, 0].mean().item()
    is_first_token_head = first_token_attention >= config.first_token_threshold
    
    return is_first_token_head, first_token_attention


def detect_bow_head(attention_matrix: torch.Tensor, config: HeadCategorizationConfig) -> Tuple[bool, float]:
    """
    Detect if head shows bag-of-words pattern (diffuse attention).
    
    Args:
        attention_matrix: [seq_len, seq_len] attention weights
        config: Configuration object
    
    Returns:
        (is_bow_head, score) where score is average entropy
    """
    seq_len = attention_matrix.shape[0]
    
    if seq_len < config.min_seq_len:
        return False, 0.0
    
    # Compute entropy for each position's attention distribution
    entropies = []
    max_attentions = []
    
    for i in range(seq_len):
        entropy = compute_attention_entropy(attention_matrix[i])
        max_attention = attention_matrix[i].max().item()
        
        entropies.append(entropy)
        max_attentions.append(max_attention)
    
    avg_entropy = np.mean(entropies)
    avg_max_attention = np.mean(max_attentions)
    
    # BoW heads have high entropy and low max attention (diffuse)
    is_bow_head = (avg_entropy >= config.bow_entropy_threshold and 
                   avg_max_attention <= config.bow_max_attention_threshold)
    
    return is_bow_head, avg_entropy


def detect_syntactic_head(attention_matrix: torch.Tensor, config: HeadCategorizationConfig) -> Tuple[bool, float]:
    """
    Detect if head shows syntactic/dependency-like patterns.
    
    This is a simplified heuristic based on consistent distance patterns.
    
    Args:
        attention_matrix: [seq_len, seq_len] attention weights
        config: Configuration object
    
    Returns:
        (is_syntactic_head, score) where score is pattern consistency
    """
    seq_len = attention_matrix.shape[0]
    
    if seq_len < config.min_seq_len:
        return False, 0.0
    
    # Check for consistent distance patterns (e.g., attending to tokens at fixed distances)
    # This is a simplified approach; more sophisticated syntactic detection would
    # require parsing or linguistic features
    
    distance_scores = []
    
    for i in range(seq_len):
        # For each position, find the most attended position
        max_idx = torch.argmax(attention_matrix[i]).item()
        distance = abs(i - max_idx)
        
        # Collect distances (excluding self-attention at distance 0)
        if distance > 0:
            distance_scores.append(distance)
    
    if not distance_scores:
        return False, 0.0
    
    # Check if there's a consistent distance pattern
    # (simple version: low variance in distances)
    distance_variance = np.var(distance_scores)
    distance_mean = np.mean(distance_scores)
    
    # Syntactic heads often have moderate, consistent distances
    # (not too short like prev-token, not too diffuse like BoW)
    pattern_score = 1.0 / (1.0 + distance_variance) if distance_mean > 1 else 0.0
    is_syntactic_head = pattern_score >= config.syntactic_distance_pattern_threshold
    
    return is_syntactic_head, pattern_score


def categorize_attention_head(attention_matrix: torch.Tensor, 
                               layer_idx: int, 
                               head_idx: int,
                               config: Optional[HeadCategorizationConfig] = None) -> Dict[str, Any]:
    """
    Categorize a single attention head based on its attention pattern.
    
    Args:
        attention_matrix: [seq_len, seq_len] attention weights for this head
        layer_idx: Layer index
        head_idx: Head index within the layer
        config: Configuration object (uses defaults if None)
    
    Returns:
        Dictionary with categorization results:
        {
            'layer': layer_idx,
            'head': head_idx,
            'category': str (one of: 'previous_token', 'first_token', 'bow', 'syntactic', 'other'),
            'scores': dict of scores for each category,
            'label': formatted label like "L{layer}-H{head}"
        }
    """
    if config is None:
        config = HeadCategorizationConfig()
    
    # Run all detection heuristics
    is_prev, prev_score = detect_previous_token_head(attention_matrix, config)
    is_first, first_score = detect_first_token_head(attention_matrix, config)
    is_bow, bow_score = detect_bow_head(attention_matrix, config)
    is_syn, syn_score = detect_syntactic_head(attention_matrix, config)
    
    # Assign category based on highest-scoring pattern
    # Priority: previous_token > first_token > bow > syntactic > other
    scores = {
        'previous_token': prev_score if is_prev else 0.0,
        'first_token': first_score if is_first else 0.0,
        'bow': bow_score if is_bow else 0.0,
        'syntactic': syn_score if is_syn else 0.0
    }
    
    # Determine primary category
    if is_prev:
        category = 'previous_token'
    elif is_first:
        category = 'first_token'
    elif is_bow:
        category = 'bow'
    elif is_syn:
        category = 'syntactic'
    else:
        category = 'other'
    
    return {
        'layer': layer_idx,
        'head': head_idx,
        'category': category,
        'scores': scores,
        'label': f"L{layer_idx}-H{head_idx}"
    }


def categorize_all_heads(activation_data: Dict[str, Any], 
                         config: Optional[HeadCategorizationConfig] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Categorize all attention heads in the model.
    
    Args:
        activation_data: Output from execute_forward_pass with attention data
        config: Configuration object (uses defaults if None)
    
    Returns:
        Dictionary mapping category names to lists of head info dicts:
        {
            'previous_token': [...],
            'first_token': [...],
            'bow': [...],
            'syntactic': [...],
            'other': [...]
        }
    """
    if config is None:
        config = HeadCategorizationConfig()
    
    # Initialize result dict
    categorized = {
        'previous_token': [],
        'first_token': [],
        'bow': [],
        'syntactic': [],
        'other': []
    }
    
    attention_outputs = activation_data.get('attention_outputs', {})
    if not attention_outputs:
        return categorized
    
    # Process each layer's attention
    for module_name, output_dict in attention_outputs.items():
        # Extract layer number from module name
        numbers = re.findall(r'\d+', module_name)
        if not numbers:
            continue
        
        layer_idx = int(numbers[0])
        attention_output = output_dict.get('output')
        
        if not isinstance(attention_output, list) or len(attention_output) < 2:
            continue
        
        # Get attention weights: [batch, heads, seq_len, seq_len]
        attention_weights = torch.tensor(attention_output[1])
        
        # Process each head
        num_heads = attention_weights.shape[1]
        seq_len = attention_weights.shape[2]
        
        if seq_len < config.min_seq_len:
            continue
        
        for head_idx in range(num_heads):
            # Extract attention matrix for this head: [seq_len, seq_len]
            head_attention = attention_weights[0, head_idx, :, :]
            
            # Categorize this head
            head_info = categorize_attention_head(head_attention, layer_idx, head_idx, config)
            
            # Add to appropriate category list
            category = head_info['category']
            categorized[category].append(head_info)
    
    return categorized


def format_categorization_summary(categorized_heads: Dict[str, List[Dict[str, Any]]]) -> str:
    """
    Format categorization results as a human-readable summary.
    
    Args:
        categorized_heads: Output from categorize_all_heads
    
    Returns:
        Formatted string summary
    """
    category_names = {
        'previous_token': 'Previous-Token Heads',
        'first_token': 'First/Positional-Token Heads',
        'bow': 'Bag-of-Words Heads',
        'syntactic': 'Syntactic Heads',
        'other': 'Other Heads'
    }
    
    summary = []
    total_heads = sum(len(heads) for heads in categorized_heads.values())
    
    summary.append(f"Total Heads: {total_heads}\n")
    summary.append("=" * 60)
    
    for category, display_name in category_names.items():
        heads = categorized_heads.get(category, [])
        summary.append(f"\n{display_name}: {len(heads)} heads")
        
        if heads:
            # Group by layer
            heads_by_layer = {}
            for head_info in heads:
                layer = head_info['layer']
                if layer not in heads_by_layer:
                    heads_by_layer[layer] = []
                heads_by_layer[layer].append(head_info['head'])
            
            # Format by layer
            for layer in sorted(heads_by_layer.keys()):
                head_indices = sorted(heads_by_layer[layer])
                summary.append(f"  Layer {layer}: Heads {head_indices}")
    
    return "\n".join(summary)

