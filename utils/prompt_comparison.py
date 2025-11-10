"""
Two-Prompt Difference Analysis

Computes and highlights differences between two prompts across:
- Attention patterns at each layer/head
- Output token probabilities at each layer
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import re


class ComparisonConfig:
    """
    Configuration for two-prompt comparison thresholds.
    
    These thresholds determine when two prompts are considered "different enough"
    to highlight for educational purposes. Tuned to catch meaningful differences
    without being overly sensitive to minor variations.
    """
    
    def __init__(self):
        # Attention difference thresholds
        # Cosine distance of 0.15 means ~85% similarity (meaningful difference)
        self.attention_cosine_threshold = 0.15  # Minimum cosine distance for "different"
        # Normalized L2 distance of 0.8 represents substantial difference
        self.attention_l2_threshold = 0.8  # Minimum normalized L2 distance
        
        # Output probability difference thresholds
        # 8% probability difference is pedagogically meaningful
        self.output_prob_threshold = 0.08  # Minimum probability difference
        
        # Top-N for summary
        self.top_n_divergent = 5  # Show top-5 most divergent layers/heads


def cosine_distance(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """
    Compute cosine distance (1 - cosine similarity) between two tensors.
    
    Args:
        tensor1, tensor2: Tensors of same shape
    
    Returns:
        Cosine distance (0 = identical, 2 = opposite)
    """
    # Flatten tensors
    vec1 = tensor1.flatten()
    vec2 = tensor2.flatten()
    
    # Compute cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0), dim=1)
    
    # Convert to distance
    cos_dist = 1.0 - cos_sim.item()
    
    return cos_dist


def normalized_l2_distance(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """
    Compute normalized L2 distance between two tensors.
    
    Args:
        tensor1, tensor2: Tensors of same shape
    
    Returns:
        Normalized L2 distance
    """
    # Flatten tensors
    vec1 = tensor1.flatten()
    vec2 = tensor2.flatten()
    
    # Compute L2 distance
    l2_dist = torch.norm(vec1 - vec2, p=2).item()
    
    # Normalize by vector length
    norm_factor = max(torch.norm(vec1, p=2).item(), torch.norm(vec2, p=2).item(), 1e-8)
    normalized_dist = l2_dist / norm_factor
    
    return normalized_dist


def compare_attention_layers(activation_data1: Dict[str, Any], 
                             activation_data2: Dict[str, Any],
                             config: Optional[ComparisonConfig] = None) -> Dict[str, Any]:
    """
    Compare attention patterns between two prompts across all layers and heads.
    
    Args:
        activation_data1: Output from execute_forward_pass for first prompt
        activation_data2: Output from execute_forward_pass for second prompt
        config: Configuration object (uses defaults if None)
    
    Returns:
        Dictionary with comparison results:
        {
            'layer_differences': [...],  # List of dicts with layer-level differences
            'head_differences': [...],   # List of dicts with head-level differences
            'divergent_layers': [...],   # Layers exceeding thresholds
            'divergent_heads': [...]     # Heads exceeding thresholds
        }
    """
    if config is None:
        config = ComparisonConfig()
    
    attention1 = activation_data1.get('attention_outputs', {})
    attention2 = activation_data2.get('attention_outputs', {})
    
    if not attention1 or not attention2:
        return {
            'layer_differences': [],
            'head_differences': [],
            'divergent_layers': [],
            'divergent_heads': []
        }
    
    layer_differences = []
    head_differences = []
    divergent_layers = []
    divergent_heads = []
    
    # Find common layers between two prompts
    modules1 = set(attention1.keys())
    modules2 = set(attention2.keys())
    common_modules = modules1.intersection(modules2)
    
    for module_name in sorted(common_modules):
        # Extract layer number
        numbers = re.findall(r'\d+', module_name)
        if not numbers:
            continue
        
        layer_idx = int(numbers[0])
        
        # Get attention weights for both prompts
        attn_output1 = attention1[module_name]['output']
        attn_output2 = attention2[module_name]['output']
        
        if not isinstance(attn_output1, list) or not isinstance(attn_output2, list):
            continue
        if len(attn_output1) < 2 or len(attn_output2) < 2:
            continue
        
        attn_weights1 = torch.tensor(attn_output1[1])  # [batch, heads, seq, seq]
        attn_weights2 = torch.tensor(attn_output2[1])
        
        # Ensure same shape (may differ if prompts have different lengths)
        if attn_weights1.shape != attn_weights2.shape:
            # Pad or truncate to match sequence length
            min_seq = min(attn_weights1.shape[2], attn_weights2.shape[2])
            attn_weights1 = attn_weights1[:, :, :min_seq, :min_seq]
            attn_weights2 = attn_weights2[:, :, :min_seq, :min_seq]
        
        num_heads = attn_weights1.shape[1]
        
        # Compute layer-level difference (average across all heads)
        layer_cos_dist = cosine_distance(attn_weights1, attn_weights2)
        layer_l2_dist = normalized_l2_distance(attn_weights1, attn_weights2)
        
        layer_diff = {
            'layer': layer_idx,
            'module_name': module_name,
            'cosine_distance': layer_cos_dist,
            'l2_distance': layer_l2_dist
        }
        layer_differences.append(layer_diff)
        
        # Check if layer is divergent
        if (layer_cos_dist >= config.attention_cosine_threshold or 
            layer_l2_dist >= config.attention_l2_threshold):
            divergent_layers.append(layer_diff)
        
        # Compute head-level differences
        for head_idx in range(num_heads):
            head_attn1 = attn_weights1[0, head_idx, :, :]
            head_attn2 = attn_weights2[0, head_idx, :, :]
            
            head_cos_dist = cosine_distance(head_attn1, head_attn2)
            head_l2_dist = normalized_l2_distance(head_attn1, head_attn2)
            
            head_diff = {
                'layer': layer_idx,
                'head': head_idx,
                'label': f"L{layer_idx}-H{head_idx}",
                'cosine_distance': head_cos_dist,
                'l2_distance': head_l2_dist
            }
            head_differences.append(head_diff)
            
            # Check if head is divergent
            if (head_cos_dist >= config.attention_cosine_threshold or 
                head_l2_dist >= config.attention_l2_threshold):
                divergent_heads.append(head_diff)
    
    # Sort divergent items by cosine distance (descending)
    divergent_layers.sort(key=lambda x: x['cosine_distance'], reverse=True)
    divergent_heads.sort(key=lambda x: x['cosine_distance'], reverse=True)
    
    return {
        'layer_differences': layer_differences,
        'head_differences': head_differences,
        'divergent_layers': divergent_layers,
        'divergent_heads': divergent_heads
    }


def compare_output_probabilities(activation_data1: Dict[str, Any],
                                 activation_data2: Dict[str, Any],
                                 model, tokenizer,
                                 config: Optional[ComparisonConfig] = None) -> List[Dict[str, Any]]:
    """
    Compare output token probabilities at each layer between two prompts.
    
    Uses logit lens to get top tokens at each layer and compares their probabilities.
    
    Args:
        activation_data1: Output from execute_forward_pass for first prompt
        activation_data2: Output from execute_forward_pass for second prompt
        model: Transformer model (for logit lens)
        tokenizer: Tokenizer
        config: Configuration object (uses defaults if None)
    
    Returns:
        List of dictionaries with probability differences at each layer
    """
    if config is None:
        config = ComparisonConfig()
    
    from .model_patterns import _get_top_tokens
    
    block_outputs1 = activation_data1.get('block_outputs', {})
    block_outputs2 = activation_data2.get('block_outputs', {})
    
    if not block_outputs1 or not block_outputs2:
        return []
    
    # Find common layers
    modules1 = set(block_outputs1.keys())
    modules2 = set(block_outputs2.keys())
    common_modules = modules1.intersection(modules2)
    
    prob_differences = []
    
    for module_name in sorted(common_modules):
        # Extract layer number
        numbers = re.findall(r'\d+', module_name)
        if not numbers:
            continue
        
        layer_idx = int(numbers[0])
        
        # Get top tokens for both prompts
        top_tokens1 = _get_top_tokens(activation_data1, module_name, model, tokenizer)
        top_tokens2 = _get_top_tokens(activation_data2, module_name, model, tokenizer)
        
        if not top_tokens1 or not top_tokens2:
            continue
        
        # Compare top predicted tokens
        top_token1, top_prob1 = top_tokens1[0]
        top_token2, top_prob2 = top_tokens2[0]
        
        # Compute probability difference
        prob_diff = abs(top_prob1 - top_prob2)
        tokens_match = (top_token1 == top_token2)
        
        prob_differences.append({
            'layer': layer_idx,
            'module_name': module_name,
            'prompt1_token': top_token1,
            'prompt1_prob': top_prob1,
            'prompt2_token': top_token2,
            'prompt2_prob': top_prob2,
            'tokens_match': tokens_match,
            'prob_difference': prob_diff
        })
    
    return prob_differences


def format_comparison_summary(comparison_results: Dict[str, Any], 
                              prob_differences: List[Dict[str, Any]],
                              config: Optional[ComparisonConfig] = None) -> str:
    """
    Format comparison results as a human-readable summary.
    
    Args:
        comparison_results: Output from compare_attention_layers
        prob_differences: Output from compare_output_probabilities
        config: Configuration object (uses defaults if None)
    
    Returns:
        Formatted string summary
    """
    if config is None:
        config = ComparisonConfig()
    
    summary = []
    
    # Summary header
    summary.append("Two-Prompt Comparison Analysis")
    summary.append("=" * 60)
    
    # Divergent layers
    divergent_layers = comparison_results['divergent_layers']
    summary.append(f"\nDivergent Layers: {len(divergent_layers)}")
    
    if divergent_layers:
        top_n = divergent_layers[:config.top_n_divergent]
        summary.append(f"  Top {len(top_n)} most divergent:")
        for layer_diff in top_n:
            summary.append(f"    Layer {layer_diff['layer']}: "
                         f"cosine_dist={layer_diff['cosine_distance']:.3f}, "
                         f"l2_dist={layer_diff['l2_distance']:.3f}")
    
    # Divergent heads
    divergent_heads = comparison_results['divergent_heads']
    summary.append(f"\nDivergent Heads: {len(divergent_heads)}")
    
    if divergent_heads:
        top_n = divergent_heads[:config.top_n_divergent]
        summary.append(f"  Top {len(top_n)} most divergent:")
        for head_diff in top_n:
            summary.append(f"    {head_diff['label']}: "
                         f"cosine_dist={head_diff['cosine_distance']:.3f}, "
                         f"l2_dist={head_diff['l2_distance']:.3f}")
    
    # Output probability differences
    if prob_differences:
        # Find layers with different top tokens
        diff_tokens = [p for p in prob_differences if not p['tokens_match']]
        summary.append(f"\nLayers with Different Top Tokens: {len(diff_tokens)}/{len(prob_differences)}")
        
        if diff_tokens:
            summary.append("  Examples:")
            for prob_diff in diff_tokens[:config.top_n_divergent]:
                summary.append(f"    Layer {prob_diff['layer']}: "
                             f"'{prob_diff['prompt1_token']}' ({prob_diff['prompt1_prob']:.3f}) vs "
                             f"'{prob_diff['prompt2_token']}' ({prob_diff['prompt2_prob']:.3f})")
    
    return "\n".join(summary)

