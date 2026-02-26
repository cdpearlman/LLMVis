"""
Attention Head Detection and Categorization

Loads pre-computed head category data from JSON (produced by scripts/analyze_heads.py)
and performs lightweight runtime verification of head activation on the current input.

Categories:
- Previous Token: attends to the immediately preceding token
- Induction: completes repeated patterns ([A][B]...[A] → [B])
- Duplicate Token: attends to earlier occurrences of the same token
- Positional / First-Token: attends to the first token or positional patterns
- Diffuse / Spread: high-entropy, evenly distributed attention
- Other: heads that don't fit the above categories
"""

import json
import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import re
from pathlib import Path


# Path to the pre-computed head categories JSON
_JSON_PATH = Path(__file__).parent / "head_categories.json"

# Cache for loaded JSON data (avoids re-reading per request)
_category_cache: Dict[str, Any] = {}


def load_head_categories(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Load pre-computed head category data for a model.
    
    Args:
        model_name: HuggingFace model name (e.g., "gpt2", "EleutherAI/pythia-70m")
    
    Returns:
        Dict with model's category data, or None if model not analyzed.
        Structure: {
            "model_name": str,
            "num_layers": int,
            "num_heads": int,
            "categories": { category_name: { "top_heads": [...], ... } },
            ...
        }
    """
    global _category_cache
    
    # Check cache first
    if model_name in _category_cache:
        return _category_cache[model_name]
    
    # Load JSON
    if not _JSON_PATH.exists():
        return None
    
    try:
        with open(_JSON_PATH, 'r') as f:
            all_data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None
    
    # Try exact match first, then common aliases
    model_data = all_data.get(model_name)
    if model_data is None:
        # Try short name (e.g., "gpt2" for "openai-community/gpt2")
        short_name = model_name.split('/')[-1] if '/' in model_name else model_name
        model_data = all_data.get(short_name)
    
    if model_data is not None:
        _category_cache[model_name] = model_data
    
    return model_data


def clear_category_cache():
    """Clear the loaded category cache (useful for testing)."""
    global _category_cache
    _category_cache = {}


def _compute_attention_entropy(attention_weights: torch.Tensor) -> float:
    """
    Compute normalized entropy of an attention distribution.
    
    Args:
        attention_weights: [seq_len] tensor of attention weights for one position
    
    Returns:
        Normalized entropy (0.0 to 1.0). 1.0 = perfectly uniform, 0.0 = fully peaked.
    """
    epsilon = 1e-10
    weights = attention_weights + epsilon
    entropy = -torch.sum(weights * torch.log(weights))
    max_entropy = np.log(len(weights))
    return (entropy / max_entropy).item() if max_entropy > 0 else 0.0


def _find_repeated_tokens(token_ids: List[int]) -> Dict[int, List[int]]:
    """
    Find tokens that appear more than once and their positions.
    
    Args:
        token_ids: List of token IDs in the sequence
    
    Returns:
        Dict mapping token_id -> list of positions where it appears (only for repeated tokens)
    """
    positions: Dict[int, List[int]] = {}
    for i, tid in enumerate(token_ids):
        if tid not in positions:
            positions[tid] = []
        positions[tid].append(i)
    
    # Keep only tokens that appear more than once
    return {tid: pos_list for tid, pos_list in positions.items() if len(pos_list) > 1}


def verify_head_activation(
    attn_matrix: torch.Tensor,
    token_ids: List[int],
    category: str
) -> float:
    """
    Verify whether a head's known role is active on the current input.
    
    Args:
        attn_matrix: [seq_len, seq_len] attention weights for this head
        token_ids: List of token IDs in the input
        category: Category name (previous_token, induction, duplicate_token, positional, diffuse)
    
    Returns:
        Activation score (0.0 to 1.0). 0.0 means the role is not triggered on this input.
    """
    seq_len = attn_matrix.shape[0]
    
    if seq_len < 2:
        return 0.0
    
    if category == "previous_token":
        # Mean of diagonal-1 values: how much each position attends to the previous position
        prev_token_attentions = []
        for i in range(1, seq_len):
            prev_token_attentions.append(attn_matrix[i, i - 1].item())
        return float(np.mean(prev_token_attentions)) if prev_token_attentions else 0.0
    
    elif category == "induction":
        # Induction pattern: [A][B]...[A] → attend to [B]
        # For each repeated token at position i where token[i]==token[j] (j < i),
        # check if position i attends to position j+1
        repeated = _find_repeated_tokens(token_ids)
        if not repeated:
            return 0.0  # No repetition → gray out
        
        induction_scores = []
        for tid, positions in repeated.items():
            for k in range(1, len(positions)):
                current_pos = positions[k]  # Later occurrence
                for prev_idx in range(k):
                    prev_pos = positions[prev_idx]  # Earlier occurrence
                    target_pos = prev_pos + 1  # The token AFTER the earlier occurrence
                    if target_pos < seq_len and current_pos < seq_len:
                        induction_scores.append(attn_matrix[current_pos, target_pos].item())
        
        return float(np.mean(induction_scores)) if induction_scores else 0.0
    
    elif category == "duplicate_token":
        # Check if later occurrences attend to earlier occurrences of the same token
        repeated = _find_repeated_tokens(token_ids)
        if not repeated:
            return 0.0  # No duplicates → gray out
        
        dup_scores = []
        for tid, positions in repeated.items():
            for k in range(1, len(positions)):
                later_pos = positions[k]
                # Sum attention to all earlier occurrences
                earlier_attention = sum(
                    attn_matrix[later_pos, positions[j]].item()
                    for j in range(k)
                )
                dup_scores.append(earlier_attention)
        
        return float(np.mean(dup_scores)) if dup_scores else 0.0
    
    elif category == "positional":
        # Mean of column-0 attention (how much each position attends to the first token)
        first_token_attention = attn_matrix[:, 0].mean().item()
        return first_token_attention
    
    elif category == "diffuse":
        # Average normalized entropy across all positions
        entropies = []
        for i in range(seq_len):
            entropies.append(_compute_attention_entropy(attn_matrix[i]))
        return float(np.mean(entropies)) if entropies else 0.0
    
    else:
        return 0.0


def get_active_head_summary(
    activation_data: Dict[str, Any],
    model_name: str
) -> Optional[Dict[str, Any]]:
    """
    Main entry point: load categories from JSON, verify each head on the current input,
    and return a UI-ready structure.
    
    Args:
        activation_data: Output from execute_forward_pass with attention data
        model_name: HuggingFace model name
    
    Returns:
        Dict with structure:
        {
            "model_available": True,
            "categories": {
                "previous_token": {
                    "display_name": str,
                    "description": str,
                    "educational_text": str,
                    "icon": str,
                    "requires_repetition": bool,
                    "suggested_prompt": str or None,
                    "is_applicable": bool,  # False if requires_repetition but no repeats
                    "heads": [
                        {"layer": int, "head": int, "offline_score": float,
                         "activation_score": float, "is_active": bool, "label": str}
                    ]
                },
                ...
            }
        }
        Returns None if model not in JSON.
    """
    model_data = load_head_categories(model_name)
    if model_data is None:
        return None
    
    # Extract attention weights and token IDs from activation data
    attention_outputs = activation_data.get('attention_outputs', {})
    input_ids = activation_data.get('input_ids', [[]])[0]
    
    if not attention_outputs or not input_ids:
        return None
    
    # Build a lookup: (layer, head) → attention_matrix [seq_len, seq_len]
    head_attention_lookup: Dict[Tuple[int, int], torch.Tensor] = {}
    
    for module_name, output_dict in attention_outputs.items():
        numbers = re.findall(r'\d+', module_name)
        if not numbers:
            continue
        
        layer_idx = int(numbers[0])
        attention_output = output_dict.get('output')
        
        if not isinstance(attention_output, list) or len(attention_output) < 2:
            continue
        
        # attention_output[1] is [batch, heads, seq_len, seq_len]
        attention_weights = torch.tensor(attention_output[1])
        num_heads = attention_weights.shape[1]
        
        for head_idx in range(num_heads):
            head_attention_lookup[(layer_idx, head_idx)] = attention_weights[0, head_idx, :, :]
    
    # Check if input has repeated tokens (needed for applicability check)
    repeated_tokens = _find_repeated_tokens(input_ids)
    has_repetition = len(repeated_tokens) > 0
    
    # Build result
    result = {
        "model_available": True,
        "categories": {}
    }
    
    categories = model_data.get("categories", {})
    
    # Define category order for consistent display
    category_order = ["previous_token", "induction", "duplicate_token", "positional", "diffuse"]
    
    for cat_key in category_order:
        cat_info = categories.get(cat_key)
        if cat_info is None:
            continue
        
        requires_repetition = cat_info.get("requires_repetition", False)
        is_applicable = not requires_repetition or has_repetition
        
        heads_result = []
        for head_entry in cat_info.get("top_heads", []):
            layer = head_entry["layer"]
            head = head_entry["head"]
            offline_score = head_entry["score"]
            
            # Get activation score on current input
            attn_matrix = head_attention_lookup.get((layer, head))
            if attn_matrix is not None and is_applicable:
                activation_score = verify_head_activation(attn_matrix, input_ids, cat_key)
            else:
                activation_score = 0.0
            
            # A head is "active" if its activation score exceeds a minimum threshold
            is_active = activation_score > 0.1 and is_applicable
            
            heads_result.append({
                "layer": layer,
                "head": head,
                "offline_score": offline_score,
                "activation_score": round(activation_score, 3),
                "is_active": is_active,
                "label": f"L{layer}-H{head}"
            })
        
        result["categories"][cat_key] = {
            "display_name": cat_info.get("display_name", cat_key),
            "description": cat_info.get("description", ""),
            "educational_text": cat_info.get("educational_text", ""),
            "icon": cat_info.get("icon", "circle"),
            "requires_repetition": requires_repetition,
            "suggested_prompt": cat_info.get("suggested_prompt"),
            "is_applicable": is_applicable,
            "heads": heads_result
        }
    
    # Add "Other" category (heads not claimed by any top list)
    result["categories"]["other"] = {
        "display_name": "Other / Unclassified",
        "description": "Heads whose patterns don't fit the simple categories above",
        "educational_text": "This head's pattern doesn't fit our simple categories — it may be doing something more complex or context-dependent.",
        "icon": "question-circle",
        "requires_repetition": False,
        "suggested_prompt": None,
        "is_applicable": True,
        "heads": []  # We don't enumerate all "other" heads to keep the UI clean
    }
    
    return result
