"""
Module and parameter discovery service.
Adapts the logic from module_selector.py for use in the dashboard.
"""

import re
import logging
from typing import Dict, List, Tuple

from ..constants import ERROR_MESSAGES, SUCCESS_MESSAGES

logger = logging.getLogger(__name__)

def extract_generalized_pattern(module_name: str) -> str:
    """
    Extract pattern from module name by replacing any numeric sequences with placeholders.
    Examples:
    - model.layers.0.self_attn -> model.layers.{N}.self_attn  
    - transformer.h.0.attn -> transformer.h.{N}.attn
    - model.decoder.layers.5.mlp -> model.decoder.layers.{N}.mlp
    """
    # Replace any sequence of digits that appears between dots or at word boundaries
    # This catches patterns like .0., .12., _0_, etc.
    pattern = re.sub(r'(\.|_)(\d+)(\.|_|$)', r'\1{N}\3', module_name)
    # Also catch cases where numbers appear directly after letters
    pattern = re.sub(r'([a-zA-Z])(\d+)(\.|_|$)', r'\1{N}\3', pattern)
    return pattern

def detect_numeric_patterns(model) -> Dict[str, List[str]]:
    """
    Detect repeating numeric patterns in module names and group them.
    
    Args:
        model: Loaded transformer model
        
    Returns:
        Dictionary mapping generalized patterns to lists of concrete module names
    """
    pattern_to_modules: Dict[str, List[str]] = {}
    
    for name, _ in model.named_modules():
        if not name:
            continue
            
        # Extract pattern by replacing numeric patterns with placeholders
        pattern = extract_generalized_pattern(name)
        
        if pattern not in pattern_to_modules:
            pattern_to_modules[pattern] = []
        pattern_to_modules[pattern].append(name)
    
    logger.info(f"Detected {len(pattern_to_modules)} unique module patterns")
    return pattern_to_modules

def categorize_module_patterns(pattern_to_modules: Dict[str, List[str]]) -> Tuple[List[str], List[str], List[str]]:
    """
    Categorize module patterns into attention, MLP, and other based on naming patterns.
    
    Args:
        pattern_to_modules: Dictionary from detect_numeric_patterns
        
    Returns:
        Tuple of (attn_patterns, mlp_patterns, other_patterns)
    """
    attn_patterns: List[str] = []
    mlp_patterns: List[str] = []
    other_patterns: List[str] = []
    
    for pattern in pattern_to_modules.keys():
        lower = pattern.lower()
        
        if ("attn" in lower) or ("attention" in lower):
            attn_patterns.append(pattern)
        elif "mlp" in lower:
            mlp_patterns.append(pattern)
        else:
            other_patterns.append(pattern)
    
    # Sort for consistent ordering
    attn_patterns.sort()
    mlp_patterns.sort()
    other_patterns.sort()
    
    logger.info(f"Categorized patterns: {len(attn_patterns)} attention, {len(mlp_patterns)} MLP, {len(other_patterns)} other")
    return attn_patterns, mlp_patterns, other_patterns

def detect_parameter_names(model) -> Dict[str, List[str]]:
    """
    Detect parameter patterns and group them by generalized pattern.
    
    Args:
        model: Loaded transformer model
        
    Returns:
        Dictionary mapping generalized patterns to parameter names
    """
    pattern_to_parameters: Dict[str, List[str]] = {}
    
    for name, _ in model.named_parameters():
        # Extract pattern by replacing numeric patterns with placeholders
        pattern = extract_generalized_pattern(name)
        
        if pattern not in pattern_to_parameters:
            pattern_to_parameters[pattern] = []
        pattern_to_parameters[pattern].append(name)
    
    logger.info(f"Detected {len(pattern_to_parameters)} unique parameter patterns")
    return pattern_to_parameters

def categorize_parameter_patterns(pattern_to_parameters: Dict[str, List[str]]) -> Tuple[List[str], List[str], List[str]]:
    """
    Categorize parameter patterns into logit lens, normalization, and other.
    
    Args:
        pattern_to_parameters: Dictionary from detect_parameter_names
        
    Returns:
        Tuple of (logit_lens_patterns, norm_patterns, other_patterns)
    """
    logit_lens_patterns: List[str] = []
    norm_patterns: List[str] = []
    other_patterns: List[str] = []
    
    for pattern in pattern_to_parameters.keys():
        lower = pattern.lower()
        
        if any(x in lower for x in ['lm_head', 'head', 'classifier', 'embed', 'wte', 'word']):
            logit_lens_patterns.append(pattern)
        elif any(x in lower for x in ['norm', 'layernorm', 'layer_norm']):
            norm_patterns.append(pattern)
        else:
            other_patterns.append(pattern)
    
    # Sort for consistent ordering
    logit_lens_patterns.sort()
    norm_patterns.sort()
    other_patterns.sort()
    
    logger.info(f"Categorized parameters: {len(logit_lens_patterns)} logit-lens, {len(norm_patterns)} norm, {len(other_patterns)} other")
    return logit_lens_patterns, norm_patterns, other_patterns

def flatten_parameter_names(patterns: List[str], pattern_to_parameters: Dict[str, List[str]]) -> List[str]:
    """
    Flatten parameter patterns to get fully-qualified parameter names.
    
    Args:
        patterns: List of patterns to flatten
        pattern_to_parameters: Pattern to parameter mapping
        
    Returns:
        Sorted list of parameter names
    """
    names = []
    for pattern in patterns:
        if pattern in pattern_to_parameters:
            names.extend(pattern_to_parameters[pattern])
    return sorted(names)

def build_dropdown_options(
    attn_patterns: List[str], 
    mlp_patterns: List[str], 
    other_patterns: List[str],
    pattern_to_parameters: Dict[str, List[str]],
    logit_lens_patterns: List[str],
    norm_patterns: List[str],
    other_param_patterns: List[str]
) -> Dict[str, List[str]]:
    """
    Build UI-ready dropdown options for the sidebar.
    
    Args:
        attn_patterns: Attention module patterns
        mlp_patterns: MLP module patterns
        other_patterns: Other module patterns
        pattern_to_parameters: Parameter pattern mapping
        logit_lens_patterns: Logit lens parameter patterns
        norm_patterns: Normalization parameter patterns
        other_param_patterns: Other parameter patterns
        
    Returns:
        Dictionary with dropdown options for each component
    """
    # Module patterns: attention first, then others
    attention_options = attn_patterns + other_patterns
    mlp_options = mlp_patterns + other_patterns
    
    # Parameter names: flatten patterns to get actual parameter names
    norm_param_options = (
        flatten_parameter_names(norm_patterns, pattern_to_parameters) +
        flatten_parameter_names(other_param_patterns, pattern_to_parameters)
    )
    
    logit_param_options = (
        flatten_parameter_names(logit_lens_patterns, pattern_to_parameters) +
        flatten_parameter_names(other_param_patterns, pattern_to_parameters)
    )
    
    options = {
        "attention_pattern_options": attention_options,
        "mlp_pattern_options": mlp_options,
        "norm_param_options": norm_param_options,
        "logit_param_options": logit_param_options
    }
    
    # Log warnings for empty categories
    for key, values in options.items():
        if not values:
            logger.warning(f"No options found for {key}")
    
    logger.info(f"Built dropdown options: {sum(len(v) for v in options.values())} total options")
    return options

def discover_model_structure(model) -> Dict[str, any]:
    """
    Discover the complete module and parameter structure of a model.
    
    Args:
        model: Loaded transformer model
        
    Returns:
        Dictionary containing all discovered patterns and dropdown options
    """
    try:
        # Discover module patterns
        pattern_to_modules = detect_numeric_patterns(model)
        attn_patterns, mlp_patterns, other_patterns = categorize_module_patterns(pattern_to_modules)
        
        # Discover parameter patterns
        pattern_to_parameters = detect_parameter_names(model)
        logit_lens_patterns, norm_patterns, other_param_patterns = categorize_parameter_patterns(pattern_to_parameters)
        
        # Build dropdown options
        dropdown_options = build_dropdown_options(
            attn_patterns, mlp_patterns, other_patterns,
            pattern_to_parameters,
            logit_lens_patterns, norm_patterns, other_param_patterns
        )
        
        # Return complete structure
        result = {
            "success": True,
            "pattern_to_modules": pattern_to_modules,
            "pattern_to_parameters": pattern_to_parameters,
            "module_patterns": {
                "attention": attn_patterns,
                "mlp": mlp_patterns,
                "other": other_patterns
            },
            "parameter_patterns": {
                "logit_lens": logit_lens_patterns,
                "norm": norm_patterns,
                "other": other_param_patterns
            },
            "dropdown_options": dropdown_options,
            "message": SUCCESS_MESSAGES["modules_found"].format(
                count=len(pattern_to_modules) + len(pattern_to_parameters)
            )
        }
        
        logger.info("Successfully discovered model structure")
        return result
        
    except Exception as e:
        logger.error(f"Failed to discover model structure: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": ERROR_MESSAGES["no_modules_found"]
        }
