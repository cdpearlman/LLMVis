"""
Model family configuration registry.

Maps specific model names to families, and families to canonical module/parameter patterns.
This allows automatic selection of appropriate modules and parameters based on model architecture.
"""

from typing import Dict, List, Optional, Any

# Model family specifications
MODEL_FAMILIES: Dict[str, Dict[str, Any]] = {
    # LLaMA-like models (LLaMA, Mistral, Qwen2)
    "llama_like": {
        "description": "LLaMA, Mistral, Qwen2 architecture",
        "templates": {
            "attention_pattern": "model.layers.{N}.self_attn",
            "mlp_pattern": "model.layers.{N}.mlp",
            "block_pattern": "model.layers.{N}",
        },
        "norm_parameter": "model.norm.weight",
        "logit_lens_pattern": "lm_head.weight",
        "norm_type": "rmsnorm",
    },
    
    # GPT-2 family
    "gpt2": {
        "description": "GPT-2 architecture",
        "templates": {
            "attention_pattern": "transformer.h.{N}.attn",
            "mlp_pattern": "transformer.h.{N}.mlp",
            "block_pattern": "transformer.h.{N}",
        },
        "norm_parameter": "transformer.ln_f.weight",
        "logit_lens_pattern": "lm_head.weight",
        "norm_type": "layernorm",
    },
    
    # OPT
    "opt": {
        "description": "OPT architecture",
        "templates": {
            "attention_pattern": "model.decoder.layers.{N}.self_attn",
            "mlp_pattern": "model.decoder.layers.{N}.fc2",
            "block_pattern": "model.decoder.layers.{N}",
        },
        "norm_parameter": "model.decoder.final_layer_norm.weight",
        "logit_lens_pattern": "lm_head.weight",
        "norm_type": "layernorm",
    },
    
    # GPT-NeoX
    "gpt_neox": {
        "description": "GPT-NeoX architecture",
        "templates": {
            "attention_pattern": "gpt_neox.layers.{N}.attention",
            "mlp_pattern": "gpt_neox.layers.{N}.mlp",
            "block_pattern": "gpt_neox.layers.{N}",
        },
        "norm_parameter": "gpt_neox.final_layer_norm.weight",
        "logit_lens_pattern": "embed_out.weight",
        "norm_type": "layernorm",
    },
    
    # BLOOM
    "bloom": {
        "description": "BLOOM architecture",
        "templates": {
            "attention_pattern": "transformer.h.{N}.self_attention",
            "mlp_pattern": "transformer.h.{N}.mlp",
            "block_pattern": "transformer.h.{N}",
        },
        "norm_parameter": "transformer.ln_f.weight",
        "logit_lens_pattern": "lm_head.weight",
        "norm_type": "layernorm",
    },
    
    # Falcon
    "falcon": {
        "description": "Falcon architecture",
        "templates": {
            "attention_pattern": "transformer.h.{N}.self_attention",
            "mlp_pattern": "transformer.h.{N}.mlp",
            "block_pattern": "transformer.h.{N}",
        },
        "norm_parameter": "transformer.ln_f.weight",
        "logit_lens_pattern": "lm_head.weight",
        "norm_type": "layernorm",
    },
    
    # MPT
    "mpt": {
        "description": "MPT architecture",
        "templates": {
            "attention_pattern": "transformer.blocks.{N}.attn",
            "mlp_pattern": "transformer.blocks.{N}.ffn",
            "block_pattern": "transformer.blocks.{N}",
        },
        "norm_parameter": "transformer.norm_f.weight",
        "logit_lens_pattern": "lm_head.weight",
        "norm_type": "layernorm",
    },
}

# Hard-coded mapping of specific model names to families
MODEL_TO_FAMILY: Dict[str, str] = {
    # Qwen models
    "Qwen/Qwen2.5-0.5B": "llama_like",
    "Qwen/Qwen2.5-1.5B": "llama_like",
    "Qwen/Qwen2.5-3B": "llama_like",
    "Qwen/Qwen2.5-7B": "llama_like",
    "Qwen/Qwen2.5-14B": "llama_like",
    "Qwen/Qwen2.5-32B": "llama_like",
    "Qwen/Qwen2.5-72B": "llama_like",
    "Qwen/Qwen2-0.5B": "llama_like",
    "Qwen/Qwen2-1.5B": "llama_like",
    "Qwen/Qwen2-7B": "llama_like",
    
    # LLaMA models
    "meta-llama/Llama-2-7b-hf": "llama_like",
    "meta-llama/Llama-2-13b-hf": "llama_like",
    "meta-llama/Llama-2-70b-hf": "llama_like",
    "meta-llama/Llama-3.1-8B": "llama_like",
    "meta-llama/Llama-3.1-70B": "llama_like",
    "meta-llama/Llama-3.2-1B": "llama_like",
    "meta-llama/Llama-3.2-3B": "llama_like",
    
    # Mistral models
    "mistralai/Mistral-7B-v0.1": "llama_like",
    "mistralai/Mistral-7B-v0.3": "llama_like",
    "mistralai/Mixtral-8x7B-v0.1": "llama_like",
    "mistralai/Mixtral-8x22B-v0.1": "llama_like",
    
    # GPT-2 models
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2",
    "gpt2-large": "gpt2",
    "gpt2-xl": "gpt2",
    "openai-community/gpt2": "gpt2",
    "openai-community/gpt2-medium": "gpt2",
    "openai-community/gpt2-large": "gpt2",
    "openai-community/gpt2-xl": "gpt2",
    
    # OPT models
    "facebook/opt-125m": "opt",
    "facebook/opt-350m": "opt",
    "facebook/opt-1.3b": "opt",
    "facebook/opt-2.7b": "opt",
    "facebook/opt-6.7b": "opt",
    "facebook/opt-13b": "opt",
    "facebook/opt-30b": "opt",
    
    # GPT-NeoX models
    "EleutherAI/gpt-neox-20b": "gpt_neox",
    "EleutherAI/pythia-70m": "gpt_neox",
    "EleutherAI/pythia-160m": "gpt_neox",
    "EleutherAI/pythia-410m": "gpt_neox",
    "EleutherAI/pythia-1b": "gpt_neox",
    "EleutherAI/pythia-1.4b": "gpt_neox",
    "EleutherAI/pythia-2.8b": "gpt_neox",
    "EleutherAI/pythia-6.9b": "gpt_neox",
    "EleutherAI/pythia-12b": "gpt_neox",
    
    # BLOOM models
    "bigscience/bloom-560m": "bloom",
    "bigscience/bloom-1b1": "bloom",
    "bigscience/bloom-1b7": "bloom",
    "bigscience/bloom-3b": "bloom",
    "bigscience/bloom-7b1": "bloom",
    
    # Falcon models
    "tiiuae/falcon-7b": "falcon",
    "tiiuae/falcon-40b": "falcon",
    
    # MPT models
    "mosaicml/mpt-7b": "mpt",
    "mosaicml/mpt-30b": "mpt",
}


def get_model_family(model_name: str) -> Optional[str]:
    """
    Get the model family for a given model name.
    
    Args:
        model_name: HuggingFace model name/path
        
    Returns:
        Family name if found, None otherwise
    """
    return MODEL_TO_FAMILY.get(model_name)


def get_family_config(family_name: str) -> Optional[Dict[str, Any]]:
    """
    Get the configuration for a model family.
    
    Args:
        family_name: Name of the model family
        
    Returns:
        Family configuration dict if found, None otherwise
    """
    return MODEL_FAMILIES.get(family_name)


def get_auto_selections(model_name: str, module_patterns: Dict[str, List[str]], 
                        param_patterns: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Get automatic dropdown selections based on model family.
    
    Args:
        model_name: HuggingFace model name
        module_patterns: Available module patterns from the model
        param_patterns: Available parameter patterns from the model
        
    Returns:
        Dict with keys: attention_selection, block_selection, norm_selection, logit_lens_selection
        Each value is a list of pattern keys that should be pre-selected
    """
    family = get_model_family(model_name)
    if not family:
        return {
            'attention_selection': [],
            'block_selection': [],
            'norm_selection': [],  # Empty list for multi-select dropdown
            'logit_lens_selection': None,
            'family_name': None
        }
    
    config = get_family_config(family)
    if not config:
        return {
            'attention_selection': [],
            'block_selection': [],
            'norm_selection': [],  # Empty list for multi-select dropdown
            'logit_lens_selection': None,
            'family_name': None
        }
    
    # Find matching patterns in the available patterns
    attention_matches = []
    block_matches = []
    norm_match = None
    logit_lens_match = None
    
    # Match attention patterns
    attention_template = config['templates'].get('attention_pattern', '')
    for pattern_key in module_patterns.keys():
        if _pattern_matches_template(pattern_key, attention_template):
            attention_matches.append(pattern_key)
    
    # Match block patterns (full layer outputs - residual stream)
    block_template = config['templates'].get('block_pattern', '')
    for pattern_key in module_patterns.keys():
        if _pattern_matches_template(pattern_key, block_template):
            block_matches.append(pattern_key)
    
    # Match normalization parameter
    # Note: norm-params-dropdown has multi=True, so return a list
    norm_parameter = config.get('norm_parameter', '')
    if norm_parameter:
        for pattern_key in param_patterns.keys():
            if _pattern_matches_template(pattern_key, norm_parameter):
                norm_match = [pattern_key]  # Return as list for multi-select dropdown
                break
    
    # Match logit lens pattern - check both parameters AND modules
    logit_pattern = config.get('logit_lens_pattern', '')
    # First try parameters
    for pattern_key in param_patterns.keys():
        if _pattern_matches_template(pattern_key, logit_pattern):
            logit_lens_match = pattern_key
            break
    # If not found in parameters, try modules
    if not logit_lens_match:
        for pattern_key in module_patterns.keys():
            if _pattern_matches_template(pattern_key, logit_pattern):
                logit_lens_match = pattern_key
                break
    
    return {
        'attention_selection': attention_matches,
        'block_selection': block_matches,
        'norm_selection': norm_match if norm_match else [],  # Ensure list for multi-select
        'logit_lens_selection': logit_lens_match,
        'family_name': family,
        'family_description': config.get('description', '')
    }


def _pattern_matches_template(pattern: str, template: str) -> bool:
    """
    Check if a pattern string matches a template.
    Templates use {N} as wildcard, patterns use {N} for the same purpose.
    
    Args:
        pattern: Pattern string like "model.layers.{N}.mlp"
        template: Template string like "model.layers.{N}.mlp"
        
    Returns:
        True if pattern matches template
    """
    if not template:
        return False
    
    # Simple check: remove {N} from both and see if they match
    pattern_normalized = pattern.replace('{N}', '').replace('.', '_')
    template_normalized = template.replace('{N}', '').replace('.', '_')
    
    # Exact match
    if pattern_normalized == template_normalized:
        return True
    
    # For logit lens: also allow matching if pattern starts with template base
    # e.g., "lm_head" matches "lm_head.weight" or "lm_head.weight" matches "lm_head"
    template_base = template_normalized.split('_weight')[0].split('_bias')[0]
    pattern_base = pattern_normalized.split('_weight')[0].split('_bias')[0]
    
    return template_base == pattern_base
