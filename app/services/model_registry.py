"""
Model registry for managing transformer models and tokenizers on CPU.
Provides caching and reuse of loaded models across sessions.
"""

import logging
from typing import Dict, Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..constants import SUPPORTED_MODELS, DEVICE

# Global registry for caching models
_model_cache: Dict[str, Tuple[AutoModelForCausalLM, AutoTokenizer]] = {}

logger = logging.getLogger(__name__)

def get_model_and_tokenizer(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load or reuse a CPU model and tokenizer for the given model name.
    
    Args:
        model_name: Model name from SUPPORTED_MODELS
        
    Returns:
        Tuple of (model, tokenizer)
        
    Raises:
        ValueError: If model_name is not supported
        RuntimeError: If model loading fails
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Model '{model_name}' not supported. Choose from: {SUPPORTED_MODELS}")
    
    # Return cached model if available
    if model_name in _model_cache:
        logger.info(f"Reusing cached model: {model_name}")
        return _model_cache[model_name]
    
    try:
        logger.info(f"Loading model: {model_name} on {DEVICE}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model on CPU with eager attention for compatibility
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            attn_implementation='eager',
            dtype=torch.float32,  # Use float32 for CPU
            device_map=None  # Don't use device_map on CPU
        )
        model.eval()
        model.to(DEVICE)
        
        # Cache the loaded model and tokenizer
        _model_cache[model_name] = (model, tokenizer)
        logger.info(f"Successfully loaded and cached model: {model_name}")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise RuntimeError(f"Model loading failed: {e}")

def get_cached_models() -> Dict[str, Tuple[AutoModelForCausalLM, AutoTokenizer]]:
    """
    Get all currently cached models.
    
    Returns:
        Dictionary mapping model names to (model, tokenizer) tuples
    """
    return _model_cache.copy()

def clear_model_cache(model_name: Optional[str] = None) -> None:
    """
    Clear cached models to free memory.
    
    Args:
        model_name: If provided, clear only this model. Otherwise clear all.
    """
    global _model_cache
    
    if model_name is not None:
        if model_name in _model_cache:
            del _model_cache[model_name]
            logger.info(f"Cleared cached model: {model_name}")
    else:
        _model_cache.clear()
        logger.info("Cleared all cached models")

def is_model_cached(model_name: str) -> bool:
    """
    Check if a model is currently cached.
    
    Args:
        model_name: Model name to check
        
    Returns:
        True if model is cached, False otherwise
    """
    return model_name in _model_cache

def get_model_info(model_name: str) -> Dict[str, any]:
    """
    Get information about a model (cached or not).
    
    Args:
        model_name: Model name to get info for
        
    Returns:
        Dictionary with model information
    """
    info = {
        "name": model_name,
        "supported": model_name in SUPPORTED_MODELS,
        "cached": is_model_cached(model_name),
        "device": DEVICE
    }
    
    if is_model_cached(model_name):
        model, tokenizer = _model_cache[model_name]
        info.update({
            "vocab_size": len(tokenizer),
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "num_layers": getattr(model.config, 'num_hidden_layers', 'unknown')
        })
    
    return info
