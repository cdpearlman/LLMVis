"""
Data contract utilities for validating and normalizing activation data structures.
Ensures compatibility with the activation_capture.py schema.
"""

import logging
from typing import Dict, List, Any, Optional
import torch

logger = logging.getLogger(__name__)

def validate_activation_data(data: Dict[str, Any]) -> bool:
    """
    Validate that activation data conforms to expected schema.
    
    Args:
        data: Activation data dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = [
        "model", "prompt", "attention_modules", "attention_outputs",
        "mlp_modules", "mlp_outputs", "norm_parameter", "logit_lens_parameter"
    ]
    
    for field in required_fields:
        if field not in data:
            logger.error(f"Missing required field: {field}")
            return False
    
    # Validate types
    if not isinstance(data["attention_modules"], list):
        logger.error("attention_modules must be a list")
        return False
        
    if not isinstance(data["mlp_modules"], list):
        logger.error("mlp_modules must be a list")
        return False
        
    if not isinstance(data["attention_outputs"], dict):
        logger.error("attention_outputs must be a dict")
        return False
        
    if not isinstance(data["mlp_outputs"], dict):
        logger.error("mlp_outputs must be a dict")
        return False
        
    if not isinstance(data["norm_parameter"], list):
        logger.error("norm_parameter must be a list")
        return False
        
    # Validate that modules match outputs
    for module in data["attention_modules"]:
        if module not in data["attention_outputs"]:
            logger.error(f"Attention module {module} not found in outputs")
            return False
            
    for module in data["mlp_modules"]:
        if module not in data["mlp_outputs"]:
            logger.error(f"MLP module {module} not found in outputs")
            return False
    
    # Validate norm_parameter has exactly one element
    if len(data["norm_parameter"]) != 1:
        logger.error(f"norm_parameter must have exactly 1 element, got {len(data['norm_parameter'])}")
        return False
    
    logger.info("Activation data validation passed")
    return True

def extract_norm_weight(data: Dict[str, Any]) -> torch.Tensor:
    """
    Extract normalization weight from the norm_parameter field.
    
    Args:
        data: Activation data dictionary
        
    Returns:
        Normalization weight tensor
        
    Raises:
        ValueError: If norm_parameter is invalid
    """
    if "norm_parameter" not in data:
        raise ValueError("norm_parameter not found in data")
    
    norm_param = data["norm_parameter"]
    if not isinstance(norm_param, list) or len(norm_param) != 1:
        raise ValueError(f"norm_parameter must be a list with exactly 1 element, got {type(norm_param)} with length {len(norm_param) if isinstance(norm_param, list) else 'N/A'}")
    
    # Convert to tensor if needed
    norm_weight = norm_param[0]
    if not isinstance(norm_weight, torch.Tensor):
        norm_weight = torch.tensor(norm_weight)
    
    logger.info(f"Extracted norm weight with shape: {norm_weight.shape}")
    return norm_weight

def extract_mlp_outputs_as_tensors(data: Dict[str, Any]) -> List[torch.Tensor]:
    """
    Extract MLP outputs as a list of tensors, ordered by layer.
    
    Args:
        data: Activation data dictionary
        
    Returns:
        List of MLP output tensors, one per layer
    """
    mlp_modules = data["mlp_modules"]
    mlp_outputs_data = data["mlp_outputs"]
    
    mlp_tensors = []
    for module_name in mlp_modules:
        if module_name not in mlp_outputs_data:
            raise ValueError(f"MLP module {module_name} not found in outputs")
        
        output_data = mlp_outputs_data[module_name]["output"]
        if isinstance(output_data, torch.Tensor):
            tensor = output_data
        else:
            tensor = torch.tensor(output_data)
        
        mlp_tensors.append(tensor)
    
    logger.info(f"Extracted {len(mlp_tensors)} MLP output tensors")
    return mlp_tensors

def extract_attention_weights(data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Extract attention weights from attention outputs.
    
    Args:
        data: Activation data dictionary
        
    Returns:
        Dictionary mapping module names to attention weight tensors
    """
    attention_modules = data["attention_modules"]
    attention_outputs_data = data["attention_outputs"]
    
    attention_weights = {}
    for module_name in attention_modules:
        if module_name not in attention_outputs_data:
            raise ValueError(f"Attention module {module_name} not found in outputs")
        
        output_data = attention_outputs_data[module_name]["output"]
        
        # Attention modules return (output, attention_weights) tuple
        # We need the attention weights (element 1)
        if isinstance(output_data, (list, tuple)) and len(output_data) >= 2:
            weights_data = output_data[1]
        else:
            raise ValueError(f"Expected attention output to be tuple with 2+ elements for {module_name}")
        
        if isinstance(weights_data, torch.Tensor):
            weights = weights_data
        else:
            weights = torch.tensor(weights_data)
        
        attention_weights[module_name] = weights
    
    logger.info(f"Extracted attention weights for {len(attention_weights)} modules")
    return attention_weights

def sort_modules_by_layer(modules: List[str]) -> List[str]:
    """
    Sort module names by their layer number.
    
    Args:
        modules: List of module names
        
    Returns:
        Sorted list of module names
    """
    import re
    
    def extract_layer_number(module_name: str) -> int:
        """Extract layer number from module name."""
        numbers = re.findall(r'\d+', module_name)
        if not numbers:
            return 0  # Default for modules without numbers
        return int(numbers[0])
    
    sorted_modules = sorted(modules, key=extract_layer_number)
    logger.info(f"Sorted {len(modules)} modules by layer number")
    return sorted_modules

def normalize_activation_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize activation data for consistent processing.
    
    Args:
        data: Raw activation data
        
    Returns:
        Normalized activation data
    """
    # Validate first
    if not validate_activation_data(data):
        raise ValueError("Activation data validation failed")
    
    # Sort modules by layer for consistent ordering
    normalized_data = data.copy()
    normalized_data["attention_modules"] = sort_modules_by_layer(data["attention_modules"])
    normalized_data["mlp_modules"] = sort_modules_by_layer(data["mlp_modules"])
    
    logger.info("Activation data normalized successfully")
    return normalized_data

def create_activation_payload(
    model_name: str,
    prompt: str,
    attention_modules: List[str],
    attention_outputs: Dict[str, Any],
    mlp_modules: List[str],
    mlp_outputs: Dict[str, Any],
    norm_parameter: List[Any],
    logit_lens_parameter: str
) -> Dict[str, Any]:
    """
    Create a properly formatted activation data payload.
    
    Args:
        model_name: Name of the model
        prompt: Input prompt
        attention_modules: List of attention module names
        attention_outputs: Attention output data
        mlp_modules: List of MLP module names
        mlp_outputs: MLP output data
        norm_parameter: Normalization parameter (as list with 1 element)
        logit_lens_parameter: Logit lens parameter name
        
    Returns:
        Formatted activation data dictionary
    """
    payload = {
        "model": model_name,
        "prompt": prompt,
        "attention_modules": attention_modules,
        "attention_outputs": attention_outputs,
        "mlp_modules": mlp_modules,
        "mlp_outputs": mlp_outputs,
        "norm_parameter": norm_parameter,
        "logit_lens_parameter": logit_lens_parameter
    }
    
    # Validate the created payload
    if not validate_activation_data(payload):
        raise ValueError("Created activation payload is invalid")
    
    logger.info("Created valid activation payload")
    return payload
