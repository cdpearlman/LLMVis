"""
Activation capture and logit lens pipeline.
Adapts logic from activation_capture.py and logit_lens_analysis.py for dashboard use.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from ..utils.data_contract import create_activation_payload, extract_norm_weight, extract_mlp_outputs_as_tensors
from ..utils.token_formatting import get_printable_tokens
from ..constants import TOP_K, DEVICE

logger = logging.getLogger(__name__)

def safe_to_serializable(obj: Any) -> Any:
    """Convert tensors to lists and handle other types for JSON serialization."""
    try:
        if torch.is_tensor(obj):
            return obj.detach().cpu().tolist()
        if isinstance(obj, (list, tuple)):
            return [safe_to_serializable(x) for x in obj]
        if isinstance(obj, dict):
            return {k: safe_to_serializable(v) for k, v in obj.items()}
        return obj
    except Exception:
        return str(obj)

def expand_pattern_to_modules(selected_pattern: str, pattern_to_modules: Dict[str, List[str]]) -> List[str]:
    """
    Expand a single generalized pattern to all concrete per-layer module names.
    
    Args:
        selected_pattern: Generalized pattern with {N} placeholder
        pattern_to_modules: Mapping from patterns to module lists
        
    Returns:
        List of concrete module names
    """
    if selected_pattern in pattern_to_modules:
        modules = pattern_to_modules[selected_pattern]
        logger.info(f"Expanded pattern '{selected_pattern}' to {len(modules)} modules")
        return modules
    else:
        logger.warning(f"Pattern '{selected_pattern}' not found in pattern_to_modules")
        return []

def register_hooks(model, module_names: List[str]) -> Tuple[Dict[str, Any], List[Any]]:
    """
    Register forward hooks on specified modules to capture activations.
    
    Args:
        model: PyTorch model
        module_names: List of module names to hook
        
    Returns:
        Tuple of (captured_data, hooks_list)
    """
    captured: Dict[str, Any] = {}
    hooks: List[Any] = []
    name_to_module = dict(model.named_modules())

    def make_hook(mod_name: str):
        def hook_fn(module, inputs, output):
            try:
                captured[mod_name] = {'output': safe_to_serializable(output)}
                if len(captured) <= 3:  # Debug first few captures
                    logger.debug(f"Captured from {mod_name}")
            except Exception as e:
                captured[mod_name] = {"error": f"{e}"}
                logger.error(f"Error capturing from {mod_name}: {e}")
        return hook_fn

    registered_count = 0
    for mod_name in module_names:
        module = name_to_module.get(mod_name)
        if module is None:
            logger.warning(f"Module '{mod_name}' not found; skipping")
            continue
        try:
            hooks.append(module.register_forward_hook(make_hook(mod_name)))
            registered_count += 1
        except Exception as e:
            logger.error(f"Failed to register hook for {mod_name}: {e}")
    
    logger.info(f"Registered {registered_count} forward hooks")
    return captured, hooks

def remove_hooks(hooks: List[Any]) -> None:
    """Remove all registered hooks."""
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

def capture_activations(
    model, 
    tokenizer: AutoTokenizer, 
    prompt: str, 
    attention_modules: List[str], 
    mlp_modules: List[str]
) -> Tuple[Dict[str, Any], Dict[str, Any], torch.Tensor]:
    """
    Capture activations from selected modules.
    
    Args:
        model: PyTorch model
        tokenizer: Tokenizer
        prompt: Input prompt
        attention_modules: List of attention module names
        mlp_modules: List of MLP module names
        
    Returns:
        Tuple of (attention_outputs, mlp_outputs, input_ids)
    """
    all_modules = attention_modules + mlp_modules
    
    if not all_modules:
        logger.warning("No modules selected for capture")
        return {}, {}, torch.tensor([])
    
    # Register hooks
    logger.info(f"Registering hooks for {len(all_modules)} modules")
    captured_activations, hooks = register_hooks(model, all_modules)

    try:
        # Run forward pass
        logger.info(f"Running forward pass with prompt: '{prompt[:50]}...'")
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            _ = model(**inputs, use_cache=False)
        
        logger.info(f"Captured activations from {len(captured_activations)} modules")
        
    finally:
        # Always remove hooks
        remove_hooks(hooks)
    
    # Separate attention and MLP outputs
    attention_outputs = {k: v for k, v in captured_activations.items() if k in attention_modules}
    mlp_outputs = {k: v for k, v in captured_activations.items() if k in mlp_modules}
    
    return attention_outputs, mlp_outputs, inputs["input_ids"]

def capture_norm_parameter(model, param_name: str) -> List[Any]:
    """
    Capture the selected normalization parameter.
    
    Args:
        model: PyTorch model
        param_name: Parameter name to capture
        
    Returns:
        List with exactly one serialized tensor
    """
    all_params = dict(model.named_parameters())
    
    if param_name not in all_params:
        raise ValueError(f"Parameter '{param_name}' not found in model")
    
    param_tensor = all_params[param_name]
    serialized = safe_to_serializable(param_tensor)
    
    logger.info(f"Captured norm parameter '{param_name}' with shape {param_tensor.shape}")
    return [serialized]

def load_logit_lens_weights(model, param_name: str) -> torch.Tensor:
    """
    Load the selected logit lens parameter tensor from the model.
    
    Args:
        model: Already loaded model instance
        param_name: Parameter name to load
        
    Returns:
        Logit lens weight tensor
    """
    logger.info(f"Loading logit lens weights: {param_name}")
    
    try:
        # Get the parameter directly from the model
        all_params = dict(model.named_parameters())
        
        if param_name not in all_params:
            raise ValueError(f"Parameter '{param_name}' not found in model")
        
        logit_lens_weight = all_params[param_name].detach().clone()
        logger.info(f"Loaded logit lens weights with shape: {logit_lens_weight.shape}")
        
        return logit_lens_weight
        
    except Exception as e:
        logger.error(f"Failed to load logit lens weights '{param_name}': {e}")
        raise ValueError(f"Could not load {param_name}: {e}")

def apply_normalization(
    hidden_states: torch.Tensor, 
    norm_weight: torch.Tensor, 
    norm_bias: torch.Tensor = None, 
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Apply layer normalization using captured weights.
    
    Args:
        hidden_states: Input tensor
        norm_weight: Normalization weight
        norm_bias: Normalization bias (None for RMSNorm)
        eps: Epsilon for numerical stability
        
    Returns:
        Normalized tensor
    """
    if norm_bias is None:
        # RMSNorm
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)
        return hidden_states * norm_weight
    else:
        # LayerNorm
        mean = hidden_states.mean(-1, keepdim=True)
        variance = ((hidden_states - mean).pow(2)).mean(-1, keepdim=True)
        hidden_states = (hidden_states - mean) * torch.rsqrt(variance + eps)
        return hidden_states * norm_weight + norm_bias

def apply_logit_lens(
    mlp_outputs: List[torch.Tensor],
    logit_lens_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    tokenizer: AutoTokenizer,
    top_k: int = TOP_K
) -> List[List[Tuple[str, float]]]:
    """
    Apply logit lens analysis to MLP outputs.
    
    Args:
        mlp_outputs: List of MLP output tensors
        logit_lens_weight: Logit lens projection weight
        norm_weight: Normalization weight
        tokenizer: Tokenizer for token decoding
        top_k: Number of top tokens to extract
        
    Returns:
        List of top-k results per layer: [[(token, prob), ...], ...]
    """
    results = []
    
    logger.info(f"Applying logit lens to {len(mlp_outputs)} layers")
    
    with torch.no_grad():
        for layer_idx, mlp_output in enumerate(mlp_outputs):
            try:
                # Ensure correct shape
                if len(mlp_output.shape) == 4:
                    mlp_output = mlp_output.squeeze(0)
                
                if len(mlp_output.shape) != 3:
                    logger.warning(f"Unexpected MLP output shape at layer {layer_idx}: {mlp_output.shape}")
                    continue
                
                # Apply normalization (RMSNorm, no bias)
                normalized = apply_normalization(mlp_output, norm_weight, None)
                
                # Project to vocabulary space
                logits = torch.matmul(normalized, logit_lens_weight.T)
                
                # Apply softmax
                probs = F.softmax(logits, dim=-1)
                
                # Get last token position
                last_token_probs = probs[0, -1, :]  # [batch=0, last_token, vocab]
                
                # Get top-k tokens
                top_probs, top_indices = torch.topk(last_token_probs, top_k)
                
                # Convert to strings
                layer_results = []
                for i in range(top_k):
                    token_id = top_indices[i].item()
                    probability = top_probs[i].item()
                    token_str = tokenizer.decode([token_id], skip_special_tokens=False)
                    layer_results.append((token_str, probability))
                
                results.append(layer_results)
                
                # Log progress
                token_strs = [f"{token}({prob:.3f})" for token, prob in layer_results]
                logger.debug(f"Layer {layer_idx}: {', '.join(token_strs)}")
                
            except Exception as e:
                logger.error(f"Error processing layer {layer_idx}: {e}")
                # Add empty results to maintain layer alignment
                results.append([])
    
    logger.info(f"Logit lens analysis complete: {len(results)} layers processed")
    return results

def run_complete_pipeline(
    model,
    tokenizer: AutoTokenizer,
    model_name: str,
    prompt: str,
    attention_pattern: str,
    mlp_pattern: str,
    norm_param_name: str,
    logit_lens_param_name: str,
    pattern_to_modules: Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    Run the complete activation capture and logit lens pipeline.
    
    Args:
        model: Loaded model
        tokenizer: Tokenizer
        model_name: Model name
        prompt: Input prompt
        attention_pattern: Selected attention pattern
        mlp_pattern: Selected MLP pattern
        norm_param_name: Selected normalization parameter
        logit_lens_param_name: Selected logit lens parameter
        pattern_to_modules: Pattern to module mapping
        
    Returns:
        Complete pipeline results including activation data and logit lens results
    """
    try:
        # Expand patterns to modules
        attention_modules = expand_pattern_to_modules(attention_pattern, pattern_to_modules)
        mlp_modules = expand_pattern_to_modules(mlp_pattern, pattern_to_modules)
        
        if not attention_modules and not mlp_modules:
            raise ValueError("No modules found for selected patterns")
        
        # Capture activations
        attention_outputs, mlp_outputs, input_ids = capture_activations(
            model, tokenizer, prompt, attention_modules, mlp_modules
        )
        
        # Capture normalization parameter
        norm_parameter = capture_norm_parameter(model, norm_param_name)
        
        # Create activation payload
        activation_data = create_activation_payload(
            model_name, prompt, attention_modules, attention_outputs,
            mlp_modules, mlp_outputs, norm_parameter, logit_lens_param_name
        )
        
        # Load logit lens weights
        logit_lens_weight = load_logit_lens_weights(model, logit_lens_param_name)
        
        # Extract data for logit lens
        mlp_tensors = extract_mlp_outputs_as_tensors(activation_data)
        norm_weight = extract_norm_weight(activation_data)
        
        # Apply logit lens
        logit_lens_results = apply_logit_lens(
            mlp_tensors, logit_lens_weight, norm_weight, tokenizer, TOP_K
        )
        
        # Get printable tokens
        tokens = get_printable_tokens(tokenizer, input_ids[0].tolist())
        
        # Compile complete results
        results = {
            'success': True,
            'activation_data': activation_data,
            'logit_lens_results': logit_lens_results,
            'tokens': tokens,
            'input_ids': input_ids.tolist(),
            'num_layers': len(mlp_modules),
            'message': f"Successfully processed {len(mlp_modules)} layers"
        }
        
        logger.info("Pipeline completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': f"Pipeline failed: {e}"
        }
