"""Model pattern extraction utilities for transformer models."""

import re
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from pyvene import RepresentationConfig, IntervenableConfig, IntervenableModel


def extract_patterns(model, use_modules=True) -> Dict[str, List[str]]:
    """Extract patterns from model modules or parameters."""
    items = model.named_modules() if use_modules else model.named_parameters()
    patterns = {}
    
    for name, _ in items:
        if not name:
            continue
        # Replace numeric sequences with {N} placeholder
        pattern = re.sub(r'(\.|_)(\d+)(\.|_|$)', r'\1{N}\3', name)
        pattern = re.sub(r'([a-zA-Z])(\d+)(\.|_|$)', r'\1{N}\3', pattern)
        
        if pattern not in patterns:
            patterns[pattern] = []
        patterns[pattern].append(name)
    
    return patterns


def load_model_and_get_patterns(model_name: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Load model from HuggingFace Hub and extract module/parameter patterns.
    
    Returns:
        (module_patterns, parameter_patterns): Pattern dictionaries mapping patterns to name lists
    """
    print(f"Loading model: {model_name}")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation='eager')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    
    # Extract patterns
    module_patterns = extract_patterns(model, use_modules=True)
    param_patterns = extract_patterns(model, use_modules=False)
    
    print(f"Found {len(module_patterns)} module patterns, {len(param_patterns)} parameter patterns")
    
    return module_patterns, param_patterns


def safe_to_serializable(obj: Any) -> Any:
    """Convert tensors to lists recursively for JSON serialization."""
    if torch.is_tensor(obj):
        # Check if tensor is a meta tensor (no data) and skip it
        if obj.device.type == 'meta':
            return None
        return obj.detach().cpu().tolist()
    if isinstance(obj, (list, tuple)):
        return [safe_to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: safe_to_serializable(v) for k, v in obj.items()}
    return obj


def get_actual_model_output(model_output, tokenizer) -> Tuple[str, float]:
    """
    Extract the predicted token from model's output.
    
    Args:
        model_output: Output from model(**inputs) containing logits
        tokenizer: Tokenizer for decoding
    
    Returns:
        (token_string, probability) for the predicted next token
    """
    with torch.no_grad():
        # Get probabilities for next token (last position)
        logits = model_output.logits[0, -1, :]  # [vocab_size]
        probs = F.softmax(logits, dim=-1)
        
        # Get top predicted token
        top_prob, top_idx = probs.max(dim=-1)
        token_str = tokenizer.decode([top_idx.item()], skip_special_tokens=False)
        
        return token_str, top_prob.item()


def execute_forward_pass(model, tokenizer, prompt: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute forward pass with PyVene IntervenableModel to capture activations from specified modules.
    
    Args:
        model: Loaded transformer model
        tokenizer: Loaded tokenizer
        prompt: Input text prompt
        config: Dict with module lists like {"attention_modules": [...], "block_modules": [...], ...}
    
    Returns:
        JSON-serializable dict with captured activations and metadata
    """
    print(f"Executing forward pass with prompt: '{prompt}'")
    
    # Extract module lists from config
    attention_modules = config.get("attention_modules", [])
    block_modules = config.get("block_modules", [])
    norm_parameters = config.get("norm_parameters", [])
    logit_lens_parameter = config.get("logit_lens_parameter")
    
    all_modules = attention_modules + block_modules
    if not all_modules:
        print("No modules specified for capture")
        return {"error": "No modules specified"}
    
    # Build IntervenableConfig from module names
    intervenable_representations = []
    for mod_name in all_modules:
        # Extract layer index from module name
        layer_match = re.search(r'\.(\d+)(?:\.|$)', mod_name)
        if not layer_match:
            return {"error": f"Invalid module name format: {mod_name}"}
        
        # Determine component type based on module name
        if 'attn' in mod_name or 'attention' in mod_name:
            component = 'attention_output'
        else:
            # Layer/block modules (e.g., "model.layers.0", "transformer.h.0")
            # These represent the residual stream (full layer output)
            component = 'block_output'
        
        intervenable_representations.append(
            RepresentationConfig(layer=int(layer_match.group(1)), component=component, unit="pos")
        )
    
    # Create IntervenableConfig and wrap model
    intervenable_config = IntervenableConfig(
        intervenable_representations=intervenable_representations
    )
    intervenable_model = IntervenableModel(intervenable_config, model)
    
    print(f"Created IntervenableModel with {len(intervenable_representations)} representations")
    
    # Prepare inputs
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Register hooks to capture activations
    captured = {}
    name_to_module = dict(intervenable_model.model.named_modules())
    
    def make_hook(mod_name: str):
        return lambda module, inputs, output: captured.update({mod_name: {"output": safe_to_serializable(output)}})
    
    hooks = [
        name_to_module[mod_name].register_forward_hook(make_hook(mod_name))
        for mod_name in all_modules if mod_name in name_to_module
    ]
    
    # Execute forward pass through underlying model and capture actual output
    with torch.no_grad():
        model_output = intervenable_model.model(**inputs, use_cache=False)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Separate outputs by type based on module name pattern
    attention_outputs = {}
    block_outputs = {}
    
    for mod_name, output in captured.items():
        if 'attn' in mod_name or 'attention' in mod_name:
            attention_outputs[mod_name] = output
        else:
            # Block/layer outputs (residual stream - full layer output)
            block_outputs[mod_name] = output
    
    # Capture normalization parameters (deprecated - kept for backward compatibility)
    all_params = dict(model.named_parameters())
    norm_data = [safe_to_serializable(all_params[p]) for p in norm_parameters if p in all_params]
    
    # Extract predicted token from model output
    actual_output = None
    try:
        output_token, output_prob = get_actual_model_output(model_output, tokenizer)
        actual_output = {"token": output_token, "probability": output_prob}
    except Exception as e:
        print(f"Warning: Could not extract model output: {e}")
    
    # Build output dictionary
    result = {
        "model": getattr(model.config, "name_or_path", "unknown"),
        "prompt": prompt,
        "input_ids": safe_to_serializable(inputs["input_ids"]),
        "attention_modules": list(attention_outputs.keys()),
        "attention_outputs": attention_outputs,
        "block_modules": list(block_outputs.keys()),
        "block_outputs": block_outputs,
        "norm_parameters": norm_parameters,
        "norm_data": norm_data,
        "logit_lens_parameter": logit_lens_parameter,
        "actual_output": actual_output
    }
    
    print(f"Captured {len(captured)} module outputs using PyVene")
    return result


def execute_forward_pass_with_layer_ablation(model, tokenizer, prompt: str, config: Dict[str, Any], 
                                             ablate_layer_num: int, reference_activation_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute forward pass with mean ablation on a specific layer.
    
    Args:
        model: Loaded transformer model
        tokenizer: Loaded tokenizer
        prompt: Input text prompt
        config: Dict with module lists like {"attention_modules": [...], "block_modules": [...], ...}
        ablate_layer_num: Layer number to ablate
        reference_activation_data: Original activation data containing the reference activations
    
    Returns:
        JSON-serializable dict with captured activations (with ablated layer)
    """
    # Extract module lists from config
    attention_modules = config.get("attention_modules", [])
    block_modules = config.get("block_modules", [])
    norm_parameters = config.get("norm_parameters", [])
    logit_lens_parameter = config.get("logit_lens_parameter")
    
    all_modules = attention_modules + block_modules
    if not all_modules:
        return {"error": "No modules specified"}
    
    # Find the target module for the layer to ablate
    target_module_name = None
    for mod_name in block_modules:
        layer_match = re.search(r'\.(\d+)(?:\.|$)', mod_name)
        if layer_match and int(layer_match.group(1)) == ablate_layer_num:
            target_module_name = mod_name
            break
    
    if not target_module_name:
        return {"error": f"Could not find module for layer {ablate_layer_num}"}
    
    # Get reference activation for mean computation
    block_outputs = reference_activation_data.get('block_outputs', {})
    if target_module_name not in block_outputs:
        return {"error": f"No reference activation found for {target_module_name}"}
    
    reference_output = block_outputs[target_module_name]['output']
    
    # Convert to tensor and compute mean across sequence dimension
    if isinstance(reference_output, list):
        ref_tensor = torch.tensor(reference_output)
    else:
        ref_tensor = reference_output
    
    # Shape is typically [batch, seq_len, hidden_dim]
    # Compute mean over seq_len dimension
    mean_activation = ref_tensor.mean(dim=1, keepdim=True)  # [batch, 1, hidden_dim]
    
    # Prepare inputs
    inputs = tokenizer(prompt, return_tensors="pt")
    seq_len = inputs['input_ids'].shape[1]
    
    # Broadcast mean to match sequence length
    ablation_value = mean_activation.expand(-1, seq_len, -1)  # [batch, seq_len, hidden_dim]
    
    # Build IntervenableConfig from module names
    intervenable_representations = []
    for mod_name in all_modules:
        layer_match = re.search(r'\.(\d+)(?:\.|$)', mod_name)
        if not layer_match:
            return {"error": f"Invalid module name format: {mod_name}"}
        
        if 'attn' in mod_name or 'attention' in mod_name:
            component = 'attention_output'
        else:
            component = 'block_output'
        
        intervenable_representations.append(
            RepresentationConfig(layer=int(layer_match.group(1)), component=component, unit="pos")
        )
    
    intervenable_config = IntervenableConfig(
        intervenable_representations=intervenable_representations
    )
    intervenable_model = IntervenableModel(intervenable_config, model)
    
    # Register hooks to capture activations
    captured = {}
    name_to_module = dict(intervenable_model.model.named_modules())
    
    def make_hook(mod_name: str):
        return lambda module, inputs, output: captured.update({mod_name: {"output": safe_to_serializable(output)}})
    
    # Register ablation hook for target module
    def ablation_hook(module, input, output):
        # Replace output with mean activation
        if isinstance(output, tuple):
            # For modules that return tuples (hidden_states, ...), replace first element
            ablated = (ablation_value,) + output[1:]
            return ablated
        else:
            return ablation_value
    
    hooks = []
    for mod_name in all_modules:
        if mod_name in name_to_module:
            if mod_name == target_module_name:
                # Apply ablation hook
                hooks.append(name_to_module[mod_name].register_forward_hook(ablation_hook))
            else:
                # Regular capture hook
                hooks.append(name_to_module[mod_name].register_forward_hook(make_hook(mod_name)))
    
    # Execute forward pass
    with torch.no_grad():
        model_output = intervenable_model.model(**inputs, use_cache=False)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Capture ablated layer output as well
    captured[target_module_name] = {"output": safe_to_serializable(ablation_value)}
    
    # Separate outputs by type
    attention_outputs = {}
    block_outputs = {}
    
    for mod_name, output in captured.items():
        if 'attn' in mod_name or 'attention' in mod_name:
            attention_outputs[mod_name] = output
        else:
            block_outputs[mod_name] = output
    
    # Capture normalization parameters
    all_params = dict(model.named_parameters())
    norm_data = [safe_to_serializable(all_params[p]) for p in norm_parameters if p in all_params]
    
    # Extract predicted token from model output
    actual_output = None
    try:
        output_token, output_prob = get_actual_model_output(model_output, tokenizer)
        actual_output = {"token": output_token, "probability": output_prob}
    except Exception as e:
        print(f"Warning: Could not extract model output: {e}")
    
    # Build output dictionary
    result = {
        "model": getattr(model.config, "name_or_path", "unknown"),
        "prompt": prompt,
        "input_ids": safe_to_serializable(inputs["input_ids"]),
        "attention_modules": list(attention_outputs.keys()),
        "attention_outputs": attention_outputs,
        "block_modules": list(block_outputs.keys()),
        "block_outputs": block_outputs,
        "norm_parameters": norm_parameters,
        "norm_data": norm_data,
        "logit_lens_parameter": logit_lens_parameter,
        "actual_output": actual_output,
        "ablated_layer": ablate_layer_num
    }
    
    return result


def logit_lens_transformation(layer_output: Any, norm_data: List[Any], model, logit_lens_parameter: str, tokenizer, norm_parameter: Optional[str] = None, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Transform layer output to top K token probabilities using logit lens.
    
    For standard logit lens, use block/layer outputs (residual stream), not component outputs.
    The residual stream contains the full hidden state with all accumulated information.
    
    Applies final layer normalization before projection (critical for correctness).
    Uses model's built-in functions to minimize computational errors.
    
    Args:
        layer_output: Hidden state from any layer (preferably block output / residual stream)
        norm_data: Not used (deprecated - using model's norm layer directly)
        model: HuggingFace model
        logit_lens_parameter: Not used (deprecated)
        tokenizer: Tokenizer for decoding
        norm_parameter: Parameter path for final norm layer (e.g., "model.norm.weight")
        top_k: Number of top tokens to return (default: 5)
    
    Returns:
        List of (token_string, probability) tuples for top K tokens
    """
    with torch.no_grad():
        # Convert to tensor and ensure proper shape [batch, seq_len, hidden_dim]
        hidden = torch.tensor(layer_output) if not isinstance(layer_output, torch.Tensor) else layer_output
        if hidden.dim() == 4:
            hidden = hidden.squeeze(0)
        
        # Step 1: Apply final layer normalization (critical for intermediate layers)
        final_norm = get_norm_layer_from_parameter(model, norm_parameter)
        if final_norm is not None:
            hidden = final_norm(hidden)
        
        # Step 2: Project to vocab space using model's lm_head
        lm_head = model.get_output_embeddings()
        logits = lm_head(hidden)
        
        # Step 3: Get probabilities via softmax
        probs = F.softmax(logits[0, -1, :], dim=-1)
        
        # Step 4: Extract top K tokens
        top_probs, top_indices = torch.topk(probs, k=top_k)
        
        return [
            (tokenizer.decode([idx.item()], skip_special_tokens=False), prob.item())
            for idx, prob in zip(top_indices, top_probs)
        ]


def get_norm_layer_from_parameter(model, norm_parameter: Optional[str]) -> Optional[Any]:
    """
    Get the final layer normalization module from the model using the norm parameter path.
    
    Args:
        model: The transformer model
        norm_parameter: Parameter path (e.g., "model.norm.weight") or None
        
    Returns:
        The normalization layer module, or None if not found
    """
    if norm_parameter:
        # Convert parameter path to module path (remove .weight/.bias suffix)
        module_path = norm_parameter.replace('.weight', '').replace('.bias', '')
        try:
            parts = module_path.split('.')
            obj = model
            for part in parts:
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            print(f"Warning: Could not find norm layer at {module_path}")
    
    # Fallback: Try common final norm layer names if no parameter specified
    for attr_path in ['model.norm', 'transformer.ln_f', 'model.decoder.final_layer_norm', 
                      'gpt_neox.final_layer_norm', 'transformer.norm_f']:
        try:
            parts = attr_path.split('.')
            obj = model
            for part in parts:
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    return None


def _get_top_tokens(activation_data: Dict[str, Any], module_name: str, model, tokenizer, top_k: int = 5) -> Optional[List[Tuple[str, float]]]:
    """
    Helper: Get top K tokens for a layer's block output.
    
    Uses block outputs (residual stream) which represent the full hidden state
    after all layer computations (attention + feedforward + residuals).
    """
    try:
        # Get block output (residual stream)
        if module_name not in activation_data.get('block_outputs', {}):
            return None
        
        layer_output = activation_data['block_outputs'][module_name]['output']
        
        # Get norm parameter from activation data (should be a single parameter or list with one item)
        norm_params = activation_data.get('norm_parameters', [])
        norm_parameter = norm_params[0] if norm_params else None
        
        return logit_lens_transformation(layer_output, [], model, None, tokenizer, norm_parameter, top_k=top_k)
    except Exception as e:
        print(f"Warning: Could not compute logit lens for {module_name}: {e}")
        return None


def get_check_token_probabilities(activation_data: Dict[str, Any], model, tokenizer, check_token: str) -> Optional[Dict[str, Any]]:
    """
    Collect check token probabilities across all layers.
    
    Tries both with and without leading space and uses the variant with higher probability.
    Returns layer numbers and probabilities for plotting.
    """
    if not check_token or not check_token.strip():
        return None
    
    try:
        # Get block modules (all layers)
        layer_modules = activation_data.get('block_modules', [])
        if not layer_modules:
            return None
        
        # Extract and sort layers
        layer_info = sorted(
            [(int(re.findall(r'\d+', name)[0]), name) 
             for name in layer_modules if re.findall(r'\d+', name)]
        )
        
        # Try tokenizing with and without leading space
        token_variants = [
            (check_token.strip(), tokenizer.encode(check_token.strip(), add_special_tokens=False)),
            (' ' + check_token.strip(), tokenizer.encode(' ' + check_token.strip(), add_special_tokens=False))
        ]
        
        # Determine which variant to use (choose one with valid token IDs)
        target_token_id = None
        for variant_text, token_ids in token_variants:
            if token_ids:
                target_token_id = token_ids[-1]  # Use last sub-token
                break
        
        if target_token_id is None:
            return None
        
        # Get norm parameter
        norm_params = activation_data.get('norm_parameters', [])
        norm_parameter = norm_params[0] if norm_params else None
        final_norm = get_norm_layer_from_parameter(model, norm_parameter)
        lm_head = model.get_output_embeddings()
        
        # Collect probabilities for all layers
        layers = []
        probabilities = []
        
        for layer_num, module_name in layer_info:
            layer_output = activation_data['block_outputs'][module_name]['output']
            
            with torch.no_grad():
                hidden = torch.tensor(layer_output) if not isinstance(layer_output, torch.Tensor) else layer_output
                if hidden.dim() == 4:
                    hidden = hidden.squeeze(0)
                
                if final_norm is not None:
                    hidden = final_norm(hidden)
                
                logits = lm_head(hidden)
                probs = F.softmax(logits[0, -1, :], dim=-1)
                prob = probs[target_token_id].item()
                
                layers.append(layer_num)
                probabilities.append(prob)
        
        return {
            'token': tokenizer.decode([target_token_id], skip_special_tokens=False),
            'layers': layers,
            'probabilities': probabilities
        }
    except Exception as e:
        print(f"Error computing check token probabilities: {e}")
        return None


def _compute_certainty(probs: List[float]) -> float:
    """
    Compute normalized certainty from probability distribution.
    Formula: certainty = 1 - H(p)/log(K) where H is Shannon entropy.
    
    Args:
        probs: List of probabilities (top-K)
    
    Returns:
        Certainty score in [0, 1] where 1 = completely certain
    """
    import math
    if not probs or len(probs) == 0:
        return 0.0
    
    # Compute Shannon entropy: H = -Σ(p_i * log(p_i))
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log(p)
    
    # Normalize by max entropy (log(K))
    max_entropy = math.log(len(probs))
    if max_entropy == 0:
        return 1.0
    
    # Certainty = 1 - normalized_entropy
    certainty = 1.0 - (entropy / max_entropy)
    return max(0.0, min(1.0, certainty))  # Clamp to [0, 1]


def _get_top_attended_tokens(activation_data: Dict[str, Any], layer_num: int, tokenizer, top_k: int = 3) -> Optional[List[Tuple[str, float]]]:
    """
    Get top-K attended input tokens for the current position (last token) in a layer.
    Averages attention across all heads.
    
    Args:
        activation_data: Output from execute_forward_pass
        layer_num: Layer number to analyze
        tokenizer: Tokenizer for decoding tokens
        top_k: Number of top attended tokens to return
    
    Returns:
        List of (token_string, attention_weight) tuples, sorted by weight (highest first)
    """
    try:
        attention_outputs = activation_data.get('attention_outputs', {})
        input_ids = activation_data.get('input_ids', [])
        
        # print(f"DEBUG _get_top_attended_tokens: layer_num={layer_num}, attention_outputs keys={list(attention_outputs.keys())}")
        
        if not attention_outputs or not input_ids:
            print(f"DEBUG _get_top_attended_tokens: Missing data - attention_outputs empty={not attention_outputs}, input_ids empty={not input_ids}")
            return None
        
        # Find attention output for this layer
        target_module = None
        for module_name in attention_outputs.keys():
            numbers = re.findall(r'\d+', module_name)
            if numbers and int(numbers[0]) == layer_num:
                target_module = module_name
                break
        
        if not target_module:
            return None
        
        attention_output = attention_outputs[target_module]['output']
        if not isinstance(attention_output, list) or len(attention_output) < 2:
            return None
        
        # Get attention weights: [batch, heads, seq_len, seq_len]
        attention_weights = torch.tensor(attention_output[1])
        
        # Average across heads: [seq_len, seq_len]
        avg_attention = attention_weights[0].mean(dim=0)
        
        # Get attention from last position to all positions
        last_pos_attention = avg_attention[-1, :]  # [seq_len]
        
        # Get top-K attended positions
        top_values, top_indices = torch.topk(last_pos_attention, min(top_k, len(last_pos_attention)))
        
        # Convert to tokens
        input_ids_tensor = torch.tensor(input_ids[0]) if isinstance(input_ids[0], list) else torch.tensor(input_ids)
        result = []
        for idx, weight in zip(top_indices, top_values):
            token_id = input_ids_tensor[idx].item()
            token_str = tokenizer.decode([token_id], skip_special_tokens=False)
            result.append((token_str, weight.item()))
        
        return result
        
    except Exception as e:
        print(f"Warning: Could not compute attended tokens for layer {layer_num}: {e}")
        return None


def extract_layer_data(activation_data: Dict[str, Any], model, tokenizer) -> List[Dict[str, Any]]:
    """
    Extract layer-by-layer data for accordion display with top-5, deltas, certainty, and attention.
    
    Returns:
        List of dicts with: layer_num, top_token, top_prob, top_5_tokens, deltas, certainty, top_attended_tokens
    """
    layer_modules = activation_data.get('block_modules', [])
    if not layer_modules:
        return []
    
    # Debug: Check if attention outputs are present
    attention_outputs = activation_data.get('attention_outputs', {})
    print(f"DEBUG extract_layer_data: Found {len(attention_outputs)} attention modules")
    
    # Extract and sort layers by layer number
    layer_info = sorted(
        [(int(re.findall(r'\d+', name)[0]), name) 
         for name in layer_modules if re.findall(r'\d+', name)]
    )
    
    logit_lens_enabled = activation_data.get('logit_lens_parameter') is not None
    layer_data = []
    prev_token_probs = {}  # Track previous layer's token probabilities
    
    for layer_num, module_name in layer_info:
        top_tokens = _get_top_tokens(activation_data, module_name, model, tokenizer, top_k=5) if logit_lens_enabled else None
        
        # Get top-3 attended tokens for this layer
        top_attended = _get_top_attended_tokens(activation_data, layer_num, tokenizer, top_k=3)
        
        if top_tokens:
            top_token, top_prob = top_tokens[0]
            
            # Compute deltas vs previous layer
            deltas = {}
            for token, prob in top_tokens:
                prev_prob = prev_token_probs.get(token, 0.0)
                deltas[token] = prob - prev_prob
            
            # Compute certainty from top-5 probabilities
            probs = [prob for _, prob in top_tokens]
            certainty = _compute_certainty(probs)
            
            layer_data.append({
                'layer_num': layer_num,
                'module_name': module_name,
                'top_token': top_token,
                'top_prob': top_prob,
                'top_3_tokens': top_tokens[:3],  # Keep for backward compatibility
                'top_5_tokens': top_tokens[:5],  # New: top-5 for bar chart
                'deltas': deltas,
                'certainty': certainty,
                'top_attended_tokens': top_attended  # New: attention view
            })
            
            # Update previous layer probabilities
            prev_token_probs = {token: prob for token, prob in top_tokens}
        else:
            layer_data.append({
                'layer_num': layer_num,
                'module_name': module_name,
                'top_token': None,
                'top_prob': None,
                'top_3_tokens': [],
                'top_5_tokens': [],
                'deltas': {},
                'certainty': 0.0,
                'top_attended_tokens': top_attended
            })
    
    return layer_data


def generate_bertviz_html(activation_data: Dict[str, Any], layer_index: int, view_type: str = 'full') -> str:
    """
    Generate BertViz attention visualization HTML using model_view.
    
    Shows all layers with the specified layer highlighted/focused.
    
    Args:
        activation_data: Output from execute_forward_pass
        layer_index: Index of layer to visualize (for context; model_view shows all layers)
        view_type: 'full' for complete visualization or 'mini' for preview
    
    Returns:
        HTML string for the visualization
    """
    try:
        from bertviz import model_view
        from transformers import AutoTokenizer
        
        # Extract attention modules and sort by layer
        attention_outputs = activation_data.get('attention_outputs', {})
        if not attention_outputs:
            return f"<p>No attention data available</p>"
        
        # Sort attention modules by layer number
        layer_attention_pairs = []
        for module_name in attention_outputs.keys():
            numbers = re.findall(r'\d+', module_name)
            if numbers:
                layer_num = int(numbers[0])
                attention_output = attention_outputs[module_name]['output']
                if isinstance(attention_output, list) and len(attention_output) >= 2:
                    # Get attention weights (element 1 of the output tuple)
                    attention_weights = torch.tensor(attention_output[1])  # [batch, heads, seq, seq]
                    layer_attention_pairs.append((layer_num, attention_weights))
        
        if not layer_attention_pairs:
            return f"<p>No valid attention data found</p>"
        
        # Sort by layer number and extract attention tensors
        layer_attention_pairs.sort(key=lambda x: x[0])
        attentions = tuple(attn for _, attn in layer_attention_pairs)
        
        # Get tokens
        input_ids = torch.tensor(activation_data['input_ids'])
        model_name = activation_data.get('model', 'unknown')
        
        # Load tokenizer and convert to tokens
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        raw_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        # Clean up tokens (remove special tokenizer artifacts like Ġ for GPT-2)
        tokens = [token.replace('Ġ', ' ') if token.startswith('Ġ') else token for token in raw_tokens]
        
        # Generate visualization based on view_type
        if view_type == 'mini':
            # Mini version: simplified HTML preview
            return f"""
            <div style="padding:10px; border:1px solid #ccc; border-radius:5px;">
                <h4>Layer {layer_index} Attention Preview</h4>
                <p><strong>Tokens:</strong> {' '.join(tokens[:8])}{'...' if len(tokens) > 8 else ''}</p>
                <p><strong>Total Layers:</strong> {len(attentions)}</p>
                <p><strong>Heads per Layer:</strong> {attentions[0].shape[1] if attentions else 'N/A'}</p>
                <p><em>Click for full model_view visualization</em></p>
            </div>
            """
        else:
            # Full version: complete bertviz model_view visualization (shows all layers)
            html_result = model_view(attentions, tokens, html_action='return')
            return html_result.data if hasattr(html_result, 'data') else str(html_result)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"<p>Error generating visualization: {str(e)}</p>"


def generate_category_bertviz_html(activation_data: Dict[str, Any], category_heads: List[Dict[str, Any]]) -> str:
    """
    Generate BertViz attention visualization HTML for a specific category of heads.
    
    Shows only the attention patterns for heads in the specified category.
    
    Args:
        activation_data: Output from execute_forward_pass
        category_heads: List of head info dicts for this category (from categorize_all_heads)
    
    Returns:
        HTML string for the visualization
    """
    try:
        from bertviz import head_view
        from transformers import AutoTokenizer
        
        if not category_heads:
            return "<p>No heads in this category.</p>"
        
        # Extract attention modules and sort by layer
        attention_outputs = activation_data.get('attention_outputs', {})
        if not attention_outputs:
            return "<p>No attention data available</p>"
        
        # Build a map of layer -> head indices for this category
        category_map = {}  # layer_num -> list of head indices
        for head_info in category_heads:
            layer = head_info['layer']
            head = head_info['head']
            if layer not in category_map:
                category_map[layer] = []
            category_map[layer].append(head)
        
        # Sort attention modules by layer number and filter heads
        # Track which layers we've already processed to avoid duplicates
        layer_attention_pairs = []
        processed_layers = set()
        
        for module_name in attention_outputs.keys():
            numbers = re.findall(r'\d+', module_name)
            if numbers:
                layer_num = int(numbers[0])
                
                # Skip layers not in this category
                if layer_num not in category_map:
                    continue
                
                # Skip if we've already processed this layer (prevents duplicate/mismatched tensors)
                if layer_num in processed_layers:
                    continue
                
                attention_output = attention_outputs[module_name]['output']
                if isinstance(attention_output, list) and len(attention_output) >= 2:
                    # Get attention weights (element 1 of the output tuple)
                    full_attention = torch.tensor(attention_output[1])  # [batch, heads, seq, seq]
                    
                    # Filter to only include heads in this category
                    head_indices = category_map[layer_num]
                    filtered_attention = full_attention[:, head_indices, :, :]  # Select specific heads
                    
                    layer_attention_pairs.append((layer_num, filtered_attention))
                    processed_layers.add(layer_num)
        
        if not layer_attention_pairs:
            return "<p>No valid attention data found for this category.</p>"
        
        # Sort by layer number and extract attention tensors
        layer_attention_pairs.sort(key=lambda x: x[0])
        attentions = tuple(attn for _, attn in layer_attention_pairs)
        
        # Get tokens
        input_ids = torch.tensor(activation_data['input_ids'])
        model_name = activation_data.get('model', 'unknown')
        
        # Load tokenizer and convert to tokens
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        raw_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        # Clean up tokens (remove special tokenizer artifacts like Ġ for GPT-2)
        tokens = [token.replace('Ġ', ' ') if token.startswith('Ġ') else token for token in raw_tokens]
        
        # Generate visualization using head_view (better for showing specific heads)
        html_result = head_view(attentions, tokens, html_action='return')
        base_html = html_result.data if hasattr(html_result, 'data') else str(html_result)
        
        # Create a legend mapping head indices to their actual layer-head labels
        legend_items = []
        head_counter = 0
        for layer_num, _ in layer_attention_pairs:
            head_indices = category_map[layer_num]
            for head_idx in head_indices:
                legend_items.append(f"Head {head_counter}: L{layer_num}-H{head_idx}")
                head_counter += 1
        
        legend_html = """
        <div style="background-color: #f8f9fa; padding: 10px; margin-bottom: 10px; border-radius: 5px; border: 1px solid #dee2e6;">
            <strong style="color: #495057;">Head Index Reference:</strong><br/>
            <div style="font-size: 12px; color: #6c757d; margin-top: 5px; display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 5px;">
                {items}
            </div>
        </div>
        """.format(items=''.join(f'<span>{item}</span>' for item in legend_items))
        
        # Prepend legend to the visualization
        return legend_html + base_html
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"<p>Error generating category visualization: {str(e)}</p>"