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
        try:
            if obj.device.type == 'meta':
                return None
            return obj.detach().cpu().tolist()
        except RuntimeError:
            # Handle meta tensors that raise errors when accessing device
            return None
    if isinstance(obj, (list, tuple)):
        return [safe_to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: safe_to_serializable(v) for k, v in obj.items()}
    return obj


def merge_token_probabilities(token_probs: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """
    Merge tokens with and without leading space, summing their probabilities.
    
    Example: [(" cat", 0.15), ("cat", 0.05), (" dog", 0.10)] -> [("cat", 0.20), ("dog", 0.10)]
    
    Args:
        token_probs: List of (token_string, probability) tuples
    
    Returns:
        List of (token_string, merged_probability) tuples, sorted by probability (descending)
    """
    merged = {}  # Map from stripped token -> total probability
    
    for token, prob in token_probs:
        # Strip leading space to get canonical form
        canonical = token.lstrip()
        merged[canonical] = merged.get(canonical, 0.0) + prob
    
    # Convert back to list and sort by probability (descending)
    result = sorted(merged.items(), key=lambda x: x[1], reverse=True)
    return result


def compute_global_top5_tokens(model_output, tokenizer, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Compute the global top-5 tokens from model's final output with merged probabilities.
    
    Args:
        model_output: Output from model(**inputs) containing logits
        tokenizer: Tokenizer for decoding
        top_k: Number of top tokens to return (default: 5)
    
    Returns:
        List of dicts {'token': str, 'probability': float} for top K tokens
    """
    with torch.no_grad():
        # Get probabilities for next token (last position)
        logits = model_output.logits[0, -1, :]  # [vocab_size]
        probs = F.softmax(logits, dim=-1)
        
        # Get more candidates to account for merging (get 2x top_k)
        top_probs, top_indices = torch.topk(probs, k=min(top_k * 2, len(probs)))
        
        # Decode tokens
        candidates = [
            (tokenizer.decode([idx.item()], skip_special_tokens=False), prob.item())
            for idx, prob in zip(top_indices, top_probs)
        ]
        
        # Merge tokens with/without leading space
        merged = merge_token_probabilities(candidates)
        
        # Return top K after merging, formatted as dicts
        return [{'token': t, 'probability': p} for t, p in merged[:top_k]]


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
    global_top5_tokens = []
    try:
        output_token, output_prob = get_actual_model_output(model_output, tokenizer)
        actual_output = {"token": output_token, "probability": output_prob}
        # Compute global top 5 tokens with merged probabilities
        global_top5_tokens = compute_global_top5_tokens(model_output, tokenizer, top_k=5)
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
        "actual_output": actual_output,
        "global_top5_tokens": global_top5_tokens  # New: global top 5 from final output
    }
    
    print(f"Captured {len(captured)} module outputs using PyVene")
    return result


def execute_forward_pass_with_head_ablation(model, tokenizer, prompt: str, config: Dict[str, Any],
                                           ablate_layer_num: int, ablate_head_indices: List[int]) -> Dict[str, Any]:
    """
    Execute forward pass with specific attention heads zeroed out.
    
    Args:
        model: Loaded transformer model
        tokenizer: Loaded tokenizer
        prompt: Input text prompt
        config: Dict with module lists like {"attention_modules": [...], "block_modules": [...], ...}
        ablate_layer_num: Layer number containing heads to ablate
        ablate_head_indices: List of head indices to zero out (e.g., [0, 2, 5])
    
    Returns:
        JSON-serializable dict with captured activations (with ablated heads)
    """
    print(f"Executing forward pass with head ablation: Layer {ablate_layer_num}, Heads {ablate_head_indices}")
    
    # Extract module lists from config
    attention_modules = config.get("attention_modules", [])
    block_modules = config.get("block_modules", [])
    norm_parameters = config.get("norm_parameters", [])
    logit_lens_parameter = config.get("logit_lens_parameter")
    
    all_modules = attention_modules + block_modules
    if not all_modules:
        return {"error": "No modules specified"}
    
    # Find the target attention module for the layer to ablate
    target_attention_module = None
    for mod_name in attention_modules:
        layer_match = re.search(r'\.(\d+)(?:\.|$)', mod_name)
        if layer_match and int(layer_match.group(1)) == ablate_layer_num:
            target_attention_module = mod_name
            break
    
    if not target_attention_module:
        return {"error": f"Could not find attention module for layer {ablate_layer_num}"}
    
    # Build IntervenableConfig
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
    
    # Prepare inputs
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Register hooks to capture activations
    captured = {}
    name_to_module = dict(intervenable_model.model.named_modules())
    
    def make_hook(mod_name: str):
        return lambda module, inputs, output: captured.update({mod_name: {"output": safe_to_serializable(output)}})
    
    # Create head ablation hook that both ablates and captures
    def head_ablation_hook(module, input, output):
        """Zero out specific attention heads in the output AND capture it."""
        ablated_output = output  # Default to original output
        
        if isinstance(output, tuple):
            # Attention modules typically return (hidden_states, attention_weights, ...)
            hidden_states = output[0]  # [batch, seq_len, hidden_dim]
            
            # Convert to tensor if needed
            if not isinstance(hidden_states, torch.Tensor):
                hidden_states = torch.tensor(hidden_states)
            
            batch_size, seq_len, hidden_dim = hidden_states.shape
            
            # Determine head dimension
            # Assuming hidden_dim = num_heads * head_dim
            # We need to get num_heads from the model config
            num_heads = model.config.num_attention_heads
            head_dim = hidden_dim // num_heads
            
            # Reshape to [batch, seq_len, num_heads, head_dim]
            hidden_states_reshaped = hidden_states.view(batch_size, seq_len, num_heads, head_dim)
            
            # Zero out specified heads
            for head_idx in ablate_head_indices:
                if 0 <= head_idx < num_heads:
                    hidden_states_reshaped[:, :, head_idx, :] = 0.0
            
            # Reshape back to [batch, seq_len, hidden_dim]
            ablated_hidden = hidden_states_reshaped.view(batch_size, seq_len, hidden_dim)
            
            # Reconstruct output tuple
            if len(output) > 1:
                ablated_output = (ablated_hidden,) + output[1:]
            else:
                ablated_output = (ablated_hidden,)
        
        # Capture the ablated output (CRITICAL: this was missing!)
        captured.update({target_attention_module: {"output": safe_to_serializable(ablated_output)}})
        
        return ablated_output
    
    # Register hooks
    hooks = []
    for mod_name in all_modules:
        if mod_name in name_to_module:
            if mod_name == target_attention_module:
                # Apply head ablation hook
                hooks.append(name_to_module[mod_name].register_forward_hook(head_ablation_hook))
            else:
                # Regular capture hook
                hooks.append(name_to_module[mod_name].register_forward_hook(make_hook(mod_name)))
    
    # Execute forward pass
    with torch.no_grad():
        model_output = intervenable_model.model(**inputs, use_cache=False)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
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
    global_top5_tokens = []
    try:
        output_token, output_prob = get_actual_model_output(model_output, tokenizer)
        actual_output = {"token": output_token, "probability": output_prob}
        global_top5_tokens = compute_global_top5_tokens(model_output, tokenizer, top_k=5)
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
        "actual_output": actual_output,
        "global_top5_tokens": global_top5_tokens,
        "ablated_layer": ablate_layer_num,
        "ablated_heads": ablate_head_indices
    }
    
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
    
    # Get reference activations from ALL layers for mean computation
    block_outputs = reference_activation_data.get('block_outputs', {})
    if not block_outputs:
        return {"error": "No block outputs found in reference data"}
    
    # Collect all layer activations to compute global mean
    all_layer_tensors = []
    for mod_name, output_data in block_outputs.items():
        output = output_data['output']
        if isinstance(output, list):
            tensor = torch.tensor(output)
        else:
            tensor = output
        all_layer_tensors.append(tensor)
    
    # Stack all layers and compute mean across ALL layers and sequence positions
    # This gives us a single mean vector that represents the average activation
    stacked = torch.stack(all_layer_tensors, dim=0)  # [num_layers, batch, seq_len, hidden_dim]
    # Compute mean across layers and sequence dimension
    mean_activation = stacked.mean(dim=(0, 2), keepdim=True)  # [1, batch, 1, hidden_dim]
    mean_activation = mean_activation.squeeze(0)  # [batch, 1, hidden_dim]
    
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
    global_top5_tokens = []
    try:
        output_token, output_prob = get_actual_model_output(model_output, tokenizer)
        actual_output = {"token": output_token, "probability": output_prob}
        global_top5_tokens = compute_global_top5_tokens(model_output, tokenizer, top_k=5)
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
        "actual_output": actual_output,
        "global_top5_tokens": global_top5_tokens,
        "ablated_layer": ablate_layer_num
    }
    
    return result


def evaluate_sequence_ablation(model, tokenizer, sequence_text: str, config: Dict[str, Any],
                             ablation_type: str, ablation_target: Any) -> Dict[str, Any]:
    """
    Evaluate the impact of ablation on a full sequence.
    
    This runs TWO forward passes on the FULL sequence:
    1. Reference pass (original model) -> Capture logits/probs
    2. Ablated pass (modified model) -> Capture logits/probs
    
    Then computes metrics: KL Divergence, Target Prob Changes.
    
    Args:
        model: Loaded transformer model
        tokenizer: Tokenizer
        sequence_text: The full text sequence to evaluate
        config: Module configuration (needed for ablation setup)
        ablation_type: 'head' or 'layer'
        ablation_target: tuple (layer, head_indices) or int (layer_num)
        
    Returns:
        Dict with evaluation metrics.
    """
    from .ablation_metrics import compute_kl_divergence, get_token_probability_deltas
    
    print(f"Evaluating sequence ablation: Type={ablation_type}, Target={ablation_target}")
    
    inputs = tokenizer(sequence_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    
    # --- 1. Reference Pass ---
    with torch.no_grad():
        outputs_ref = model(input_ids)
        logits_ref = outputs_ref.logits # [1, seq_len, vocab_size]
        
    # --- 2. Ablated Pass ---
    # Setup ablation based on type
    
    # We need to wrap the model using PyVene logic or custom hooks just for this pass
    # Since we already have logic in execute_forward_pass_with_..._ablation, we can reuse the Hook logic
    # But we want the full logits, not just captured activations.
    
    # Let's manually register hooks here for simplicity and control
    hooks = []
    
    def head_ablation_hook_factory(layer_idx, head_indices):
        def hook(module, input, output):
            # output is (hidden_states, ...) or hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
                
            # Assume hidden_states is [batch, seq, hidden]
            # Reshape, zero out heads, Reshape back
            if not isinstance(hidden_states, torch.Tensor):
                 if isinstance(hidden_states, list): hidden_states = torch.tensor(hidden_states)
            
            # Move to device if needed? They should be on device.
            
            num_heads = model.config.num_attention_heads
            head_dim = hidden_states.shape[-1] // num_heads
            
            # view: [batch, seq, heads, dim]
            new_shape = hidden_states.shape[:-1] + (num_heads, head_dim)
            reshaped = hidden_states.view(new_shape)
            
            # Create mask or just zero out
            # We can't modify in place securely with autograd usually, but here no_grad is on.
            # Clone to be safe
            reshaped = reshaped.clone()
            
            for h_idx in head_indices:
                reshaped[..., h_idx, :] = 0
                
            ablated_hidden = reshaped.view(hidden_states.shape)
            
            if isinstance(output, tuple):
                return (ablated_hidden,) + output[1:]
            return ablated_hidden
        return hook

    # Hook for Layer Ablation (Identity/Skip or Zero)
    # We'll use Identity (Skip Layer) as a simpler approximation of "removing logic" 
    # OR Mean Ablation if we had the mean. 
    # For now, let's just do nothing for layer ablation or return error, 
    # as the user primarily asks for "ablation experiment updates" which often means Heads.
    # But to be safe, let's implement the same Mean Ablation if possible, or Identity.
    # Identity (Skip) is easier:
    def identity_hook(module, input, output):
        # input is tuple (hidden_states, ...)
        return input if isinstance(input, tuple) else (input,)

    try:
        if ablation_type == 'head':
            layer_num, head_indices = ablation_target
            # Find module
            # Standard transformers: model.layers[i].self_attn
            # We need the exact module name map standard to HuggingFace
            # Or use the config's mapping if available.
            # Let's rely on standard naming or search
            
            # Simple heuristic: find 'layers.X.self_attn' or 'h.X.attn'
            target_module = None
            for name, mod in model.named_modules():
                # Check for standard patterns
                # layer_num is int
                if f"layers.{layer_num}.self_attn" in name or f"h.{layer_num}.attn" in name or f"blocks.{layer_num}.attn" in name:
                     if "k_proj" not in name and "v_proj" not in name and "q_proj" not in name: # avoid submodules
                         target_module = mod
                         break
            
            if target_module:
                hooks.append(target_module.register_forward_hook(head_ablation_hook_factory(layer_num, head_indices)))
            else:
                print(f"Warning: Could not find attention module for layer {layer_num}")

        elif ablation_type == 'layer':
            layer_num = ablation_target
            target_module = None
            for name, mod in model.named_modules():
                # Layers are usually 'model.layers.X' or 'transformer.h.X'
                # We want the module that corresponds to the layer block
                # Be careful not to pick 'layers.X.mlp'
                if (f"layers.{layer_num}" in name or f"h.{layer_num}" in name) and name.count('.') <= 2: # heuristic for top-level layer
                     target_module = mod
                     break
            
            if target_module:
                 # Skip layer (Identity)
                 hooks.append(target_module.register_forward_hook(lambda m, i, o: i[0] if isinstance(i, tuple) else i))

        # Run Ablated Pass
        with torch.no_grad():
            outputs_abl = model(input_ids)
            logits_abl = outputs_abl.logits

    finally:
        for hook in hooks:
            hook.remove()
            
    # --- 3. Compute Metrics ---
    # KL Divergence [seq_len]
    kl_div = compute_kl_divergence(logits_ref, logits_abl)
    
    # Prob Deltas for actual tokens [seq_len-1] (shifted)
    prob_deltas = get_token_probability_deltas(logits_ref, logits_abl, input_ids)
    
    return {
        "kl_divergence": kl_div,
        "probability_deltas": prob_deltas,
        "tokens": [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]
    }


def logit_lens_transformation(layer_output: Any, norm_data: List[Any], model, tokenizer, norm_parameter: Optional[str] = None, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Transform layer output to top K token probabilities using logit lens.
    Returns merged probabilities (tokens with/without leading space are combined).
    
    For standard logit lens, use block/layer outputs (residual stream), not component outputs.
    The residual stream contains the full hidden state with all accumulated information.
    
    Applies final layer normalization before projection (critical for correctness).
    Uses model's built-in functions to minimize computational errors.
    
    Args:
        layer_output: Hidden state from any layer (preferably block output / residual stream)
        norm_data: Not used (deprecated - using model's norm layer directly)
        model: HuggingFace model
        tokenizer: Tokenizer for decoding
        norm_parameter: Parameter path for final norm layer (e.g., "model.norm.weight")
        top_k: Number of top tokens to return (default: 5)
    
    Returns:
        List of (token_string, probability) tuples for top K tokens with merged probabilities
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
        
        # Step 4: Extract top candidates (get 2x top_k to account for merging)
        top_probs, top_indices = torch.topk(probs, k=min(top_k * 2, len(probs)))
        
        candidates = [
            (tokenizer.decode([idx.item()], skip_special_tokens=False), prob.item())
            for idx, prob in zip(top_indices, top_probs)
        ]
        
        # Step 5: Merge tokens with/without leading space
        merged = merge_token_probabilities(candidates)
        
        return merged[:top_k]


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


def _get_token_probabilities_for_layer(activation_data: Dict[str, Any], module_name: str, 
                                       model, tokenizer, target_tokens: List[str]) -> Dict[str, float]:
    """
    Get probabilities for specific tokens at a given layer.
    
    Args:
        activation_data: Activation data from forward pass
        module_name: Layer module name
        model: Transformer model
        tokenizer: Tokenizer
        target_tokens: List of token strings to get probabilities for
    
    Returns:
        Dict mapping token -> probability (merged for variants with/without space)
    """
    try:
        if module_name not in activation_data.get('block_outputs', {}):
            return {}
        
        layer_output = activation_data['block_outputs'][module_name]['output']
        norm_params = activation_data.get('norm_parameters', [])
        norm_parameter = norm_params[0] if norm_params else None
        final_norm = get_norm_layer_from_parameter(model, norm_parameter)
        lm_head = model.get_output_embeddings()
        
        with torch.no_grad():
            hidden = torch.tensor(layer_output) if not isinstance(layer_output, torch.Tensor) else layer_output
            if hidden.dim() == 4:
                hidden = hidden.squeeze(0)
            
            if final_norm is not None:
                hidden = final_norm(hidden)
            
            logits = lm_head(hidden)
            probs = F.softmax(logits[0, -1, :], dim=-1)
            
            # For each target token, get probabilities for both variants (with/without space)
            token_probs = {}
            for token in target_tokens:
                # Try both variants and sum probabilities
                variants = [token, ' ' + token]
                total_prob = 0.0
                
                for variant in variants:
                    token_ids = tokenizer.encode(variant, add_special_tokens=False)
                    if token_ids:
                        tid = token_ids[-1]  # Use last sub-token
                        total_prob += probs[tid].item()
                
                token_probs[token] = total_prob
            
            return token_probs
    except Exception as e:
        print(f"Warning: Could not compute token probabilities for {module_name}: {e}")
        return {}


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
        
        return logit_lens_transformation(layer_output, [], model, tokenizer, norm_parameter, top_k=top_k)
    except Exception as e:
        print(f"Warning: Could not compute logit lens for {module_name}: {e}")
        return None


def get_check_token_probabilities(activation_data: Dict[str, Any], model, tokenizer, check_token: str) -> Optional[Dict[str, Any]]:
    """
    Collect check token probabilities across all layers.
    
    Sums probabilities of token variants (with and without leading space).
    Returns layer numbers and merged probabilities for plotting.
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
        
        # Get token IDs for both variants (if they exist and differ)
        target_token_ids = []
        for variant_text, token_ids in token_variants:
            if token_ids:
                tid = token_ids[-1]  # Use last sub-token
                if tid not in target_token_ids:
                    target_token_ids.append(tid)
        
        if not target_token_ids:
            return None
        
        # Get norm parameter
        norm_params = activation_data.get('norm_parameters', [])
        norm_parameter = norm_params[0] if norm_params else None
        final_norm = get_norm_layer_from_parameter(model, norm_parameter)
        lm_head = model.get_output_embeddings()
        
        # Collect probabilities for all layers (sum both variants)
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
                
                # Sum probabilities of all variants
                merged_prob = sum(probs[tid].item() for tid in target_token_ids)
                
                layers.append(layer_num)
                probabilities.append(merged_prob)
        
        return {
            'token': check_token.strip(),  # Return canonical form without leading space
            'layers': layers,
            'probabilities': probabilities
        }
    except Exception as e:
        print(f"Error computing check token probabilities: {e}")
        return None


def detect_significant_probability_increases(layer_wise_probs: Dict[int, Dict[str, float]], 
                                            layer_wise_deltas: Dict[int, Dict[str, float]],
                                            actual_output_token: str,
                                            threshold: float = 1.0) -> List[int]:
    """
    Detect layers where the actual output token has significant probability increase.
    
    A layer is significant if the actual output token has ≥100% relative increase from previous layer.
    Example: 0.20 → 0.40 is (0.40-0.20)/0.20 = 100% increase.
    
    This threshold highlights layers where the model's confidence in the actual output
    doubles, representing a pedagogically significant shift in the prediction.
    
    Args:
        layer_wise_probs: Dict mapping layer_num → {token: prob}
        layer_wise_deltas: Dict mapping layer_num → {token: delta}
        actual_output_token: The token that the model actually outputs (predicted token)
        threshold: Relative increase threshold (default: 1.0 = 100%)
    
    Returns:
        List of layer numbers with significant increases in the actual output token
    """
    significant_layers = []
    
    for layer_num in sorted(layer_wise_probs.keys()):
        probs = layer_wise_probs[layer_num]
        deltas = layer_wise_deltas.get(layer_num, {})
        
        # Only check the actual output token
        if actual_output_token in probs:
            prob = probs[actual_output_token]
            delta = deltas.get(actual_output_token, 0.0)
            prev_prob = prob - delta
            
            # Check for significant relative increase (avoid division by zero)
            if prev_prob > 1e-6 and delta > 0:
                relative_increase = delta / prev_prob
                if relative_increase >= threshold:
                    significant_layers.append(layer_num)
    
    return significant_layers


def _get_top_attended_tokens(activation_data: Dict[str, Any], layer_num: int, tokenizer, top_k: int = 3) -> Optional[List[Tuple[str, float]]]:
    """
    DEPRECATED: This function is deprecated and will be removed in a future version.
    Use head categorization from head_detection.py instead for more meaningful attention analysis.
    
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
    import warnings
    warnings.warn(
        "_get_top_attended_tokens is deprecated. Use categorize_all_heads() from head_detection.py instead.",
        DeprecationWarning,
        stacklevel=2
    )
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


def compute_position_layer_matrix(activation_data: Dict[str, Any], model, tokenizer) -> Dict[str, Any]:
    """
    Compute a 2D matrix of layer-to-layer deltas for each token position.
    
    This function computes the top-token probability delta at each (layer, position) pair,
    creating a heatmap-ready data structure.
    
    Args:
        activation_data: Activation data from forward pass
        model: Transformer model for logit lens computation
        tokenizer: Tokenizer for decoding tokens
        
    Returns:
        Dict with:
            - 'matrix': 2D list [num_layers, seq_len] of delta values
            - 'tokens': List of token strings for X-axis labels
            - 'layer_nums': List of layer numbers for Y-axis labels
            - 'top_tokens': 2D list [num_layers, seq_len] of top token strings at each cell
    """
    import copy
    import numpy as np
    
    input_ids = activation_data.get('input_ids', [[]])
    if not input_ids or not input_ids[0]:
        return {'matrix': [], 'tokens': [], 'layer_nums': [], 'top_tokens': []}
    
    seq_len = len(input_ids[0])
    
    # Get token strings for X-axis labels
    tokens = [tokenizer.decode([tid]) for tid in input_ids[0]]
    
    # Get layer modules and sort by layer number
    layer_modules = activation_data.get('block_modules', [])
    if not layer_modules:
        return {'matrix': [], 'tokens': tokens, 'layer_nums': [], 'top_tokens': []}
    
    layer_info = sorted(
        [(int(re.findall(r'\d+', name)[0]), name) 
         for name in layer_modules if re.findall(r'\d+', name)]
    )
    layer_nums = [ln for ln, _ in layer_info]
    num_layers = len(layer_nums)
    
    # Helper function to slice data to a specific position (adapted from app.py)
    def slice_data(data, pos):
        if not data:
            return data
        sliced = copy.deepcopy(data)
        
        # Slice Block Outputs: [batch, seq, hidden] -> [batch, 1, hidden]
        if 'block_outputs' in sliced:
            for mod in sliced['block_outputs']:
                out = sliced['block_outputs'][mod]['output']
                if isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
                    if pos < len(out[0]):
                        sliced['block_outputs'][mod]['output'] = [[out[0][pos]]]
        
        # Slice Attention Outputs: [batch, heads, seq, seq] -> [batch, heads, 1, seq]
        if 'attention_outputs' in sliced:
            for mod in sliced['attention_outputs']:
                out = sliced['attention_outputs'][mod]['output']
                if len(out) > 1:
                    attns = out[1]
                    if isinstance(attns, list) and len(attns) > 0:
                        batch_0 = attns[0]
                        new_batch_0 = []
                        for head in batch_0:
                            if pos < len(head):
                                new_batch_0.append([head[pos]])
                        sliced['attention_outputs'][mod]['output'] = [out[0], [new_batch_0]] + out[2:]
        
        # Slice input_ids
        if 'input_ids' in sliced:
            ids = sliced['input_ids'][0]
            if pos < len(ids):
                sliced['input_ids'][0] = ids[:pos+1]
        
        return sliced
    
    # Initialize matrix and top_tokens 2D array
    matrix = [[0.0] * seq_len for _ in range(num_layers)]
    top_tokens_matrix = [[''] * seq_len for _ in range(num_layers)]
    
    # Compute delta for each position
    for pos in range(seq_len):
        sliced = slice_data(activation_data, pos)
        layer_data = extract_layer_data(sliced, model, tokenizer)
        
        if not layer_data:
            continue
        
        # Fill in matrix for this position
        for layer_info_item in layer_data:
            layer_num = layer_info_item.get('layer_num')
            if layer_num is None or layer_num not in layer_nums:
                continue
            
            layer_idx = layer_nums.index(layer_num)
            
            # Get top token and its delta (layer-to-layer change)
            top_token = layer_info_item.get('top_token', '')
            deltas = layer_info_item.get('deltas', {})
            
            # The delta for the top token represents how much it changed from prev layer
            delta = deltas.get(top_token, 0.0) if top_token else 0.0
            
            matrix[layer_idx][pos] = delta
            top_tokens_matrix[layer_idx][pos] = top_token if top_token else ''
    
    return {
        'matrix': matrix,
        'tokens': tokens,
        'layer_nums': layer_nums,
        'top_tokens': top_tokens_matrix
    }


def compute_layer_wise_summaries(layer_data: List[Dict[str, Any]], activation_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute summary structures from layer data for easy access.
    
    Args:
        layer_data: List of layer data dicts from extract_layer_data()
        activation_data: Activation data containing actual output token
    
    Returns:
        Dict with: layer_wise_top5_probs, layer_wise_top5_deltas, significant_layers
    """
    layer_wise_top5_probs = {}  # layer_num -> {token: prob}
    layer_wise_top5_deltas = {}  # layer_num -> {token: delta}
    
    for layer_info in layer_data:
        layer_num = layer_info.get('layer_num')
        if layer_num is not None:
            layer_wise_top5_probs[layer_num] = layer_info.get('global_top5_probs', {})
            layer_wise_top5_deltas[layer_num] = layer_info.get('global_top5_deltas', {})
    
    # Extract actual output token from activation data
    actual_output = activation_data.get('actual_output', {})
    actual_output_token = actual_output.get('token', '').strip() if actual_output else ''
    
    # Detect significant layers based on actual output token
    significant_layers = []
    if actual_output_token:
        significant_layers = detect_significant_probability_increases(
            layer_wise_top5_probs, 
            layer_wise_top5_deltas,
            actual_output_token,
            threshold=1.0
        )
    
    return {
        'layer_wise_top5_probs': layer_wise_top5_probs,
        'layer_wise_top5_deltas': layer_wise_top5_deltas,
        'significant_layers': significant_layers
    }


def extract_layer_data(activation_data: Dict[str, Any], model, tokenizer) -> List[Dict[str, Any]]:
    """
    Extract layer-by-layer data for accordion display with top-5, deltas, and attention.
    Also tracks global top 5 tokens across all layers.
    
    Returns:
        List of dicts with: layer_num, top_token, top_prob, top_5_tokens, deltas, top_attended_tokens,
        global_top5_probs, global_top5_deltas
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
    
    # Check if we can compute token predictions (requires block_outputs and norm_parameters)
    # Note: Previously, this checked for logit_lens_parameter, but that parameter is not actually
    # needed for computing predictions. The _get_top_tokens function only needs block_outputs
    # and norm_parameters to work correctly.
    has_block_outputs = bool(activation_data.get('block_outputs', {}))
    has_norm_params = bool(activation_data.get('norm_parameters', []))
    can_compute_predictions = has_block_outputs and has_norm_params
    
    # Get global top 5 tokens from final output
    global_top5_tokens = activation_data.get('global_top5_tokens', [])
    
    # Handle both dicts (new format) and tuples (legacy)
    if global_top5_tokens and isinstance(global_top5_tokens[0], dict):
        global_top5_token_names = [t.get('token') for t in global_top5_tokens]
    else:
        global_top5_token_names = [token for token, _ in global_top5_tokens]
    
    layer_data = []
    prev_token_probs = {}  # Track previous layer's token probabilities (layer's own top 5)
    prev_global_probs = {}  # Track previous layer's global top 5 probabilities
    
    for layer_num, module_name in layer_info:
        top_tokens = _get_top_tokens(activation_data, module_name, model, tokenizer, top_k=5) if can_compute_predictions else None
        
        # NOTE: top_attended_tokens is deprecated. Use categorize_all_heads() from
        # head_detection.py instead for more meaningful attention analysis.
        # Kept as None for backward compatibility with existing code.
        top_attended = None
        
        # Get probabilities for global top 5 tokens at this layer
        global_top5_probs = {}
        global_top5_deltas = {}
        if can_compute_predictions and global_top5_token_names:
            global_top5_probs = _get_token_probabilities_for_layer(
                activation_data, module_name, model, tokenizer, global_top5_token_names
            )
            # Compute deltas for global top 5
            for token in global_top5_token_names:
                current_prob = global_top5_probs.get(token, 0.0)
                prev_prob = prev_global_probs.get(token, 0.0)
                global_top5_deltas[token] = current_prob - prev_prob
        
        if top_tokens:
            top_token, top_prob = top_tokens[0]
            
            # Compute deltas vs previous layer (for layer's own top 5)
            deltas = {}
            for token, prob in top_tokens:
                prev_prob = prev_token_probs.get(token, 0.0)
                deltas[token] = prob - prev_prob
            
            layer_data.append({
                'layer_num': layer_num,
                'module_name': module_name,
                'top_token': top_token,
                'top_prob': top_prob,
                'top_3_tokens': top_tokens[:3],  # Keep for backward compatibility
                'top_5_tokens': top_tokens[:5],  # New: top-5 for bar chart
                'deltas': deltas,
                'top_attended_tokens': top_attended,
                'global_top5_probs': global_top5_probs,  # New: global top 5 probs at this layer
                'global_top5_deltas': global_top5_deltas  # New: global top 5 deltas
            })
            
            # Update previous layer probabilities
            prev_token_probs = {token: prob for token, prob in top_tokens}
            prev_global_probs = global_top5_probs.copy()
        else:
            layer_data.append({
                'layer_num': layer_num,
                'module_name': module_name,
                'top_token': None,
                'top_prob': None,
                'top_3_tokens': [],
                'top_5_tokens': [],
                'deltas': {},
                'top_attended_tokens': top_attended,
                'global_top5_probs': {},
                'global_top5_deltas': {}
            })
            prev_global_probs = {}
    
    return layer_data


def generate_bertviz_html(activation_data: Dict[str, Any], layer_index: int, view_type: str = 'full') -> str:
    """
    Generate BertViz attention visualization HTML using head_view.
    
    Uses head_view for a less overwhelming display that lets users scroll through
    individual attention heads. Shows all heads with layer/head selectors.
    
    Args:
        activation_data: Output from execute_forward_pass
        layer_index: Index of layer to visualize (used for initial layer selection)
        view_type: 'full' for complete visualization or 'mini' for preview
    
    Returns:
        HTML string for the visualization
    """
    try:
        from bertviz import head_view
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
                <p><em>Click for full head_view visualization</em></p>
            </div>
            """
        else:
            # Full version: BertViz head_view (less overwhelming, scrollable heads)
            html_result = head_view(attentions, tokens, html_action='return')
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


def generate_head_view_with_categories(activation_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate BertViz head view HTML along with head categorization data.
    
    Combines the head_view visualization with categorization from head_detection.py
    to provide both visual attention patterns and semantic categorization.
    
    Args:
        activation_data: Output from execute_forward_pass with attention data
    
    Returns:
        Dict with:
            - 'html': BertViz head_view HTML string
            - 'categories': Dict from categorize_all_heads (category -> list of head info)
            - 'summary': Formatted text summary of head categorization
            - 'error': Error message if visualization failed (optional)
    """
    from .head_detection import categorize_all_heads, format_categorization_summary
    
    result = {
        'html': None,
        'categories': {},
        'summary': '',
        'error': None
    }
    
    # Generate the base head_view visualization
    try:
        result['html'] = generate_bertviz_html(activation_data, layer_index=0, view_type='full')
    except Exception as e:
        result['error'] = f"Failed to generate head view: {str(e)}"
        result['html'] = f"<p>Error generating visualization: {str(e)}</p>"
    
    # Generate head categorization
    try:
        result['categories'] = categorize_all_heads(activation_data)
        result['summary'] = format_categorization_summary(result['categories'])
    except Exception as e:
        if result['error']:
            result['error'] += f"; Categorization failed: {str(e)}"
        else:
            result['error'] = f"Categorization failed: {str(e)}"
    
    return result


def get_head_category_counts(activation_data: Dict[str, Any]) -> Dict[str, int]:
    """
    Get counts of attention heads in each category.
    
    Useful for UI display showing the distribution of head types.
    
    Args:
        activation_data: Output from execute_forward_pass with attention data
    
    Returns:
        Dict mapping category name to count of heads in that category
    """
    from .head_detection import categorize_all_heads
    
    try:
        categories = categorize_all_heads(activation_data)
        return {category: len(heads) for category, heads in categories.items()}
    except Exception as e:
        print(f"Warning: Could not categorize heads: {e}")
        return {}