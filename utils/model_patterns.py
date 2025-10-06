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


def logit_lens_transformation(layer_output: Any, norm_data: List[Any], model, logit_lens_parameter: str, tokenizer, norm_parameter: Optional[str] = None) -> List[Tuple[str, float]]:
    """
    Transform layer output to top 3 token probabilities using logit lens.
    
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
    
    Returns:
        List of (token_string, probability) tuples for top 3 tokens
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
        
        # Step 4: Extract top 3 tokens
        top_probs, top_indices = torch.topk(probs, k=3)
        
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


def token_to_color(token: str) -> str:
    """Convert token to consistent color using hash."""
    import hashlib
    hash_val = int(hashlib.md5(token.encode()).hexdigest()[:6], 16)
    hue = hash_val % 360
    return f'hsl({hue}, 70%, 50%)'


def _get_top_tokens(activation_data: Dict[str, Any], module_name: str, model, tokenizer) -> Optional[List[Tuple[str, float]]]:
    """
    Helper: Get top 3 tokens for a layer's block output.
    
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
        
        return logit_lens_transformation(layer_output, [], model, None, tokenizer, norm_parameter)
    except Exception as e:
        print(f"Warning: Could not compute logit lens for {module_name}: {e}")
        return None


def _get_check_token_probability(activation_data: Dict[str, Any], module_name: str, model, tokenizer, check_token: str) -> Optional[Tuple[str, float]]:
    """
    Helper: Get probability for a specific check token at a layer's output.
    
    Tries both with and without leading space and uses the variant with higher probability.
    If check_token produces multiple sub-tokens, uses the last sub-token as per specs.
    """
    try:
        # Try tokenizing with and without leading space (different token IDs)
        token_variants = [
            (check_token, tokenizer.encode(check_token, add_special_tokens=False)),
            (' ' + check_token, tokenizer.encode(' ' + check_token, add_special_tokens=False))
        ]
        
        # Get block output (needed for probability computation)
        if module_name not in activation_data.get('block_outputs', {}):
            print(f"Warning: module_name '{module_name}' not in block_outputs")
            return None
        
        layer_output = activation_data['block_outputs'][module_name]['output']
        
        # Get norm parameter (same as _get_top_tokens)
        norm_params = activation_data.get('norm_parameters', [])
        norm_parameter = norm_params[0] if norm_params else None
        
        # Apply logit lens transformation once (shared for both variants)
        with torch.no_grad():
            hidden = torch.tensor(layer_output) if not isinstance(layer_output, torch.Tensor) else layer_output
            if hidden.dim() == 4:
                hidden = hidden.squeeze(0)
            
            # Apply final norm
            final_norm = get_norm_layer_from_parameter(model, norm_parameter)
            if final_norm is not None:
                hidden = final_norm(hidden)
            
            # Project to vocab space
            lm_head = model.get_output_embeddings()
            logits = lm_head(hidden)
            probs = F.softmax(logits[0, -1, :], dim=-1)
            
            # Evaluate both variants and choose the one with higher probability
            best_prob = 0.0
            best_token_str = None
            best_variant = None
            
            for variant_text, token_ids in token_variants:
                if not token_ids:
                    continue
                
                # Use last sub-token if multiple (as per plans.md)
                target_token_id = token_ids[-1]
                token_str = tokenizer.decode([target_token_id], skip_special_tokens=False)
                prob = probs[target_token_id].item()
                
                print(f"DEBUG check_token variant: '{variant_text}' -> token_ids={token_ids}, target_id={target_token_id}, decoded='{token_str}', prob={prob:.6f}")
                
                if prob > best_prob:
                    best_prob = prob
                    best_token_str = token_str
                    best_variant = variant_text
            
            if best_token_str is None:
                print(f"Warning: check_token '{check_token}' produced no valid token IDs")
                return None
            
            print(f"DEBUG check_token SELECTED for {module_name}: variant='{best_variant}', token='{best_token_str}', prob={best_prob:.6f}")
            
            return (best_token_str, best_prob)
    except Exception as e:
        print(f"Warning: Could not compute check token probability for {module_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def _create_node(layer_num: int, module_name: str, x_pos: int, top_token: Optional[Tuple[str, float]] = None) -> Dict[str, Any]:
    """Helper: Create a Cytoscape node."""
    label = f'L{layer_num}'
    if top_token:
        token, prob = top_token
        label = f'L{layer_num}\n{token}\n{prob:.2f}'
    
    return {
        'data': {'id': f'layer_{layer_num}', 'label': label, 'layer_num': layer_num, 'module_name': module_name},
        'position': {'x': x_pos, 'y': 200}
    }


def _create_edge(src_layer: int, tgt_layer: int, token: str, prob: float, rank: int = 0) -> Dict[str, Any]:
    """Helper: Create a Cytoscape edge."""
    edge_id = f'edge_{src_layer}_{tgt_layer}_{rank}' if rank else f'edge_{src_layer}_{tgt_layer}'
    return {
        'data': {
            'id': edge_id,
            'source': f'layer_{src_layer}',
            'target': f'layer_{tgt_layer}' if isinstance(tgt_layer, int) else tgt_layer,
            'token': token,
            'probability': prob,
            'label': f"{token} | {prob:.3f}",  # Tooltip on hover
            'width': max(2, prob * 10),
            'opacity': max(0.3, prob),
            'color': token_to_color(token)
        }
    }


def format_data_for_cytoscape(activation_data: Dict[str, Any], model, tokenizer, check_token: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Convert activation data to Cytoscape format with nodes (layers) and edges (top-3 tokens + optional check token).
    
    Uses block outputs (full layer outputs / residual stream) for logit lens visualization.
    
    Args:
        activation_data: Output from execute_forward_pass
        model: The transformer model
        tokenizer: The tokenizer
        check_token: Optional token to track as 4th edge (with minimum opacity)
    """
    # Get block modules (full layer outputs)
    layer_modules = activation_data.get('block_modules', [])
    if not layer_modules:
        return []
    
    # Extract and sort layers by layer number
    layer_info = sorted(
        [(int(re.findall(r'\d+', name)[0]), name) 
         for name in layer_modules if re.findall(r'\d+', name)]
    )
    
    elements = []
    logit_lens_enabled = activation_data.get('logit_lens_parameter') is not None
    
    # Create layer nodes
    for i, (layer_num, module_name) in enumerate(layer_info):
        top_tokens = _get_top_tokens(activation_data, module_name, model, tokenizer) if logit_lens_enabled else None
        top_token = top_tokens[0] if top_tokens else None
        elements.append(_create_node(layer_num, module_name, i * 120 + 60, top_token))
    
    # Create edges between consecutive layers
    if logit_lens_enabled:
        for i in range(len(layer_info) - 1):
            curr_layer_num, curr_module = layer_info[i]
            next_layer_num = layer_info[i + 1][0]
            
            top_tokens = _get_top_tokens(activation_data, curr_module, model, tokenizer)
            if top_tokens:
                for rank, (token, prob) in enumerate(top_tokens):
                    elements.append(_create_edge(curr_layer_num, next_layer_num, token, prob, rank))
            
            # Add 4th edge for check token if provided
            if check_token and check_token.strip():
                check_result = _get_check_token_probability(activation_data, curr_module, model, tokenizer, check_token.strip())
                if check_result:
                    token, prob = check_result
                    # Create edge with rank=3 (4th edge)
                    # Use actual probability for edge data, but enforce minimum opacity for visibility
                    edge = _create_edge(curr_layer_num, next_layer_num, token, prob, rank=3)
                    # Override opacity with minimum of 0.25 for visibility (even if prob is 0)
                    edge['data']['opacity'] = max(0.25, prob)
                    elements.append(edge)
    
    # Add final output node
    actual_output = activation_data.get('actual_output')
    if actual_output and layer_info:
        last_layer_num = layer_info[-1][0]
        output_token, output_prob = actual_output['token'], actual_output['probability']
        
        # Output node
        elements.append({
            'data': {'id': 'output_node', 'label': f'Output\n{output_token[:8]}\n{output_prob:.2f}'},
            'position': {'x': len(layer_info) * 120 + 60, 'y': 200}
        })
        
        # Edge to output
        elements.append(_create_edge(last_layer_num, 'output_node', output_token, output_prob))
    
    return elements


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