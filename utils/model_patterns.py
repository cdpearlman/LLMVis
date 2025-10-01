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


def execute_forward_pass(model, tokenizer, prompt: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute forward pass with PyVene IntervenableModel to capture activations from specified modules.
    
    Args:
        model: Loaded transformer model
        tokenizer: Loaded tokenizer
        prompt: Input text prompt
        config: Dict with module lists like {"attention_modules": [...], "mlp_modules": [...], ...}
    
    Returns:
        JSON-serializable dict with captured activations and metadata
    """
    print(f"Executing forward pass with prompt: '{prompt}'")
    
    # Extract module lists from config
    attention_modules = config.get("attention_modules", [])
    mlp_modules = config.get("mlp_modules", [])
    other_modules = config.get("other_modules", [])
    norm_parameters = config.get("norm_parameters", [])
    logit_lens_parameter = config.get("logit_lens_parameter")
    
    all_modules = attention_modules + mlp_modules + other_modules
    if not all_modules:
        print("No modules specified for capture")
        return {"error": "No modules specified"}
    
    # Build IntervenableConfig from module names
    intervenable_representations = []
    for mod_name in all_modules:
        # Extract layer index from module name
        layer_match = re.search(r'\.(\d+)(?:\.|$)', mod_name)
        if not layer_match:
            print(f"ERROR: Could not extract layer number from module: {mod_name}")
            return {"error": f"Invalid module name format: {mod_name}"}
        
        layer_idx = int(layer_match.group(1))
        
        # Determine component type based on module category
        if mod_name in mlp_modules:
            component = 'mlp_output'
        elif mod_name in attention_modules:
            component = 'attention_output'
        else:
            component = 'block_output'
        
        intervenable_representations.append(
            RepresentationConfig(
                layer=layer_idx,
                component=component,
                unit="pos",
                max_number_of_units=None
            )
        )
    
    # Create IntervenableConfig and wrap model
    intervenable_config = IntervenableConfig(
        intervenable_representations=intervenable_representations
    )
    intervenable_model = IntervenableModel(intervenable_config, model)
    
    print(f"Created IntervenableModel with {len(intervenable_representations)} representations")
    
    # Prepare inputs
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Capture activations via hooks on underlying model
    captured = {}
    hooks = []
    name_to_module = dict(intervenable_model.model.named_modules())
    
    def make_hook(mod_name: str):
        def hook_fn(module, inputs, output):
            captured[mod_name] = {"output": safe_to_serializable(output)}
        return hook_fn
    
    for mod_name in all_modules:
        if mod_name in name_to_module:
            hooks.append(name_to_module[mod_name].register_forward_hook(make_hook(mod_name)))
    
    # Execute forward pass through underlying model
    with torch.no_grad():
        _ = intervenable_model.model(**inputs, use_cache=False)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Separate outputs by type
    attention_outputs = {k: v for k, v in captured.items() if k in attention_modules}
    mlp_outputs = {k: v for k, v in captured.items() if k in mlp_modules}
    other_outputs = {k: v for k, v in captured.items() if k in other_modules}
    
    # Capture normalization parameters
    norm_data = []
    if norm_parameters:
        all_params = dict(model.named_parameters())
        for param_name in norm_parameters:
            if param_name in all_params:
                norm_data.append(safe_to_serializable(all_params[param_name]))
    
    # Build output dictionary
    result = {
        "model": getattr(model.config, "name_or_path", "unknown"),
        "prompt": prompt,
        "input_ids": safe_to_serializable(inputs["input_ids"]),
        "attention_modules": attention_modules,
        "attention_outputs": attention_outputs,
        "mlp_modules": mlp_modules,
        "mlp_outputs": mlp_outputs,
        "other_modules": other_modules,
        "other_outputs": other_outputs,
        "norm_parameters": norm_parameters,
        "norm_data": norm_data,
        "logit_lens_parameter": logit_lens_parameter
    }
    
    print(f"Captured {len(captured)} module outputs using PyVene")
    return result


def logit_lens_transformation(mlp_output: Any, norm_data: List[Any], model, logit_lens_parameter: str, tokenizer) -> List[Tuple[str, float]]:
    """
    Transform MLP/layer output to top 3 token probabilities using logit lens.
    Uses model's own output head for correct projection to vocabulary space.
    
    Note: Assumes the input hidden states already include any necessary normalization
    (e.g., in GPT-2, hidden_states[-1] already includes ln_f; in LLaMA, you may need
    to apply model.norm separately for intermediate layers).
    
    Args:
        mlp_output: Hidden state from any layer (from execute_forward_pass)
        norm_data: Not used - kept for backward compatibility  
        model: HuggingFace model with get_output_embeddings() method
        logit_lens_parameter: Not used - kept for backward compatibility
        tokenizer: Model tokenizer for token decoding
    
    Returns:
        List of (token_string, probability) tuples for top 3 tokens
    """
    # Convert to tensor if needed
    if not isinstance(mlp_output, torch.Tensor):
        mlp_output = torch.tensor(mlp_output)
    
    with torch.no_grad():
        # Ensure correct shape [batch, seq_len, hidden_dim]
        if len(mlp_output.shape) == 4:
            mlp_output = mlp_output.squeeze(0)
        
        # Project to vocabulary space using model's output head
        # This uses the correct tied/untied embeddings automatically
        lm_head = model.get_output_embeddings()
        logits = lm_head(mlp_output)
        
        # Get probabilities for last token (next token prediction)
        probs = F.softmax(logits, dim=-1)
        last_token_probs = probs[0, -1, :]  # [vocab_size]
        
        # Get top 3 tokens
        top_probs, top_indices = torch.topk(last_token_probs, 3)
        
        # Convert to token strings
        result = []
        for i in range(3):
            token_id = top_indices[i].item()
            probability = top_probs[i].item()
            token_str = tokenizer.decode([token_id], skip_special_tokens=False)
            result.append((token_str, probability))
        
        return result


def token_to_color(token: str) -> str:
    """Convert token to consistent color using hash."""
    import hashlib
    hash_val = int(hashlib.md5(token.encode()).hexdigest()[:6], 16)
    hue = hash_val % 360
    return f'hsl({hue}, 70%, 50%)'


def format_data_for_cytoscape(activation_data: Dict[str, Any], model, tokenizer) -> List[Dict[str, Any]]:
    """
    Convert activation data to Cytoscape format with nodes (layers) and edges (top-3 tokens).
    """
    elements = []
    mlp_modules = activation_data.get('mlp_modules', [])
    if not mlp_modules:
        print("DEBUG: No mlp_modules found, returning empty elements")
        return elements
    
    # Extract and sort layers
    layer_info = [(int(re.findall(r'\d+', name)[0]), name) for name in mlp_modules if re.findall(r'\d+', name)]
    layer_info.sort()
    print(f"DEBUG: layer_info = {layer_info}")
    
    # Create nodes with positioning and top token stats
    print(f"DEBUG: Creating {len(layer_info)} nodes...")
    for i, (layer_num, module_name) in enumerate(layer_info):
        # Get top token for this layer for node label
        mlp_output = activation_data['mlp_outputs'][module_name]['output']
        norm_data = activation_data.get('norm_data', [])
        logit_lens_param = activation_data.get('logit_lens_parameter')
        
        node_label = f'L{layer_num}'
        print(f"DEBUG: Processing layer {layer_num} ({module_name})")
        if logit_lens_param:
            try:
                print(f"DEBUG: Computing logit lens for layer {layer_num}...")
                top_tokens = logit_lens_transformation(mlp_output, norm_data, model, logit_lens_param, tokenizer)
                print(f"DEBUG: Layer {layer_num} top_tokens = {top_tokens}")
                if top_tokens:
                    token, prob = top_tokens[0]
                    node_label = f'L{layer_num}\n{token[:6]}\n{prob:.2f}'
                    print(f"DEBUG: Layer {layer_num} node_label = {node_label}")
            except Exception as e:
                print(f"Warning: Could not compute logit lens for layer {layer_num}: {e}")
        
        node_data = {
            'data': {
                'id': f'layer_{layer_num}',
                'label': node_label,
                'layer_num': layer_num,
                'module_name': module_name
            },
            'position': {'x': i * 120 + 60, 'y': 200}  # Horizontal layout
        }
        elements.append(node_data)
        print(f"DEBUG: Added node for layer {layer_num}")
    
    # Create edges for top-3 tokens between consecutive layers
    print(f"DEBUG: Creating edges between {len(layer_info)} layers...")
    print(f"DEBUG: Will create {len(layer_info) - 1} edge groups")
    
    for i in range(len(layer_info) - 1):
        current_layer_num, current_module = layer_info[i]
        next_layer_num, _ = layer_info[i + 1]
        
        print(f"DEBUG: Creating edges from layer {current_layer_num} to {next_layer_num}")
        print(f"DEBUG: Using module {current_module}")
        
        mlp_output = activation_data['mlp_outputs'][current_module]['output']
        norm_data = activation_data.get('norm_data', [])
        logit_lens_param = activation_data.get('logit_lens_parameter')
        
        print(f"DEBUG: logit_lens_param = {logit_lens_param}")
        
        if logit_lens_param:
            try:
                print(f"DEBUG: Computing logit lens for edge {current_layer_num}->{next_layer_num}...")
                top_tokens = logit_lens_transformation(mlp_output, norm_data, model, logit_lens_param, tokenizer)
                print(f"DEBUG: Edge {current_layer_num}->{next_layer_num} top_tokens = {top_tokens}")
                
                for rank, (token, prob) in enumerate(top_tokens):
                    edge_data = {
                        'data': {
                            'id': f'edge_{current_layer_num}_{next_layer_num}_{rank}',
                            'source': f'layer_{current_layer_num}',
                            'target': f'layer_{next_layer_num}',
                            'token': token,
                            'probability': prob,
                            'width': max(2, prob * 10),
                            'opacity': max(0.3, prob),
                            'color': token_to_color(token)
                        }
                    }
                    elements.append(edge_data)
                    print(f"DEBUG: Added edge {current_layer_num}->{next_layer_num} rank {rank}: token='{token}', prob={prob:.4f}")
            except Exception as e:
                print(f"Warning: Could not compute logit lens for edge {current_layer_num}->{next_layer_num}: {e}")
                # Simple fallback edge
                fallback_edge = {
                    'data': {
                        'id': f'edge_{current_layer_num}_{next_layer_num}',
                        'source': f'layer_{current_layer_num}',
                        'target': f'layer_{next_layer_num}',
                        'width': 2, 'opacity': 0.5, 'color': '#ccc'
                    }
                }
                elements.append(fallback_edge)
                print(f"DEBUG: Added fallback edge {current_layer_num}->{next_layer_num}")
        else:
            print(f"DEBUG: No logit_lens_param, skipping edges for {current_layer_num}->{next_layer_num}")
    
    print(f"DEBUG: Total elements created: {len(elements)}")
    nodes = [e for e in elements if 'source' not in e['data']]
    edges = [e for e in elements if 'source' in e['data']]
    print(f"DEBUG: Nodes: {len(nodes)}, Edges: {len(edges)}")
    print(f"=== DEBUG: format_data_for_cytoscape END ===\n")
    
    return elements


def generate_bertviz_html(activation_data: Dict[str, Any], layer_index: int, view_type: str = 'full') -> str:
    """
    Generate BertViz attention visualization HTML for a specific layer.
    
    Args:
        activation_data: Output from execute_forward_pass
        layer_index: Index of layer to visualize
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
            return f"<p>No attention data available for layer {layer_index}</p>"
        
        # Find attention module for the specified layer
        target_module = None
        for module_name in attention_outputs.keys():
            numbers = re.findall(r'\d+', module_name)
            if numbers and int(numbers[0]) == layer_index:
                target_module = module_name
                break
        
        if not target_module:
            return f"<p>Layer {layer_index} not found in attention data</p>"
        
        # Get attention weights (element 1 of the output tuple)
        attention_output = attention_outputs[target_module]['output']
        if not isinstance(attention_output, list) or len(attention_output) < 2:
            return f"<p>Invalid attention format for layer {layer_index}</p>"
        
        attention_weights = torch.tensor(attention_output[1])  # [batch, heads, seq, seq]
        
        # Get tokens
        input_ids = torch.tensor(activation_data['input_ids'])
        model_name = activation_data.get('model', 'unknown')
        
        # Load tokenizer and convert to tokens
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        raw_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        tokens = [token.replace('Ġ', ' ') if token.startswith('Ġ') else token for token in raw_tokens]
        
        # Generate visualization based on view_type
        if view_type == 'mini':
            # Mini version: simplified HTML preview
            return f"""
            <div style="padding:10px; border:1px solid #ccc; border-radius:5px;">
                <h4>Layer {layer_index} Attention Preview</h4>
                <p><strong>Tokens:</strong> {' '.join(tokens[:8])}{'...' if len(tokens) > 8 else ''}</p>
                <p><strong>Attention Shape:</strong> {list(attention_weights.shape)}</p>
                <p><em>Click for full visualization</em></p>
            </div>
            """
        else:
            # Full version: complete bertviz visualization
            attentions = (attention_weights,)  # Single layer tuple
            html_result = head_view(attentions, tokens, html_action='return')
            return html_result.data if hasattr(html_result, 'data') else str(html_result)
            
    except Exception as e:
        return f"<p>Error generating visualization: {str(e)}</p>"