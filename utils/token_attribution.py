"""
Token Attribution utility using Integrated Gradients.

Provides gradient-based attribution to identify which input tokens
most influenced the model's output prediction.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional


def compute_integrated_gradients(
    model,
    tokenizer,
    text: str,
    target_token_id: Optional[int] = None,
    n_steps: int = 50,
    baseline_type: str = 'pad'
) -> Dict[str, Any]:
    """
    Compute Integrated Gradients attribution for input tokens.
    
    This method computes how much each input token contributes to the model's
    prediction of the target token (or the top predicted token if not specified).
    
    Args:
        model: HuggingFace transformer model
        tokenizer: Tokenizer for the model
        text: Input text to analyze
        target_token_id: Optional specific token ID to compute attribution for.
                        If None, uses the model's top predicted token.
        n_steps: Number of interpolation steps (higher = more accurate, slower)
        baseline_type: Type of baseline embedding ('pad', 'zero', 'mask')
    
    Returns:
        Dict with:
            - 'tokens': List of input token strings
            - 'token_ids': List of input token IDs
            - 'attributions': List of attribution scores (one per token)
            - 'normalized_attributions': Attribution scores normalized to [0, 1]
            - 'target_token': The token being attributed (string)
            - 'target_token_id': The token ID being attributed
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    # Get embedding layer
    if hasattr(model, 'transformer'):
        # GPT-2 style
        embedding_layer = model.transformer.wte
    elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        # LLaMA/Qwen style
        embedding_layer = model.model.embed_tokens
    else:
        raise ValueError("Could not find embedding layer in model")
    
    # Get input embeddings
    input_embeds = embedding_layer(input_ids)  # [1, seq_len, hidden_dim]
    
    # Create baseline embeddings
    if baseline_type == 'pad':
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        baseline_ids = torch.full_like(input_ids, pad_token_id)
        baseline_embeds = embedding_layer(baseline_ids)
    elif baseline_type == 'zero':
        baseline_embeds = torch.zeros_like(input_embeds)
    else:  # 'mask'
        mask_token_id = getattr(tokenizer, 'mask_token_id', tokenizer.unk_token_id or 0)
        baseline_ids = torch.full_like(input_ids, mask_token_id)
        baseline_embeds = embedding_layer(baseline_ids)
    
    # If no target specified, get the model's top prediction
    if target_token_id is None:
        with torch.no_grad():
            outputs = model(inputs_embeds=input_embeds)
            logits = outputs.logits[0, -1, :]  # [vocab_size]
            target_token_id = logits.argmax().item()
    
    target_token = tokenizer.decode([target_token_id])
    
    # Compute integrated gradients
    # We interpolate between baseline and input embeddings
    scaled_inputs = []
    for step in range(n_steps + 1):
        alpha = step / n_steps
        scaled_input = baseline_embeds + alpha * (input_embeds - baseline_embeds)
        scaled_inputs.append(scaled_input)
    
    # Stack all scaled inputs
    scaled_inputs = torch.cat(scaled_inputs, dim=0)  # [n_steps+1, seq_len, hidden_dim]
    
    # Enable gradients
    scaled_inputs.requires_grad_(True)
    
    # Forward pass for all scaled inputs
    # Process in batches if memory is a concern
    batch_size = min(n_steps + 1, 10)  # Process 10 at a time
    all_grads = []
    
    for i in range(0, n_steps + 1, batch_size):
        batch_inputs = scaled_inputs[i:i + batch_size]
        batch_inputs = batch_inputs.detach().requires_grad_(True)
        
        outputs = model(inputs_embeds=batch_inputs)
        # Get logits for the target token at the last position
        target_logits = outputs.logits[:, -1, target_token_id]  # [batch_size]
        
        # Sum and backprop
        target_logits.sum().backward()
        
        # Collect gradients
        all_grads.append(batch_inputs.grad.detach())
    
    # Concatenate all gradients
    gradients = torch.cat(all_grads, dim=0)  # [n_steps+1, seq_len, hidden_dim]
    
    # Average gradients (Riemann sum approximation)
    avg_gradients = gradients.mean(dim=0)  # [seq_len, hidden_dim]
    
    # Compute integrated gradients: (input - baseline) * avg_gradient
    # Then sum over hidden dimension to get per-token attribution
    delta = (input_embeds - baseline_embeds).squeeze(0)  # [seq_len, hidden_dim]
    attributions = (delta * avg_gradients).sum(dim=-1)  # [seq_len]
    
    # Convert to list
    attributions_list = attributions.detach().cpu().tolist()
    
    # Normalize to [0, 1] for visualization
    attr_abs = [abs(a) for a in attributions_list]
    max_attr = max(attr_abs) if attr_abs else 1.0
    normalized = [a / max_attr if max_attr > 0 else 0 for a in attr_abs]
    
    # Get token strings
    tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]
    
    return {
        'tokens': tokens,
        'token_ids': input_ids[0].tolist(),
        'attributions': attributions_list,
        'normalized_attributions': normalized,
        'target_token': target_token,
        'target_token_id': target_token_id
    }


def compute_simple_gradient_attribution(
    model,
    tokenizer,
    text: str,
    target_token_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute simple gradient-based attribution (faster than Integrated Gradients).
    
    This is a simpler approach that just computes the gradient of the output
    with respect to the input embeddings in a single pass.
    
    Args:
        model: HuggingFace transformer model
        tokenizer: Tokenizer for the model
        text: Input text to analyze
        target_token_id: Optional specific token ID to compute attribution for
    
    Returns:
        Dict with attribution information
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    # Get embedding layer
    if hasattr(model, 'transformer'):
        embedding_layer = model.transformer.wte
    elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        embedding_layer = model.model.embed_tokens
    else:
        raise ValueError("Could not find embedding layer in model")
    
    # Get input embeddings and enable gradients
    input_embeds = embedding_layer(input_ids)
    input_embeds = input_embeds.detach().requires_grad_(True)
    
    # Forward pass
    outputs = model(inputs_embeds=input_embeds)
    logits = outputs.logits[0, -1, :]  # [vocab_size]
    
    # If no target specified, use top prediction
    if target_token_id is None:
        target_token_id = logits.argmax().item()
    
    target_token = tokenizer.decode([target_token_id])
    
    # Backprop from target logit
    target_logit = logits[target_token_id]
    target_logit.backward()
    
    # Get gradients and compute attribution (L2 norm over hidden dim)
    gradients = input_embeds.grad.squeeze(0)  # [seq_len, hidden_dim]
    attributions = gradients.norm(dim=-1)  # [seq_len]
    
    # Convert to list
    attributions_list = attributions.detach().cpu().tolist()
    
    # Normalize
    max_attr = max(attributions_list) if attributions_list else 1.0
    normalized = [a / max_attr if max_attr > 0 else 0 for a in attributions_list]
    
    # Get token strings
    tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]
    
    return {
        'tokens': tokens,
        'token_ids': input_ids[0].tolist(),
        'attributions': attributions_list,
        'normalized_attributions': normalized,
        'target_token': target_token,
        'target_token_id': target_token_id
    }


def create_attribution_visualization_data(attribution_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Format attribution results for visualization.
    
    Args:
        attribution_result: Output from compute_integrated_gradients or compute_simple_gradient_attribution
    
    Returns:
        List of dicts with token info and color intensity for visualization
    """
    tokens = attribution_result['tokens']
    normalized = attribution_result['normalized_attributions']
    raw = attribution_result['attributions']
    
    viz_data = []
    for i, (tok, norm, raw_val) in enumerate(zip(tokens, normalized, raw)):
        # Map normalized value to color intensity (0 = white, 1 = deep color)
        # Use a blue-to-red scale where positive = red, negative = blue
        if raw_val >= 0:
            r = int(255)
            g = int(255 * (1 - norm * 0.7))
            b = int(255 * (1 - norm * 0.7))
        else:
            r = int(255 * (1 - norm * 0.7))
            g = int(255 * (1 - norm * 0.7))
            b = int(255)
        
        viz_data.append({
            'token': tok,
            'index': i,
            'attribution': raw_val,
            'normalized': norm,
            'color': f'rgb({r},{g},{b})',
            'text_color': '#000000' if norm < 0.5 else '#ffffff'
        })
    
    return viz_data

