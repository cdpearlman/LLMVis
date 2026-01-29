"""
Beam search utility for text generation and sequence analysis.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from utils.model_patterns import get_norm_layer_from_parameter
import re

def _make_head_ablation_hook(head_indices: List[int], num_heads: int):
    """
    Create a hook that zeros out specific attention heads.
    """
    def hook(module, input, output):
        # output is typically (attn_output, past_key_values, attn_weights) or just attn_output
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
            
        # Check if we need to convert from list/tuple (some models might behave oddly)
        if not isinstance(hidden_states, torch.Tensor):
            return output # Safety skip
            
        batch_size, seq_len, hidden_dim = hidden_states.shape
        head_dim = hidden_dim // num_heads
        
        # Reshape to [batch, seq_len, num_heads, head_dim]
        # We need to clone to modify safetly
        hidden_states_reshaped = hidden_states.view(batch_size, seq_len, num_heads, head_dim).clone()
        
        # Zero out specified heads
        for head_idx in head_indices:
            if 0 <= head_idx < num_heads:
                hidden_states_reshaped[:, :, head_idx, :] = 0.0
        
        # Reshape back
        ablated_hidden = hidden_states_reshaped.view(batch_size, seq_len, hidden_dim)
        
        if isinstance(output, tuple):
            return (ablated_hidden,) + output[1:]
        return ablated_hidden
    return hook

def _apply_ablation_hooks(model, ablation_config: Dict[int, List[int]]) -> List[Any]:
    """
    Apply ablation hooks to the model.
    Returns a list of hooks to remove later.
    """
    hooks = []
    if not ablation_config:
        return hooks
        
    num_heads = getattr(model.config, 'num_attention_heads', None)
    if not num_heads:
        print("Warning: Could not determine num_attention_heads for ablation.")
        return hooks

    # Heuristic to find attention modules
    # We iterate named_modules and try to match layer numbers
    name_to_module = dict(model.named_modules())
    
    for layer_num, head_indices in ablation_config.items():
        if not head_indices:
            continue
            
        target_module = None
        target_name = None
        
        # Try to find the attention module for this layer
        # Common patterns:
        # GPT-2: transformer.h.{i}.attn
        # Llama/Mistral: model.layers.{i}.self_attn
        # BERT: bert.encoder.layer.{i}.attention.self
        
        candidates = []
        for name, mod in name_to_module.items():
            # Check if layer number matches segments
            parts = name.split('.')
            if str(layer_num) in parts:
                # Check for attention keywords
                if 'attn' in name or 'attention' in name:
                    # Exclude known sub-components like projections
                    if not any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'out_proj', 'dense', 'LayerNorm', 'ln_']):
                        candidates.append((name, mod))
        
        # Pick the most likely candidate (shortest name usually top level attn block)
        if candidates:
            # Sort by length
            candidates.sort(key=lambda x: len(x[0]))
            target_name, target_module = candidates[0]
            
            # Register hook
            # print(f"Applying ablation to {target_name} for Layer {layer_num}, Heads {head_indices}")
            hooks.append(target_module.register_forward_hook(
                _make_head_ablation_hook(head_indices, num_heads)
            ))
        else:
            print(f"Warning: Could not find attention module for layer {layer_num} in beam search.")
            
    return hooks

def perform_beam_search(model, tokenizer, prompt: str, beam_width: int = 3, max_new_tokens: int = 10, 
                       ablation_config: Optional[Dict[int, List[int]]] = None) -> List[Dict[str, Any]]:
    """
    Perform beam search generation.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt: Input text
        beam_width: Number of beams to keep
        max_new_tokens: Maximum tokens to generate
        ablation_config: Optional dict {layer_num: [head_indices]} to zero out heads
        
    Returns:
        List of dicts containing:
        - 'text': Generated text
        - 'score': Final log probability score
        - 'sequence_ids': Token IDs of the sequence
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    device = model.device
    input_ids = input_ids.to(device)
    
    # Apply ablation hooks if configured
    hooks = _apply_ablation_hooks(model, ablation_config)
    
    try:
        # List of (sequence_tensor, score, cumulative_log_prob)
        # score is usually normalized by length or just cumulative log prob
        # We'll use cumulative log prob for simplicity
        beams = [(input_ids, 0.0)]
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                candidates = []
                
                for seq, score in beams:
                    # Get next token probabilities
                    outputs = model(seq)
                    next_token_logits = outputs.logits[:, -1, :]
                    
                    # Apply log_softmax to get log probabilities
                    next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)
                    
                    # Get top-k candidates for this beam
                    top_log_probs, top_indices = torch.topk(next_token_log_probs, beam_width, dim=-1)
                    
                    for i in range(beam_width):
                        token_id = top_indices[0, i].item()
                        log_prob = top_log_probs[0, i].item()
                        
                        new_seq = torch.cat([seq, torch.tensor([[token_id]], device=device)], dim=1)
                        new_score = score + log_prob
                        candidates.append((new_seq, new_score))
                
                # Select top-k globally
                candidates.sort(key=lambda x: x[1], reverse=True)
                beams = candidates[:beam_width]
                
                # Optional: Stop if all beams end with EOS (simplified here)
        
        results = []
        for seq, score in beams:
            text = tokenizer.decode(seq[0], skip_special_tokens=True)
            results.append({
                'text': text,
                'score': score,
                'sequence_ids': seq[0].tolist()
            })
        
        return results
        
    finally:
        # Ensure hooks are removed even if error occurs
        for hook in hooks:
            hook.remove()

def compute_sequence_trajectory(activation_data: Dict[str, Any], model, tokenizer) -> Dict[int, List[float]]:
    """
    Compute the trajectory of the sequence score across layers.
    
    For each layer, calculates the probability assigned to the *actual* next token 
    at each step of the sequence.
    
    Args:
        activation_data: Data from execute_forward_pass (must contain block_outputs for all layers)
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        
    Returns:
        Dict mapping layer_num -> list of scores (one per step in the generated sequence)
    """
    if not activation_data or 'block_outputs' not in activation_data:
        return {}
        
    # Extract layer outputs
    block_outputs = activation_data['block_outputs']
    input_ids = activation_data['input_ids']
    
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids)
    
    # Identify tokens: input_ids shape is [1, seq_len]
    # The "generated" part starts after the prompt, but here we likely have the full sequence.
    # We want to evaluate P(token_t | tokens_<t) for the whole sequence or just the new part?
    # Usually, we visualize the whole sequence.
    
    # We need the logits from each layer
    # block_outputs keys are like "model.layers.0", "model.layers.1", etc.
    
    # Sort layers
    import re
    layer_info = sorted(
        [(int(re.findall(r'\d+', name)[0]), name) 
         for name in block_outputs.keys() if re.findall(r'\d+', name)]
    )
    
    # Get norm parameter for logit lens
    norm_params = activation_data.get('norm_parameters', [])
    norm_parameter = norm_params[0] if norm_params else None
    final_norm = get_norm_layer_from_parameter(model, norm_parameter)
    lm_head = model.get_output_embeddings()
    
    trajectories = {}
    
    # We only care about predictions for positions 0 to N-1 (predicting 1 to N)
    target_ids = input_ids[0, 1:] 
    
    with torch.no_grad():
        for layer_num, module_name in layer_info:
            output_data = block_outputs[module_name]['output']
            
            # Convert to tensor [batch, seq_len, hidden_dim]
            hidden = torch.tensor(output_data) if not isinstance(output_data, torch.Tensor) else output_data
            if hidden.dim() == 4: # PyVene sometimes returns [1, 1, seq_len, dim] ? No usually [1, seq, dim]
                # If shape is weird, adjust
                pass
            
            # Ensure batch dim
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)
                
            # Apply final norm
            if final_norm is not None:
                hidden = final_norm(hidden)
            
            # Project to logits
            logits = lm_head(hidden) # [batch, seq_len, vocab_size]
            
            # We want log probs of the *next* token
            # Logits at pos t predict token at t+1
            # So we take logits at [0, :-1, :] and gather targets [0, 1:]
            
            shift_logits = logits[0, :-1, :]
            log_probs = F.log_softmax(shift_logits, dim=-1)
            
            # Gather log probs of the actual target tokens
            # target_ids shape [seq_len-1]
            target_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
            
            trajectories[layer_num] = target_log_probs.tolist()
            
    return trajectories
