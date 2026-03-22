"""
Beam search utility for text generation and sequence analysis.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
import re

_OUTPUT_PROJ_NAMES = ['c_proj', 'o_proj', 'out_proj', 'dense']


def _find_output_proj_submodule(attn_module, attn_module_name: str = ""):
    """Find output projection submodule within an attention module.
    Returns (name, submodule). Raises ValueError if not found."""
    children = dict(attn_module.named_children())
    for proj_name in _OUTPUT_PROJ_NAMES:
        if proj_name in children:
            return proj_name, children[proj_name]
    raise ValueError(
        f"No output projection found in {attn_module_name or type(attn_module).__name__}. "
        f"Children: {list(children.keys())}. Expected one of: {_OUTPUT_PROJ_NAMES}"
    )


def _make_head_ablation_pre_hook(head_indices: List[int], num_heads: int):
    """
    Create a pre-hook on the output projection that zeros out specific attention
    heads BEFORE projection mixing, where per-head dims are still separable.
    """
    def pre_hook(module, args):
        x = args[0].clone()
        head_dim = x.shape[-1] // num_heads
        for head_idx in head_indices:
            if 0 <= head_idx < num_heads:
                start = head_idx * head_dim
                end = (head_idx + 1) * head_dim
                x[:, :, start:end] = 0.0
        return (x,)
    return pre_hook

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

            # Register pre-hook on output projection (before heads are mixed)
            try:
                _, proj_mod = _find_output_proj_submodule(target_module, target_name)
                hooks.append(proj_mod.register_forward_pre_hook(
                    _make_head_ablation_pre_hook(head_indices, num_heads)
                ))
            except ValueError as e:
                print(f"Warning: {e}")
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
