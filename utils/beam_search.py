"""
Beam search utility for text generation and sequence analysis.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any
import numpy as np
from utils.model_patterns import get_norm_layer_from_parameter

def perform_beam_search(model, tokenizer, prompt: str, beam_width: int = 3, max_new_tokens: int = 10) -> List[Dict[str, Any]]:
    """
    Perform beam search generation.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt: Input text
        beam_width: Number of beams to keep
        max_new_tokens: Maximum tokens to generate
        
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
