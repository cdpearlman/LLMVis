
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional

def compute_kl_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor) -> List[float]:
    """
    Compute KL Divergence KL(P || Q) for each position in the sequence.
    P is the reference distribution (logits_p), Q is the ablated distribution (logits_q).
    
    Args:
        logits_p: Reference logits [batch, seq_len, vocab_size]
        logits_q: Ablated logits [batch, seq_len, vocab_size]
        
    Returns:
        List of KL divergence values for each position.
    """
    with torch.no_grad():
        # Ensure batch size 1 or handle appropriately
        if logits_p.dim() == 3:
            logits_p = logits_p.squeeze(0)
        if logits_q.dim() == 3:
            logits_q = logits_q.squeeze(0)
            
        # P = softmax(logits_p)
        # Q = softmax(logits_q)
        # KL(P||Q) = sum(P * (log P - log Q))
        
        # Use log_softmax for stability
        log_probs_p = F.log_softmax(logits_p, dim=-1)
        log_probs_q = F.log_softmax(logits_q, dim=-1)
        probs_p = torch.exp(log_probs_p)
        
        # Element-wise KL
        kl_divs = torch.sum(probs_p * (log_probs_p - log_probs_q), dim=-1)
        
        return kl_divs.tolist()

def score_sequence(model, tokenizer, text: str) -> float:
    """
    Compute the total log probability (score) of a text sequence.
    
    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        text: The sequence to score
        
    Returns:
        Total log probability.
    """
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits # [1, seq_len, vocab_size]
        
        # We want P(token_i | tokens_<i)
        # The logits at position i-1 predict position i
        
        # Shift logits and labels
        shift_logits = logits[0, :-1, :].contiguous()
        shift_labels = input_ids[0, 1:].contiguous()
        
        # Helper to pick specific token probabilities
        # log_softmax
        log_probs_all = F.log_softmax(shift_logits, dim=-1)
        
        # Gather only the target label log probs
        # gather needs index column vector
        target_log_probs = log_probs_all.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
        
        total_score = target_log_probs.sum().item()
        
        return total_score

def get_token_probability_deltas(logits_ref: torch.Tensor, logits_abl: torch.Tensor, input_ids: torch.Tensor) -> List[float]:
    """
    Compute the change in probability (Prob_abl - Prob_ref) for the actual target tokens.
    
    Args:
        logits_ref: Reference logits
        logits_abl: Ablated logits
        input_ids: The sequence token IDs [1, seq_len]
        
    Returns:
        List of probability deltas for each position (starting from first prediction).
    """
    with torch.no_grad():
        if logits_ref.dim() == 3: logits_ref = logits_ref.squeeze(0)
        if logits_abl.dim() == 3: logits_abl = logits_abl.squeeze(0)
        
        target_ids = input_ids[0, 1:] # Targets are from index 1 onwards
        
        # Probabilities
        probs_ref = F.softmax(logits_ref[:-1], dim=-1) # Predicts 1..N
        probs_abl = F.softmax(logits_abl[:-1], dim=-1)
        
        # Gather target probs
        ref_target_probs = probs_ref.gather(1, target_ids.unsqueeze(1)).squeeze(1)
        abl_target_probs = probs_abl.gather(1, target_ids.unsqueeze(1)).squeeze(1)
        
        deltas = (abl_target_probs - ref_target_probs).tolist()
        
        return deltas
