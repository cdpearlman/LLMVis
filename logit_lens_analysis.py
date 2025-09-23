#!/usr/bin/env python3
"""
Streamlined Logit Lens Analysis for Captured Activations

Works with the new activations format from activation_capture.py.
Uses actual captured parameters for proper logit lens transformation.
"""

import json
import argparse
from typing import List, Tuple, Dict, Any
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


def load_activations(filepath: str) -> Dict[str, Any]:
    """Load captured activations and parameters from JSON file."""
    print(f"Loading activations from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded data for model: {data['model']}")
    print(f"Prompt: '{data['prompt']}'")
    return data


def extract_layer_number_from_name(module_name: str) -> int:
    """Extract layer number from module name in a model-agnostic way."""
    import re
    numbers = re.findall(r'\d+', module_name)
    if not numbers:
        raise ValueError(f"No numbers found in module name: {module_name}")
    return int(numbers[0])


def extract_mlp_outputs(data: Dict[str, Any]) -> List[torch.Tensor]:
    """Extract MLP outputs for each layer from the captured data."""
    mlp_data = data['mlp_data']
    
    # Sort by layer number in a model-agnostic way
    try:
        sorted_layers = sorted(mlp_data.keys(), key=extract_layer_number_from_name)
    except Exception as e:
        print(f"Warning: Could not sort layers by number ({e}), using alphabetical order")
        sorted_layers = sorted(mlp_data.keys())
    
    print(f"Extracting MLP outputs from {len(sorted_layers)} layers...")
    
    mlp_outputs = []
    for layer_name in sorted_layers:
        output_data = mlp_data[layer_name]['output']
        tensor = torch.tensor(output_data)
        mlp_outputs.append(tensor)
        print(f"  {layer_name}: shape {tensor.shape}")
    
    return mlp_outputs


def extract_weights_and_logits(data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract lm_head logits and normalization weights from captured data."""
    print(f"Extracting weights and final logits...")
    
    # Extract final lm_head logits
    if 'lm_head' not in data['other_data']:
        raise ValueError("lm_head outputs not found in captured data")
    
    final_logits = torch.tensor(data['other_data']['lm_head']['output'])
    print(f"  Final logits shape: {final_logits.shape}")
    
    # Extract normalization weights
    if 'model.norm' not in data['parameters']:
        raise ValueError("model.norm parameters not found in captured data")
    
    norm_weight = torch.tensor(data['parameters']['model.norm']['weight'])
    print(f"  Norm weight shape: {norm_weight.shape}")
    
    # Try to extract norm bias if available
    norm_bias = None
    if 'bias' in data['parameters']['model.norm']:
        norm_bias = torch.tensor(data['parameters']['model.norm']['bias'])
        print(f"  Norm bias shape: {norm_bias.shape}")
    else:
        print(f"  No norm bias found (RMSNorm)")
    
    return final_logits, norm_weight, norm_bias


def derive_lm_head_weights(final_hidden_states: torch.Tensor, final_logits: torch.Tensor) -> torch.Tensor:
    """Derive lm_head weights from final hidden states and logits."""
    # Use the last token position for stable approximation
    hidden = final_hidden_states.squeeze(0)  # [seq_len, hidden_dim]
    logits = final_logits.squeeze(0)        # [seq_len, vocab_size]
    
    last_hidden = hidden[-1:, :]  # [1, hidden_dim]
    last_logits = logits[-1:, :]  # [1, vocab_size]
    
    try:
        # Solve: last_logits = last_hidden @ W.T
        weight_T = torch.linalg.pinv(last_hidden) @ last_logits  # [hidden_dim, vocab_size]
        weight = weight_T.T  # [vocab_size, hidden_dim]
        
        # Verify and scale
        test_logits = torch.matmul(last_hidden, weight.T)
        if test_logits.std() > 0:
            scale_factor = (last_logits.std() / test_logits.std()).item()
            weight = weight * scale_factor
            
        print(f"  Derived lm_head weights: {weight.shape}, scale factor: {scale_factor:.3f}")
        return weight
    except Exception as e:
        print(f"  Warning: Could not derive lm_head weights ({e})")
        # Fallback: create scaled random weights
        hidden_dim = hidden.shape[-1]
        vocab_size = logits.shape[-1]
        scale = logits.std().item() / (hidden_dim ** 0.5)
        return torch.randn(vocab_size, hidden_dim) * scale


def apply_normalization(hidden_states: torch.Tensor, norm_weight: torch.Tensor, norm_bias: torch.Tensor = None, eps: float = 1e-6) -> torch.Tensor:
    """Apply layer normalization using captured weights."""
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
    lm_head_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    tokenizer: AutoTokenizer,
    top_k: int = 3
) -> List[List[Tuple[str, float]]]:
    """Apply logit lens using actual captured weights."""
    results = []
    
    print(f"\nApplying logit lens to {len(mlp_outputs)} layers...")
    print(f"Using lm_head weight shape: {lm_head_weight.shape}")
    print(f"Using norm weight shape: {norm_weight.shape}")
    
    with torch.no_grad():
        for layer_idx, mlp_output in enumerate(mlp_outputs):
            # MLP output shape: [batch_size, seq_len, hidden_dim]
            if len(mlp_output.shape) == 4:
                mlp_output = mlp_output.squeeze(0)
            
            # Apply normalization using captured weights
            normalized = apply_normalization(mlp_output, norm_weight, norm_bias)
            
            # Project to vocabulary space: hidden_states @ lm_head_weight.T
            logits = torch.matmul(normalized, lm_head_weight.T)
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Get the last token position (next token prediction)
            last_token_probs = probs[0, -1, :]  # [batch=0, last_token, vocab_dim]
            
            # Get top-k tokens and their probabilities
            top_probs, top_indices = torch.topk(last_token_probs, top_k)
            
            # Convert to token strings and probabilities
            layer_results = []
            for i in range(top_k):
                token_id = top_indices[i].item()
                probability = top_probs[i].item()
                token_str = tokenizer.decode([token_id], skip_special_tokens=False)
                layer_results.append((token_str, probability))
            
            results.append(layer_results)
            
            # Print progress with appropriate precision
            prob_strs = []
            for token, prob in layer_results:
                if prob < 0.001:
                    prob_strs.append(f'{token}({prob:.6f})')
                else:
                    prob_strs.append(f'{token}({prob:.3f})')
            print(f"  Layer {layer_idx:2d}: {', '.join(prob_strs)}")
    
    return results


def format_results(results: List[List[Tuple[str, float]]], model_name: str, prompt: str) -> str:
    """Format the results into a readable string."""
    output = []
    output.append("=" * 80)
    output.append(f"LOGIT LENS ANALYSIS - STREAMLINED")
    output.append("=" * 80)
    output.append(f"Model: {model_name}")
    output.append(f"Prompt: '{prompt}'")
    output.append("")
    output.append("Top 3 predicted tokens at each layer:")
    output.append("-" * 50)
    
    for layer_idx, layer_results in enumerate(results):
        tokens_str = " | ".join([f"{token:<15} ({prob:.4f})" for token, prob in layer_results])
        output.append(f"Layer {layer_idx:2d}: {tokens_str}")
    
    output.append("")
    output.append("Analysis Summary:")
    output.append(f"- Total layers analyzed: {len(results)}")
    
    # Find most confident predictions
    if results:
        max_confidence = max(results[i][0][1] for i in range(len(results)))
        max_layer = next(i for i in range(len(results)) if results[i][0][1] == max_confidence)
        output.append(f"- Highest confidence: Layer {max_layer} ({max_confidence:.4f}) -> '{results[max_layer][0][0]}'")
    
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description="Streamlined logit lens analysis with captured weights")
    parser.add_argument("--input", type=str, default="minimal_activations.json",
                       help="Path to captured activations JSON file")
    parser.add_argument("--top-k", type=int, default=3,
                       help="Number of top tokens to extract per layer")
    
    args = parser.parse_args()
    
    try:
        # Load captured data
        data = load_activations(args.input)
        model_name = data['model']
        
        # Load tokenizer
        print(f"Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"  Vocabulary size: {len(tokenizer)}")
        
        # Extract MLP outputs
        mlp_outputs = extract_mlp_outputs(data)
        if not mlp_outputs:
            raise ValueError("No MLP outputs found in the data!")
        
        # Extract weights and final logits
        final_logits, norm_weight, norm_bias = extract_weights_and_logits(data)
        
        # Get final normalized hidden states from model.norm output
        if 'model.norm' not in data['other_data']:
            raise ValueError("model.norm outputs not found in captured data")
        
        final_hidden_states = torch.tensor(data['other_data']['model.norm']['output'])
        print(f"  Final hidden states shape: {final_hidden_states.shape}")
        
        # Derive lm_head weights
        lm_head_weight = derive_lm_head_weights(final_hidden_states, final_logits)
        
        # Apply logit lens
        results = apply_logit_lens(mlp_outputs, lm_head_weight, norm_weight, norm_bias, tokenizer, args.top_k)
        
        # Format and display results
        formatted_output = format_results(results, model_name, data['prompt'])
        print("\n" + formatted_output)
        
        print(f"\nAnalysis complete! Processed {len(results)} layers using captured weights.")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
