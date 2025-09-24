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
from transformers import AutoTokenizer, AutoModel


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
    """Extract MLP outputs for each layer from the captured data using new format."""
    mlp_modules = data['mlp_modules']
    mlp_outputs_data = data['mlp_outputs']
    
    print(f"Extracting MLP outputs from {len(mlp_modules)} layers...")
    
    mlp_outputs = []
    for module_name in mlp_modules:
        if module_name not in mlp_outputs_data:
            raise ValueError(f"Module {module_name} not found in mlp_outputs data")
        
        output_data = mlp_outputs_data[module_name]['output']
        tensor = torch.tensor(output_data)
        mlp_outputs.append(tensor)
        print(f"  {module_name}: shape {tensor.shape}")
    
    return mlp_outputs


def extract_norm_weights(data: Dict[str, Any]) -> torch.Tensor:
    """Extract normalization weights from captured norm_parameter data."""
    print(f"Extracting normalization weights...")
    
    # Extract normalization weights from norm_parameter
    if 'norm_parameter' not in data:
        raise ValueError("norm_parameter not found in captured data")
    
    norm_weight = torch.tensor(data['norm_parameter'])
    print(f"  Norm weight shape: {norm_weight.shape}")
    print(f"  Using RMSNorm (no bias)")
    
    return norm_weight


def load_logit_lens_weights(model_name: str, logit_lens_parameter: str) -> torch.Tensor:
    """Dynamically load the logit lens weights from the model."""
    print(f"Loading logit lens weights from {logit_lens_parameter}...")
    
    try:
        # Load the model to get the embedding weights
        model = AutoModel.from_pretrained(model_name)
        
        # Extract the parameter by name (e.g., "model.embed_tokens.weight")
        # For AutoModel, we need to skip the first 'model' part as it's already the model instance
        param_parts = logit_lens_parameter.split('.')
        if param_parts[0] == 'model':
            param_parts = param_parts[1:]  # Skip the 'model' prefix
        
        param_tensor = model
        for part in param_parts:
            param_tensor = getattr(param_tensor, part)
        
        # The embedding weights are typically [vocab_size, hidden_dim]
        # For logit lens, we need them as [vocab_size, hidden_dim]
        logit_lens_weight = param_tensor.detach().clone()
        print(f"  Loaded logit lens weights: {logit_lens_weight.shape}")
        
        return logit_lens_weight
        
    except Exception as e:
        print(f"  Error loading logit lens weights: {e}")
        print(f"  Available attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
        raise ValueError(f"Could not load {logit_lens_parameter} from model {model_name}")


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
    logit_lens_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    tokenizer: AutoTokenizer,
    top_k: int = 3
) -> List[List[Tuple[str, float]]]:
    """Apply logit lens using actual captured weights."""
    results = []
    
    print(f"\nApplying logit lens to {len(mlp_outputs)} layers...")
    print(f"Using logit lens weight shape: {logit_lens_weight.shape}")
    print(f"Using norm weight shape: {norm_weight.shape}")
    
    with torch.no_grad():
        for layer_idx, mlp_output in enumerate(mlp_outputs):
            # MLP output shape: [batch_size, seq_len, hidden_dim]
            if len(mlp_output.shape) == 4:
                mlp_output = mlp_output.squeeze(0)
            
            # Apply normalization using captured weights (RMSNorm, no bias)
            normalized = apply_normalization(mlp_output, norm_weight, None)
            
            # Project to vocabulary space: hidden_states @ logit_lens_weight.T
            logits = torch.matmul(normalized, logit_lens_weight.T)
            
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
    parser.add_argument("--input", type=str, default="activations.json",
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
        
        # Extract MLP outputs using new format
        mlp_outputs = extract_mlp_outputs(data)
        if not mlp_outputs:
            raise ValueError("No MLP outputs found in the data!")
        
        # Extract normalization weights from norm_parameter
        norm_weight = extract_norm_weights(data)
        
        # Load logit lens weights dynamically from model
        logit_lens_parameter = data['logit_lens_parameter']
        logit_lens_weight = load_logit_lens_weights(model_name, logit_lens_parameter)
        
        # Apply logit lens
        results = apply_logit_lens(mlp_outputs, logit_lens_weight, norm_weight, tokenizer, args.top_k)
        
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
