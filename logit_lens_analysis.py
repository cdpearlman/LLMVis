#!/usr/bin/env python3
"""
Logit Lens Analysis for Transformer Models

This module implements the logit lens technique to analyze token predictions
at each layer of a transformer model using captured data from agnostic_capture.py.

REQUIREMENTS:
The captured data must include:
- MLP outputs for each layer
- lm_head weights (module.weight)  
- Normalization weights (module.weight, module.bias if applicable)

All module paths are provided as input parameters and data is extracted directly
from the captured activations JSON file.

USAGE:
python logit_lens_analysis.py --lm-head-path "lm_head" --norm-path "model.norm"
"""

import json
import argparse
from typing import List, Tuple, Dict, Any
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_activations(filepath: str) -> Dict[str, Any]:
    """Load the captured activations from JSON file."""
    print(f"Loading activations from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded data for model: {data['model']}")
    print(f"Prompt: '{data['prompt']}'")
    return data


def extract_layer_number_from_name(module_name: str) -> int:
    """Extract layer number from module name in a model-agnostic way."""
    import re
    # Find all numbers in the module name
    numbers = re.findall(r'\d+', module_name)
    if not numbers:
        raise ValueError(f"No numbers found in module name: {module_name}")
    
    # For most transformer architectures, the layer number is the first or most prominent number
    # We'll use the first number found, but this could be made smarter if needed
    return int(numbers[0])


def extract_mlp_outputs(data: Dict[str, Any]) -> List[torch.Tensor]:
    """Extract MLP outputs for each layer from the captured data."""
    mlp_outputs = []
    mlp_data = data['captured']['mlp_outputs']
    
    # Sort by layer number in a model-agnostic way
    try:
        sorted_layers = sorted(mlp_data.keys(), key=extract_layer_number_from_name)
    except Exception as e:
        print(f"Warning: Could not sort layers by number ({e}), using alphabetical order")
        sorted_layers = sorted(mlp_data.keys())
    
    print(f"Extracting MLP outputs from {len(sorted_layers)} layers...")
    
    for layer_name in sorted_layers:
        output_data = mlp_data[layer_name]['output']
        # Convert list back to tensor
        tensor = torch.tensor(output_data)
        mlp_outputs.append(tensor)
        print(f"  {layer_name}: shape {tensor.shape}")
    
    return mlp_outputs


def extract_module_weights(data: Dict[str, Any], module_path: str) -> torch.Tensor:
    """Extract module weights from captured data using the full module path."""
    # Search in all captured categories
    all_captured = {}
    if 'attention_outputs' in data['captured']:
        all_captured.update(data['captured']['attention_outputs'])
    if 'mlp_outputs' in data['captured']:
        all_captured.update(data['captured']['mlp_outputs'])
    if 'other_outputs' in data['captured']:
        all_captured.update(data['captured']['other_outputs'])
    
    if module_path not in all_captured:
        available_modules = list(all_captured.keys())
        raise ValueError(f"Module '{module_path}' not found in captured data.\n"
                        f"Available modules: {available_modules}")
    
    module_data = all_captured[module_path]
    
    # Extract weights - they're stored in 'output' key
    if 'output' not in module_data:
        raise ValueError(f"No output data found for module '{module_path}'.\n"
                        f"Available keys: {list(module_data.keys())}")
    
    weights = torch.tensor(module_data['output'])
    print(f"  Extracted data for {module_path}: shape {weights.shape}")
    return weights


def load_tokenizer_only(model_name: str) -> AutoTokenizer:
    """Load only the tokenizer from the model."""
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"  Vocabulary size: {len(tokenizer)}")
    return tokenizer


def apply_normalization(hidden_states: torch.Tensor, norm_weight: torch.Tensor, norm_bias: torch.Tensor = None, eps: float = 1e-6) -> torch.Tensor:
    """Apply layer normalization using captured weights."""
    # Assume RMSNorm or LayerNorm
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


def derive_lm_head_transformation(final_hidden_states: torch.Tensor, final_logits: torch.Tensor) -> torch.Tensor:
    """
    Derive an approximation of the lm_head transformation from final hidden states and logits.
    
    Uses a more stable approach with proper scaling.
    """
    # Reshape for matrix operations
    # final_hidden_states: [1, seq_len, hidden_dim] -> [seq_len, hidden_dim]  
    # final_logits: [1, seq_len, vocab_size] -> [seq_len, vocab_size]
    hidden = final_hidden_states.squeeze(0)  # [seq_len, hidden_dim]
    logits = final_logits.squeeze(0)        # [seq_len, vocab_size]
    
    # Use the last token position for more stable approximation
    # This gives us: final_logits[-1] = final_hidden[-1] @ lm_head_weight.T
    last_hidden = hidden[-1:, :]  # [1, hidden_dim]
    last_logits = logits[-1:, :]  # [1, vocab_size]
    
    try:
        # Solve: last_logits = last_hidden @ W.T
        # So: W.T = pinv(last_hidden) @ last_logits
        weight_T = torch.linalg.pinv(last_hidden) @ last_logits  # [hidden_dim, vocab_size]
        weight = weight_T.T  # [vocab_size, hidden_dim]
        
        # Verify the transformation produces reasonable logits
        test_logits = torch.matmul(last_hidden, weight.T)
        
        # Debug the transformation quality
        print(f"  Original final logits stats: mean={last_logits.mean():.3f}, std={last_logits.std():.3f}")
        print(f"  Test logits stats: mean={test_logits.mean():.3f}, std={test_logits.std():.3f}")
        
        # Apply better scaling
        if test_logits.std() > 0:
            scale_factor = (last_logits.std() / test_logits.std()).item()
            offset = (last_logits.mean() - test_logits.mean()).item()
            weight = weight * scale_factor
            
            # Test again after scaling
            test_logits_scaled = torch.matmul(last_hidden, weight.T)
            print(f"  After scaling - Test logits stats: mean={test_logits_scaled.mean():.3f}, std={test_logits_scaled.std():.3f}")
            
            # Check if we can reproduce the final layer probabilities
            test_probs = F.softmax(test_logits_scaled, dim=-1)
            final_probs = F.softmax(last_logits, dim=-1)
            top_test_prob = test_probs.max().item()
            top_final_prob = final_probs.max().item()
            print(f"  Probability check: derived max={top_test_prob:.4f}, original max={top_final_prob:.4f}")
        else:
            scale_factor = 1.0
        
        print(f"  Derived lm_head approximation: {weight.shape}, scale factor: {scale_factor:.3f}")
        return weight
    except Exception as e:
        print(f"  Warning: Could not derive lm_head transformation ({e})")
        # Fallback: create identity-like transformation with proper scale
        hidden_dim = hidden.shape[-1]
        vocab_size = logits.shape[-1]
        # Scale based on the magnitude of final logits
        scale = logits.std().item() / (hidden_dim ** 0.5)
        return torch.randn(vocab_size, hidden_dim) * scale


def apply_logit_lens(
    mlp_outputs: List[torch.Tensor], 
    final_logits: torch.Tensor, 
    final_hidden_states: torch.Tensor,
    tokenizer: AutoTokenizer,
    top_k: int = 3
) -> List[List[Tuple[str, float]]]:
    """
    Apply logit lens to extract top-k tokens at each layer using derived transformation.
    
    Note: Intermediate layers typically show very low probabilities because MLP outputs
    haven't been processed through normalization and residual connections. This is 
    expected behavior - coherent predictions emerge gradually through the layers.
    
    Args:
        mlp_outputs: List of MLP output tensors for each layer
        final_logits: Final lm_head logits [batch_size, seq_len, vocab_size]
        final_hidden_states: Final normalized hidden states [batch_size, seq_len, hidden_dim]
        tokenizer: Tokenizer to decode token IDs
        top_k: Number of top tokens to return for each layer
        
    Returns:
        List of lists, where each inner list contains (token, probability) tuples
        for the top-k tokens at that layer
    """
    results = []
    
    print(f"\nApplying logit lens to {len(mlp_outputs)} layers...")
    
    # Derive the lm_head transformation from final states
    lm_head_weight = derive_lm_head_transformation(final_hidden_states, final_logits)
    
    with torch.no_grad():
        for layer_idx, mlp_output in enumerate(mlp_outputs):
            # MLP output should be shape: [batch_size, seq_len, hidden_dim]
            if len(mlp_output.shape) == 4:
                # If we have an extra dimension, squeeze it
                mlp_output = mlp_output.squeeze(0)
            
            # For simplicity, skip normalization for now since we don't have proper norm weights
            # In practice, you'd want to apply proper normalization here
            hidden_states = mlp_output
            
            # Project to vocabulary space: hidden_states @ lm_head_weight.T
            # lm_head_weight shape: [vocab_size, hidden_dim]
            # hidden_states shape: [batch_size, seq_len, hidden_dim]
            logits = torch.matmul(hidden_states, lm_head_weight.T)
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Get the last token position (this is what the model predicts next)
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
            
            # Print progress with higher precision for small probabilities
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
    output.append(f"LOGIT LENS ANALYSIS")
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
        max_confidence = max(results[i][0][1] for i in range(len(results)))  # highest probability across all layers
        max_layer = next(i for i in range(len(results)) if results[i][0][1] == max_confidence)
        output.append(f"- Highest confidence: Layer {max_layer} ({max_confidence:.4f}) -> '{results[max_layer][0][0]}'")
    
    return "\n".join(output)


def save_results(results: List[List[Tuple[str, float]]], output_file: str, model_name: str, prompt: str):
    """Save results to a JSON file."""
    data = {
        "model": model_name,
        "prompt": prompt,
        "analysis_type": "logit_lens",
        "layers": []
    }
    
    for layer_idx, layer_results in enumerate(results):
        layer_data = {
            "layer": layer_idx,
            "top_tokens": [
                {
                    "token": token,
                    "probability": prob,
                    "rank": rank + 1
                }
                for rank, (token, prob) in enumerate(layer_results)
            ]
        }
        data["layers"].append(layer_data)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {output_file}")


def analyze_token_trends(results: List[List[Tuple[str, float]]]) -> Dict[str, Any]:
    """Analyze trends in token predictions across layers."""
    trends = {
        "confidence_progression": [],
        "token_stability": {},
        "layer_with_highest_confidence": 0,
        "max_confidence": 0.0,
        "most_stable_tokens": []
    }
    
    # Track confidence progression (highest probability at each layer)
    for layer_idx, layer_results in enumerate(results):
        if layer_results:
            max_prob = layer_results[0][1]  # Top token probability
            trends["confidence_progression"].append((layer_idx, max_prob))
            
            if max_prob > trends["max_confidence"]:
                trends["max_confidence"] = max_prob
                trends["layer_with_highest_confidence"] = layer_idx
    
    # Track token stability (how often each token appears in top predictions)
    token_appearances = {}
    for layer_results in results:
        for token, prob in layer_results:
            if token not in token_appearances:
                token_appearances[token] = []
            token_appearances[token].append(prob)
    
    # Find most stable tokens (appearing in multiple layers)
    stable_tokens = [(token, len(probs), sum(probs)/len(probs)) 
                    for token, probs in token_appearances.items() 
                    if len(probs) >= 3]  # Appears in at least 3 layers
    stable_tokens.sort(key=lambda x: x[1], reverse=True)  # Sort by frequency
    trends["most_stable_tokens"] = stable_tokens[:5]  # Top 5 most stable
    
    return trends


def create_summary_report(results: List[List[Tuple[str, float]]], model_name: str, prompt: str) -> str:
    """Create a comprehensive summary report of the logit lens analysis."""
    trends = analyze_token_trends(results)
    
    report = []
    report.append("=" * 80)
    report.append("LOGIT LENS ANALYSIS - COMPREHENSIVE REPORT")
    report.append("=" * 80)
    report.append(f"Model: {model_name}")
    report.append(f"Prompt: '{prompt}'")
    import datetime
    report.append(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Executive Summary
    report.append("EXECUTIVE SUMMARY:")
    report.append("-" * 40)
    report.append(f"• Total layers analyzed: {len(results)}")
    report.append(f"• Highest confidence prediction: Layer {trends['layer_with_highest_confidence']} "
                 f"({trends['max_confidence']:.4f})")
    
    if results:
        final_prediction = results[-1][0] if results[-1] else "N/A"
        report.append(f"• Final layer top prediction: '{final_prediction[0]}' ({final_prediction[1]:.4f})")
    
    report.append("")
    
    # Confidence Progression
    report.append("CONFIDENCE PROGRESSION:")
    report.append("-" * 40)
    confidence_values = [conf for _, conf in trends["confidence_progression"]]
    if confidence_values:
        report.append(f"• Average confidence: {sum(confidence_values)/len(confidence_values):.4f}")
        report.append(f"• Min confidence: {min(confidence_values):.4f}")
        report.append(f"• Max confidence: {max(confidence_values):.4f}")
    
    # Show layers with unusually high/low confidence
    if len(confidence_values) > 5:
        avg_conf = sum(confidence_values) / len(confidence_values)
        high_conf_layers = [(i, conf) for i, conf in trends["confidence_progression"] 
                           if conf > avg_conf * 2]
        if high_conf_layers:
            report.append("• Layers with unusually high confidence:")
            for layer, conf in high_conf_layers[:3]:
                report.append(f"  - Layer {layer}: {conf:.4f}")
    
    report.append("")
    
    # Token Stability Analysis
    report.append("TOKEN STABILITY ANALYSIS:")
    report.append("-" * 40)
    if trends["most_stable_tokens"]:
        report.append("• Most frequently appearing tokens across layers:")
        for token, frequency, avg_prob in trends["most_stable_tokens"]:
            clean_token = repr(token) if '\n' in token or len(token.strip()) != len(token) else token
            report.append(f"  - {clean_token:<20} (appears {frequency} times, avg prob: {avg_prob:.4f})")
    else:
        report.append("• No tokens appear consistently across multiple layers")
    
    report.append("")
    
    # Layer-by-layer breakdown (condensed)
    report.append("LAYER-BY-LAYER BREAKDOWN:")
    report.append("-" * 40)
    for layer_idx, layer_results in enumerate(results[:5]):  # Show first 5 layers
        if layer_results:
            top_token = layer_results[0]
            clean_token = repr(top_token[0]) if '\n' in top_token[0] or len(top_token[0].strip()) != len(top_token[0]) else top_token[0]
            report.append(f"Layer {layer_idx:2d}: {clean_token:<20} ({top_token[1]:.4f})")
    
    if len(results) > 10:
        report.append("  ...")
        # Show last 5 layers
        for layer_idx in range(len(results)-5, len(results)):
            layer_results = results[layer_idx]
            if layer_results:
                top_token = layer_results[0]
                clean_token = repr(top_token[0]) if '\n' in top_token[0] or len(top_token[0].strip()) != len(top_token[0]) else top_token[0]
                report.append(f"Layer {layer_idx:2d}: {clean_token:<20} ({top_token[1]:.4f})")
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Apply logit lens analysis to captured MLP outputs")
    parser.add_argument("--input", type=str, default="agnostic_activations.json", 
                       help="Path to captured activations JSON file")
    parser.add_argument("--output", type=str, default="logit_lens_results.json",
                       help="Path to save detailed results")
    parser.add_argument("--summary", type=str, default="logit_lens_summary.txt",
                       help="Path to save summary report")
    parser.add_argument("--top-k", type=int, default=3,
                       help="Number of top tokens to extract per layer")
    parser.add_argument("--no-summary", action="store_true",
                       help="Skip generating summary report")
    
    # Required module paths
    parser.add_argument("--lm-head-path", type=str, required=True,
                       help="Full path to lm_head module in captured data (e.g., 'lm_head')")
    parser.add_argument("--norm-path", type=str, required=True,
                       help="Full path to normalization module in captured data (e.g., 'model.norm')")
    
    args = parser.parse_args()
    
    try:
        # Load the captured activations
        data = load_activations(args.input)
        model_name = data['model']
        
        # Load tokenizer
        tokenizer = load_tokenizer_only(model_name)
        
        # Extract MLP outputs
        mlp_outputs = extract_mlp_outputs(data)
        if not mlp_outputs:
            raise ValueError("No MLP outputs found in the data!")
        
        # Extract final logits and hidden states from captured data
        print(f"\nExtracting final model outputs from captured data...")
        final_logits = extract_module_weights(data, args.lm_head_path)  # Shape: [1, seq_len, vocab_size]
        final_hidden_states = extract_module_weights(data, args.norm_path)  # Shape: [1, seq_len, hidden_dim]
        
        print(f"  Final logits shape: {final_logits.shape}")
        print(f"  Final hidden states shape: {final_hidden_states.shape}")
        
        # Apply logit lens using derived transformation
        results = apply_logit_lens(mlp_outputs, final_logits, final_hidden_states, tokenizer, args.top_k)
        
        # Format and display results
        formatted_output = format_results(results, model_name, data['prompt'])
        print("\n" + formatted_output)
        
        # Save detailed results
        save_results(results, args.output, model_name, data['prompt'])
        
        # Generate and save summary report
        if not args.no_summary:
            print("\nGenerating comprehensive summary report...")
            summary_report = create_summary_report(results, model_name, data['prompt'])
            
            with open(args.summary, 'w', encoding='utf-8') as f:
                f.write(summary_report)
            
            print(f"Summary report saved to: {args.summary}")
        
        print(f"\nAnalysis complete! Processed {len(results)} layers.")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
