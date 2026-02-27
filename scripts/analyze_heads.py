#!/usr/bin/env python
"""
Offline Head Analysis Script

Uses TransformerLens to analyze attention head behaviors across test inputs
and generates a JSON file with head categories for each model.

Usage:
    python scripts/analyze_heads.py --model gpt2
    python scripts/analyze_heads.py --model gpt2 gpt2-medium EleutherAI/pythia-70m
    python scripts/analyze_heads.py --all

Output:
    Writes to utils/head_categories.json
"""

import os
os.environ["USE_TF"] = "0"  # Prevent TensorFlow noise

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

JSON_OUTPUT_PATH = PROJECT_ROOT / "utils" / "head_categories.json"

# ============================================================================
# TransformerLens model name mapping
# ============================================================================
# TL uses its own naming conventions. Map from HuggingFace names
# (used in our model_config.py) to TL names.

HF_TO_TL_NAME = {
    "gpt2": "gpt2-small",
    "openai-community/gpt2": "gpt2-small",
    "gpt2-medium": "gpt2-medium",
    "openai-community/gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
    "openai-community/gpt2-large": "gpt2-large",
    "gpt2-xl": "gpt2-xl",
    "openai-community/gpt2-xl": "gpt2-xl",
    "EleutherAI/gpt-neo-125M": "gpt-neo-125M",
    "EleutherAI/pythia-70m": "pythia-70m",
    "EleutherAI/pythia-160m": "pythia-160m",
    "EleutherAI/pythia-410m": "pythia-410m",
    "EleutherAI/pythia-1b": "pythia-1b",
    "EleutherAI/pythia-1.4b": "pythia-1.4b",
    "facebook/opt-125m": "opt-125m",
    "facebook/opt-350m": "opt-350m",
    "facebook/opt-1.3b": "opt-1.3b",
    "Qwen/Qwen2.5-0.5B": "qwen2.5-0.5b",
}

# Default models to analyze
DEFAULT_MODELS = ["gpt2"]

ALL_PRIORITY_MODELS = [
    "gpt2",
    "gpt2-medium",
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "facebook/opt-125m",
    "Qwen/Qwen2.5-0.5B",
]

# ============================================================================
# Category metadata (shared across all models)
# ============================================================================

CATEGORY_METADATA = {
    "previous_token": {
        "display_name": "Previous Token",
        "description": "Attends to the immediately preceding token — like reading left to right",
        "icon": "arrow-left",
        "educational_text": "This head looks at the word right before the current one. Like reading left to right, it helps track local word-by-word patterns.",
        "requires_repetition": False,
    },
    "induction": {
        "display_name": "Induction",
        "description": "Completes repeated patterns: if it saw [A][B] before and now sees [A], it predicts [B]",
        "icon": "repeat",
        "educational_text": "This head finds patterns that happened before and predicts they'll happen again. If it saw 'the cat' earlier, it expects the same words to follow.",
        "requires_repetition": True,
        "suggested_prompt": "Try: 'The cat sat on the mat. The cat' — the repeated 'The cat' lets induction heads activate.",
    },
    "duplicate_token": {
        "display_name": "Duplicate Token",
        "description": "Notices when the same word appears more than once",
        "icon": "clone",
        "educational_text": "This head notices when the same word appears more than once, like a highlighter for repeated words. It helps the model track which words have already been said.",
        "requires_repetition": True,
        "suggested_prompt": "Try a prompt with repeated words like 'The cat sat. The cat slept.' to see duplicate-token heads light up.",
    },
    "positional": {
        "display_name": "Positional / First-Token",
        "description": "Always pays attention to the very first word, using it as an anchor point",
        "icon": "map-pin",
        "educational_text": "This head always pays attention to the very first word, using it as an anchor point. The first token serves as a 'default' position when no other token is specifically relevant.",
        "requires_repetition": False,
    },
    "diffuse": {
        "display_name": "Diffuse / Spread",
        "description": "Spreads attention evenly across many words, gathering general context",
        "icon": "expand-arrows-alt",
        "educational_text": "This head spreads its attention evenly across many words, gathering general context rather than focusing on one spot. It provides a 'big picture' summary of the input.",
        "requires_repetition": False,
    },
}


# ============================================================================
# Test input generation
# ============================================================================

def generate_test_inputs(tokenizer) -> Dict[str, List[str]]:
    """Generate categorized test inputs for head analysis."""
    
    # Natural language prompts for general analysis
    natural_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "In the beginning, there was nothing but darkness and silence.",
        "Machine learning models process data to make predictions about the future.",
        "She walked through the park and noticed the flowers blooming everywhere.",
        "The president announced new economic policies at the press conference today.",
        "After years of research, scientists finally discovered the missing link.",
        "The library was quiet except for the occasional turning of pages.",
        "Programming is both an art and a science requiring careful thought.",
        "The restaurant on the corner served the best pizza in the entire city.",
        "Education is the most powerful tool for changing the world around us.",
        "The storm clouds gathered on the horizon as the wind began to howl.",
        "Mathematics provides the foundation for understanding complex physical phenomena.",
        "The children played happily in the garden while their parents watched.",
        "Economic growth depends on innovation, investment, and human capital development.",
        "The old man sat on the bench and watched the pigeons gather crumbs.",
        "Artificial intelligence will transform every industry in the coming decades.",
        "The river flowed gently through the valley between the tall mountains.",
        "Good communication skills are essential for success in any professional career.",
        "The concert hall was packed with enthusiastic fans waiting for the show.",
        "Climate change poses significant challenges for agriculture and food security.",
    ]
    
    # Repetitive prompts for induction / duplicate detection
    repetitive_prompts = [
        "The cat sat on the mat. The cat sat on the mat.",
        "One two three four five. One two three four five.",
        "Hello world hello world hello world hello world.",
        "Alice went to the store. Bob went to the store. Alice went to the store.",
        "The dog chased the ball. The dog chased the ball. The dog chased.",
        "Red blue green red blue green red blue green red.",
        "I like apples and I like oranges and I like apples.",
        "The sun rises in the east. The sun sets in the west. The sun rises.",
        "Monday Tuesday Wednesday Monday Tuesday Wednesday Monday.",
        "She said hello and he said hello and she said hello again.",
        "The key to success is practice. The key to success is patience.",
        "We went to the park and then we went to the park again.",
        "First second third first second third first second third.",
        "The teacher asked the student. The student asked the teacher. The teacher asked.",
        "North south east west north south east west north south.",
        "Open the door. Close the door. Open the door. Close the door.",
        "The big red ball bounced. The big red ball rolled.",
        "Cat dog cat dog cat dog cat dog cat dog.",
        "Learn practice improve learn practice improve learn practice.",
        "The man walked. The woman walked. The man walked. The woman walked.",
    ]
    
    return {
        "natural": natural_prompts,
        "repetitive": repetitive_prompts,
    }


# ============================================================================
# Head scoring functions
# ============================================================================

def score_previous_token(attn_patterns: torch.Tensor) -> torch.Tensor:
    """
    Score each head for previous-token behavior.
    
    For each position i > 0, check attention to position i-1.
    Returns [n_layers, n_heads] scores.
    """
    n_layers, n_heads, seq_len, _ = attn_patterns.shape
    
    if seq_len < 2:
        return torch.zeros(n_layers, n_heads)
    
    scores = torch.zeros(n_layers, n_heads)
    for i in range(1, seq_len):
        scores += attn_patterns[:, :, i, i - 1]
    scores /= (seq_len - 1)
    
    return scores


def score_positional(attn_patterns: torch.Tensor) -> torch.Tensor:
    """
    Score each head for first-token / positional behavior.
    
    Measures mean attention to position 0 across all positions.
    Returns [n_layers, n_heads] scores.
    """
    # Mean of column 0 across all query positions
    return attn_patterns[:, :, :, 0].mean(dim=-1)


def score_diffuse(attn_patterns: torch.Tensor) -> torch.Tensor:
    """
    Score each head for diffuse / bag-of-words behavior.
    
    Measures normalized entropy of attention distribution.
    Returns [n_layers, n_heads] scores.
    """
    n_layers, n_heads, seq_len, _ = attn_patterns.shape
    
    epsilon = 1e-10
    p = attn_patterns + epsilon
    entropy = -torch.sum(p * torch.log(p), dim=-1)  # [layers, heads, seq_len]
    max_entropy = np.log(seq_len)
    normalized = entropy / max_entropy if max_entropy > 0 else entropy
    
    return normalized.mean(dim=-1)  # Average over positions


def score_induction(attn_patterns: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """
    Score each head for induction behavior.
    
    For repeated tokens: if token[i] == token[j] (j < i), check attention from i to j+1.
    Returns [n_layers, n_heads] scores.
    """
    n_layers, n_heads, seq_len, _ = attn_patterns.shape
    scores = torch.zeros(n_layers, n_heads)
    count = 0
    
    for i in range(2, seq_len):
        for j in range(0, i - 1):
            if tokens[i].item() == tokens[j].item():
                target = j + 1
                if target < seq_len:
                    scores += attn_patterns[:, :, i, target]
                    count += 1
    
    if count > 0:
        scores /= count
    
    return scores


def score_duplicate_token(attn_patterns: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """
    Score each head for duplicate-token behavior.
    
    For repeated tokens: check attention from later to earlier occurrence.
    Returns [n_layers, n_heads] scores.
    """
    n_layers, n_heads, seq_len, _ = attn_patterns.shape
    scores = torch.zeros(n_layers, n_heads)
    count = 0
    
    for i in range(1, seq_len):
        for j in range(0, i):
            if tokens[i].item() == tokens[j].item():
                scores += attn_patterns[:, :, i, j]
                count += 1
    
    if count > 0:
        scores /= count
    
    return scores


# ============================================================================
# Main analysis
# ============================================================================

def analyze_model(model_name: str, device: str = "cpu") -> Dict[str, Any]:
    """
    Run full head analysis for a model.
    
    Returns a dict ready for JSON serialization.
    """
    from transformer_lens import HookedTransformer
    
    tl_name = HF_TO_TL_NAME.get(model_name, model_name)
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_name} (TL name: {tl_name})")
    print(f"{'='*60}")
    
    print("Loading model...")
    model = HookedTransformer.from_pretrained(tl_name, device=device)
    
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    print(f"  Layers: {n_layers}, Heads per layer: {n_heads}")
    
    # Generate test inputs
    test_inputs = generate_test_inputs(model.tokenizer)
    
    # Accumulators for scores
    prev_token_scores = torch.zeros(n_layers, n_heads)
    positional_scores = torch.zeros(n_layers, n_heads)
    diffuse_scores = torch.zeros(n_layers, n_heads)
    induction_scores = torch.zeros(n_layers, n_heads)
    duplicate_scores = torch.zeros(n_layers, n_heads)
    
    natural_count = 0
    repetitive_count = 0
    
    # Analyze natural prompts (for prev_token, positional, diffuse)
    print("\nAnalyzing natural prompts...")
    for prompt in test_inputs["natural"]:
        try:
            tokens = model.to_tokens(prompt)
            if tokens.shape[1] < 3:
                continue
            
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens)
            
            # Stack attention patterns: [n_layers, n_heads, seq_len, seq_len]
            attn_patterns = torch.stack([
                cache["pattern", layer][0]  # Remove batch dim
                for layer in range(n_layers)
            ])
            
            prev_token_scores += score_previous_token(attn_patterns)
            positional_scores += score_positional(attn_patterns)
            diffuse_scores += score_diffuse(attn_patterns)
            natural_count += 1
            
        except Exception as e:
            print(f"  Warning: Skipped prompt: {e}")
            continue
    
    print(f"  Processed {natural_count} natural prompts")
    
    # Analyze repetitive prompts (for induction + duplicate)
    print("Analyzing repetitive prompts...")
    for prompt in test_inputs["repetitive"]:
        try:
            tokens = model.to_tokens(prompt)
            if tokens.shape[1] < 4:
                continue
            
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens)
            
            attn_patterns = torch.stack([
                cache["pattern", layer][0]
                for layer in range(n_layers)
            ])
            
            induction_scores += score_induction(attn_patterns, tokens[0])
            duplicate_scores += score_duplicate_token(attn_patterns, tokens[0])
            
            # Also accumulate general scores for these prompts
            prev_token_scores += score_previous_token(attn_patterns)
            positional_scores += score_positional(attn_patterns)
            diffuse_scores += score_diffuse(attn_patterns)
            natural_count += 1
            
            repetitive_count += 1
            
        except Exception as e:
            print(f"  Warning: Skipped prompt: {e}")
            continue
    
    print(f"  Processed {repetitive_count} repetitive prompts")
    
    # Average scores
    if natural_count > 0:
        prev_token_scores /= natural_count
        positional_scores /= natural_count
        diffuse_scores /= natural_count
    if repetitive_count > 0:
        induction_scores /= repetitive_count
        duplicate_scores /= repetitive_count
    
    # Select top heads per category
    all_category_scores = {
        "previous_token": prev_token_scores,
        "induction": induction_scores,
        "duplicate_token": duplicate_scores,
        "positional": positional_scores,
        "diffuse": diffuse_scores,
    }
    
    # Print score summaries
    print("\nScore summaries (max per category):")
    for cat_name, scores in all_category_scores.items():
        max_score = scores.max().item()
        max_idx = scores.argmax()
        max_layer = max_idx // n_heads
        max_head = max_idx % n_heads
        print(f"  {cat_name:20s}: max={max_score:.4f} at L{max_layer}-H{max_head}")
    
    # Build category data
    categories_data = {}
    
    for cat_name, scores in all_category_scores.items():
        top_heads = select_top_heads(scores, n_layers, n_heads, cat_name)
        
        cat_entry = dict(CATEGORY_METADATA[cat_name])
        cat_entry["top_heads"] = top_heads
        categories_data[cat_name] = cat_entry
        
        print(f"\n  {cat_name} ({len(top_heads)} heads):")
        for h in top_heads:
            print(f"    L{h['layer']}-H{h['head']}: {h['score']:.4f}")
    
    # Build the full model entry
    model_entry = {
        "model_name": model_name,
        "num_layers": n_layers,
        "num_heads": n_heads,
        "analysis_date": time.strftime("%Y-%m-%d"),
        "categories": categories_data,
        "all_scores": {
            cat: scores.tolist()
            for cat, scores in all_category_scores.items()
        }
    }
    
    return model_entry


def select_top_heads(
    scores: torch.Tensor,
    n_layers: int,
    n_heads: int,
    category: str,
    max_heads: int = 8,
    primary_threshold: float = 0.25,
    min_threshold: float = 0.10,
) -> List[Dict[str, Any]]:
    """
    Select the top heads for a category, enforcing layer diversity.
    
    Strategy:
    1. Take all heads above primary_threshold
    2. Ensure we include the best head from each layer above min_threshold
    3. Cap at max_heads, keeping highest scores
    """
    candidates = []
    
    for layer in range(n_layers):
        for head in range(n_heads):
            score = scores[layer, head].item()
            if score > min_threshold:
                candidates.append({
                    "layer": layer,
                    "head": head,
                    "score": round(score, 4),
                })
    
    # Sort by score descending
    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    # Select: prioritize above primary_threshold, then fill with layer diversity
    selected = []
    selected_keys = set()
    layers_covered = set()
    
    # First pass: take all above primary threshold
    for c in candidates:
        if c["score"] >= primary_threshold and len(selected) < max_heads:
            key = (c["layer"], c["head"])
            if key not in selected_keys:
                selected.append(c)
                selected_keys.add(key)
                layers_covered.add(c["layer"])
    
    # Second pass: ensure layer diversity (best from each uncovered layer)
    for c in candidates:
        if len(selected) >= max_heads:
            break
        if c["layer"] not in layers_covered:
            key = (c["layer"], c["head"])
            if key not in selected_keys:
                selected.append(c)
                selected_keys.add(key)
                layers_covered.add(c["layer"])
    
    # Sort final result by layer, then head
    selected.sort(key=lambda x: (x["layer"], x["head"]))
    
    return selected[:max_heads]


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze attention head categories using TransformerLens")
    parser.add_argument("--model", nargs="+", default=None,
                        help="HuggingFace model name(s) to analyze (e.g., gpt2, EleutherAI/pythia-70m)")
    parser.add_argument("--all", action="store_true",
                        help="Analyze all priority models")
    parser.add_argument("--device", default="cpu",
                        help="Device to use (cpu or cuda)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: utils/head_categories.json)")
    args = parser.parse_args()
    
    # Determine models to analyze
    if args.all:
        models = ALL_PRIORITY_MODELS
    elif args.model:
        models = args.model
    else:
        models = DEFAULT_MODELS
    
    output_path = Path(args.output) if args.output else JSON_OUTPUT_PATH
    
    # Load existing data if present
    existing_data = {}
    if output_path.exists():
        try:
            with open(output_path, 'r') as f:
                existing_data = json.load(f)
            print(f"Loaded existing data from {output_path} ({len(existing_data)} models)")
        except (json.JSONDecodeError, IOError):
            pass
    
    # Analyze each model
    for model_name in models:
        try:
            result = analyze_model(model_name, device=args.device)
            
            # Store under the HuggingFace name
            existing_data[model_name] = result
            
            # Also store under the short name for lookup
            short_name = model_name.split('/')[-1] if '/' in model_name else None
            if short_name and short_name != model_name:
                existing_data[short_name] = result
            
        except Exception as e:
            print(f"\nERROR analyzing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(existing_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Done! Wrote {len(existing_data)} model entries to {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
