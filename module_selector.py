import re
import sys
import json
from typing import Dict, List, Tuple, Any
import argparse
import torch
from transformers import AutoModelForCausalLM


def detect_numeric_patterns(model) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Detect repeating numeric patterns in module names and group them.
    Returns (pattern_to_modules, module_to_pattern)
    """
    pattern_to_modules: Dict[str, List[str]] = {}
    module_to_pattern: Dict[str, List[str]] = {}
    
    for name, _ in model.named_modules():
        if not name:
            continue
            
        # Extract pattern by replacing numeric patterns with placeholders
        pattern = extract_generalized_pattern(name)
        
        if pattern not in pattern_to_modules:
            pattern_to_modules[pattern] = []
        pattern_to_modules[pattern].append(name)
        module_to_pattern[name] = pattern
    
    return pattern_to_modules, module_to_pattern


def extract_generalized_pattern(module_name: str) -> str:
    """
    Extract pattern from module name by replacing any numeric sequences with placeholders.
    Examples:
    - model.layers.0.self_attn -> model.layers.{N}.self_attn  
    - transformer.h.0.attn -> transformer.h.{N}.attn
    - model.decoder.layers.5.mlp -> model.decoder.layers.{N}.mlp
    """
    # Replace any sequence of digits that appears between dots or at word boundaries
    # This catches patterns like .0., .12., _0_, etc.
    pattern = re.sub(r'(\.|_)(\d+)(\.|_|$)', r'\1{N}\3', module_name)
    # Also catch cases where numbers appear directly after letters
    pattern = re.sub(r'([a-zA-Z])(\d+)(\.|_|$)', r'\1{N}\3', pattern)
    return pattern


def categorize_modules(pattern_to_modules: Dict[str, List[str]]) -> Tuple[List[str], List[str], List[str], Dict[str, List[str]]]:
    """
    Categorize module patterns into attention, MLP, and other based on naming patterns.
    Returns (attn_patterns, mlp_patterns, other_patterns, pattern_to_modules)
    """
    attn_patterns: List[str] = []
    mlp_patterns: List[str] = []
    other_patterns: List[str] = []
    
    for pattern in pattern_to_modules.keys():
        lower = pattern.lower()
        
        if ("attn" in lower) or ("attention" in lower):
            attn_patterns.append(pattern)
        elif "mlp" in lower:
            mlp_patterns.append(pattern)
        else:
            other_patterns.append(pattern)
    
    return attn_patterns, mlp_patterns, other_patterns, pattern_to_modules


def print_group(title: str, items: List[str]) -> None:
    """Print a formatted group of items with a title"""
    print("=" * 80)
    print(title)
    print("=" * 80)
    if not items:
        print("(none)")
        return
    for idx, name in enumerate(items):
        print(f"[{idx}] {name}")


def parse_selection(user_input: str, options: List[str]) -> List[str]:
    """Parse user selection input (indices, exact names, or suffix matches)"""
    selected: List[str] = []
    if not user_input.strip():
        return selected
        
    tokens = [t.strip() for t in user_input.split(',') if t.strip()]
    name_set = set(options)
    
    for tok in tokens:
        # Allow index selection
        if tok.isdigit():
            idx = int(tok)
            if 0 <= idx < len(options):
                selected.append(options[idx])
            else:
                print(f"Warning: index {idx} out of range; skipping")
            continue
            
        # Allow exact name match
        if tok in name_set:
            selected.append(tok)
            continue
            
        # Allow suffix match for convenience
        matches = [n for n in options if n.endswith(tok)]
        if len(matches) == 1:
            selected.append(matches[0])
        elif len(matches) > 1:
            print(f"Warning: ambiguous token '{tok}' matches {len(matches)} modules; skipping")
        else:
            print(f"Warning: token '{tok}' did not match any module; skipping")
            
    return list(dict.fromkeys(selected))  # dedupe, preserve order


def get_user_selection(
    attn_patterns: List[str], 
    mlp_patterns: List[str], 
    other_patterns: List[str]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Get user selection of patterns through interactive input.
    Returns (selected_attn_patterns, selected_mlp_patterns, selected_other_patterns)
    """
    print_group("Attention-like module patterns (will apply to all layers)", attn_patterns)
    print_group("MLP-like module patterns (will apply to all layers)", mlp_patterns)
    print_group("Other module patterns", other_patterns)

    print("\nEnter comma-separated selections. You may use indices (shown left), exact patterns, or unique suffixes.")
    print("Selected patterns will be applied to ALL layers that have them.")
    try:
        attn_sel = input("Select attention patterns: ")
        mlp_sel = input("Select MLP patterns: ")
        other_sel = input("Select other patterns: ")
    except (EOFError, KeyboardInterrupt):
        print("\nInput interrupted. Exiting.")
        sys.exit(0)

    sel_attn_patterns = parse_selection(attn_sel, attn_patterns)
    sel_mlp_patterns = parse_selection(mlp_sel, mlp_patterns)
    sel_other_patterns = parse_selection(other_sel, other_patterns)
    
    return sel_attn_patterns, sel_mlp_patterns, sel_other_patterns


def expand_patterns_to_modules(
    sel_attn_patterns: List[str],
    sel_mlp_patterns: List[str], 
    sel_other_patterns: List[str],
    pattern_to_modules: Dict[str, List[str]]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Expand selected patterns to actual module names while keeping track of categories.
    Returns (selected_attn_modules, selected_mlp_modules, selected_other_modules)
    """
    selected_attn_modules = []
    selected_mlp_modules = []
    selected_other_modules = []
    
    for pattern in sel_attn_patterns:
        if pattern in pattern_to_modules:
            selected_attn_modules.extend(pattern_to_modules[pattern])
    
    for pattern in sel_mlp_patterns:
        if pattern in pattern_to_modules:
            selected_mlp_modules.extend(pattern_to_modules[pattern])
    
    for pattern in sel_other_patterns:
        if pattern in pattern_to_modules:
            selected_other_modules.extend(pattern_to_modules[pattern])
    
    return selected_attn_modules, selected_mlp_modules, selected_other_modules


def save_module_selections(
    selected_attn_modules: List[str],
    selected_mlp_modules: List[str], 
    selected_other_modules: List[str],
    model_name: str,
    prompt: str,
    output_path: str = "module_selections.json"
) -> None:
    """
    Save module selections to a JSON file for use by activation_capture.py.
    """
    selections = {
        "model_name": model_name,
        "prompt": prompt,
        "selected_modules": {
            "attention": selected_attn_modules,
            "mlp": selected_mlp_modules,
            "other": selected_other_modules
        },
        "total_modules": len(selected_attn_modules) + len(selected_mlp_modules) + len(selected_other_modules)
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selections, f, indent=2)
    
    print(f"\nSaved module selections to: {output_path}")
    print(f"Total modules selected: {selections['total_modules']}")


def select_modules(model, model_name: str, prompt: str) -> None:
    """
    Main orchestrator function for module selection.
    Saves selections to module_selections.json.
    """
    # Detect patterns
    pattern_to_modules, _ = detect_numeric_patterns(model)
    
    # Categorize patterns
    attn_patterns, mlp_patterns, other_patterns, _ = categorize_modules(pattern_to_modules)
    
    # Get user selection
    sel_attn_patterns, sel_mlp_patterns, sel_other_patterns = get_user_selection(
        attn_patterns, mlp_patterns, other_patterns
    )
    
    # Expand to actual module names
    selected_attn_modules, selected_mlp_modules, selected_other_modules = expand_patterns_to_modules(
        sel_attn_patterns, sel_mlp_patterns, sel_other_patterns, pattern_to_modules
    )
    
    # Print summary
    selected_patterns = sel_attn_patterns + sel_mlp_patterns + sel_other_patterns
    total_modules = len(selected_attn_modules) + len(selected_mlp_modules) + len(selected_other_modules)
    
    print(f"\nSelected {len(selected_patterns)} patterns expanding to {total_modules} actual modules:")
    for pattern in selected_patterns:
        count = len(pattern_to_modules.get(pattern, []))
        print(f" - {pattern} ({count} instances)")
    
    # Save selections to file
    save_module_selections(
        selected_attn_modules, 
        selected_mlp_modules, 
        selected_other_modules,
        model_name,
        prompt
    )


def main():
    """Main function for module selection"""
    parser = argparse.ArgumentParser(description="Module pattern detection and selection")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B", help="Model name or path")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation='eager')
    model.eval()
    
    # Get prompt from user
    try:
        prompt = input("\nEnter prompt for activation capture: ")
    except (EOFError, KeyboardInterrupt):
        print("\nInput interrupted. Exiting.")
        sys.exit(0)
    
    select_modules(model, args.model, prompt)


if __name__ == "__main__":
    main() 