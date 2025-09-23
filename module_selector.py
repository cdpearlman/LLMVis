import re
import sys
import json
from typing import Dict, List, Tuple, Optional
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


def categorize_modules(pattern_to_modules: Dict[str, List[str]]) -> Tuple[List[str], List[str], List[str]]:
    """
    Categorize module patterns into attention, MLP, and other based on naming patterns.
    Returns (attn_patterns, mlp_patterns, other_patterns)
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
    
    return attn_patterns, mlp_patterns, other_patterns


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


def get_module_selection(attn_patterns: List[str], mlp_patterns: List[str], other_patterns: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Get user selection of attention, MLP, and other patterns.
    Returns (selected_attn_patterns, selected_mlp_patterns, selected_other_patterns)
    """
    print_group("Attention module patterns (will apply to all layers)", attn_patterns)
    print_group("MLP module patterns (will apply to all layers)", mlp_patterns)
    print_group("Other module patterns (for model-agnostic selection)", other_patterns)

    print("\nEnter comma-separated selections. You may use indices (shown left), exact patterns, or unique suffixes.")
    print("You can select from 'other' patterns and assign them as attention or MLP modules.")
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
    Expand selected patterns to actual module names.
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


def detect_parameter_patterns(model) -> Dict[str, List[str]]:
    """
    Detect parameter patterns and categorize them.
    Returns pattern_to_parameters dictionary
    """
    pattern_to_parameters: Dict[str, List[str]] = {}
    
    for name, _ in model.named_parameters():
        # Extract pattern by replacing numeric patterns with placeholders
        pattern = extract_generalized_pattern(name)
        
        if pattern not in pattern_to_parameters:
            pattern_to_parameters[pattern] = []
        pattern_to_parameters[pattern].append(name)
    
    return pattern_to_parameters


def categorize_parameter_patterns(pattern_to_parameters: Dict[str, List[str]]) -> Tuple[List[str], List[str], List[str]]:
    """
    Categorize parameter patterns into logit lens, normalization, and other.
    Returns (logit_lens_patterns, norm_patterns, other_patterns)
    """
    logit_lens_patterns: List[str] = []
    norm_patterns: List[str] = []
    other_patterns: List[str] = []
    
    for pattern in pattern_to_parameters.keys():
        lower = pattern.lower()
        
        if any(x in lower for x in ['lm_head', 'head', 'classifier', 'embed', 'wte', 'word']):
            logit_lens_patterns.append(pattern)
        elif any(x in lower for x in ['norm', 'layernorm', 'layer_norm']):
            norm_patterns.append(pattern)
        else:
            other_patterns.append(pattern)
    
    return logit_lens_patterns, norm_patterns, other_patterns


def get_parameter_selection(
    logit_lens_patterns: List[str],
    norm_patterns: List[str],
    other_patterns: List[str],
    pattern_to_parameters: Dict[str, List[str]]
) -> Tuple[Optional[str], List[str], List[str]]:
    """
    Get user selection of logit lens, normalization, and other parameters.
    Returns (logit_lens_parameter_name, norm_parameter_names, other_parameter_names)
    """
    print("\n" + "="*80)
    print("PARAMETER SELECTION")
    print("="*80)
    
    # Show logit lens parameters
    print("\nLOGIT LENS PARAMETERS")
    print("💡 For logit lens analysis - maps activations to vocabulary space")
    print("-" * 60)
    if logit_lens_patterns:
        for idx, pattern in enumerate(logit_lens_patterns):
            count = len(pattern_to_parameters.get(pattern, []))
            print(f"[{idx}] {pattern} ({count} instances)")
    else:
        print("(none found)")
    
    # Show normalization parameters
    print("\nNORMALIZATION PARAMETERS")
    print("💡 Layer normalization weights - needed for proper scaling")
    print("-" * 60)
    if norm_patterns:
        for idx, pattern in enumerate(norm_patterns):
            count = len(pattern_to_parameters.get(pattern, []))
            print(f"[{idx}] {pattern} ({count} instances)")
    else:
        print("(none found)")
    
    # Show other parameters
    print("\nOTHER PARAMETERS")
    print("💡 Other model parameters (for model-agnostic selection)")
    print("-" * 60)
    if other_patterns:
        for idx, pattern in enumerate(other_patterns):
            count = len(pattern_to_parameters.get(pattern, []))
            print(f"[{idx}] {pattern} ({count} instances)")
    else:
        print("(none found)")

    print("\nSelect parameter patterns:")
    
    try:
        # Get single logit lens parameter
        logit_sel = ""
        if logit_lens_patterns:
            logit_sel = input("Select ONE logit lens pattern (index or name): ")
        
        # Get normalization parameters
        norm_sel = ""
        if norm_patterns:
            norm_sel = input("Select normalization patterns: ")
        
        # Get other parameters
        other_sel = ""
        if other_patterns:
            other_sel = input("Select other patterns: ")
            
    except (EOFError, KeyboardInterrupt):
        print("\nInput interrupted. Exiting.")
        sys.exit(0)

    # Parse logit lens selection (should be single)
    logit_lens_parameter = None
    if logit_sel.strip() and logit_lens_patterns:
        sel_logit_patterns = parse_selection(logit_sel, logit_lens_patterns)
        if sel_logit_patterns:
            # Get first parameter from first pattern
            first_pattern = sel_logit_patterns[0]
            if first_pattern in pattern_to_parameters:
                logit_lens_parameter = pattern_to_parameters[first_pattern][0]
                print(f"Selected logit lens parameter: {logit_lens_parameter}")
    
    # Parse normalization selections
    sel_norm_patterns = parse_selection(norm_sel, norm_patterns)
    norm_parameter_names = []
    for pattern in sel_norm_patterns:
        if pattern in pattern_to_parameters:
            norm_parameter_names.extend(pattern_to_parameters[pattern])
    
    # Parse other parameter selections
    sel_other_patterns = parse_selection(other_sel, other_patterns)
    other_parameter_names = []
    for pattern in sel_other_patterns:
        if pattern in pattern_to_parameters:
            other_parameter_names.extend(pattern_to_parameters[pattern])
    
    return logit_lens_parameter, norm_parameter_names, other_parameter_names


def save_selections(
    selected_attn_modules: List[str],
    selected_mlp_modules: List[str], 
    selected_other_modules: List[str],
    logit_lens_parameter: Optional[str],
    norm_parameter_names: List[str],
    model_name: str,
    prompt: str,
    output_path: str = "module_selections.json"
) -> None:
    """
    Save module and parameter selections to JSON file in the specified format.
    Note: Other modules are collected but not saved in the output format.
    """
    selections = {
        "model": model_name,
        "prompt": prompt,
        "attention_modules": selected_attn_modules,
        "mlp_modules": selected_mlp_modules,
        "norm_parameters": norm_parameter_names,
        "logit_lens_parameter": logit_lens_parameter
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selections, f, indent=2)
    
    print(f"\nSaved selections to: {output_path}")
    print(f"Attention modules: {len(selected_attn_modules)}")
    print(f"MLP modules: {len(selected_mlp_modules)}")
    print(f"Other modules selected: {len(selected_other_modules)} (available for manual assignment)")
    print(f"Norm parameters: {len(norm_parameter_names)}")
    print(f"Logit lens parameter: {logit_lens_parameter}")


def select_modules(model, model_name: str, prompt: str) -> None:
    """
    Main orchestrator function for module and parameter selection.
    Saves selections to module_selections.json in the specified format.
    """
    print("=== FORWARD PASS TO DETECT MODULES AND PARAMETERS ===")
    
    # 1. Detect and categorize module patterns
    pattern_to_modules, _ = detect_numeric_patterns(model)
    attn_patterns, mlp_patterns, other_patterns = categorize_modules(pattern_to_modules)
    
    # 2. Get user selection for modules
    sel_attn_patterns, sel_mlp_patterns, sel_other_patterns = get_module_selection(
        attn_patterns, mlp_patterns, other_patterns
    )
    
    # 3. Expand patterns to actual module names
    selected_attn_modules, selected_mlp_modules, selected_other_modules = expand_patterns_to_modules(
        sel_attn_patterns, sel_mlp_patterns, sel_other_patterns, pattern_to_modules
    )
    
    print(f"\nSelected {len(selected_attn_modules)} attention modules, {len(selected_mlp_modules)} MLP modules, and {len(selected_other_modules)} other modules")
    
    # 3.5. Optionally reassign other modules for model-agnostic support
    if selected_other_modules:
        print("\n" + "="*80)
        print("REASSIGN OTHER MODULES (for model-agnostic support)")
        print("="*80)
        print("You can reassign 'other' modules to be treated as attention or MLP modules:")
        for idx, module in enumerate(selected_other_modules):
            print(f"[{idx}] {module}")
        
        try:
            reassign_input = input("Enter indices to reassign as attention modules (comma-separated): ")
            if reassign_input.strip():
                reassign_indices = [int(i.strip()) for i in reassign_input.split(',') if i.strip().isdigit()]
                for idx in reassign_indices:
                    if 0 <= idx < len(selected_other_modules):
                        module = selected_other_modules[idx]
                        selected_attn_modules.append(module)
                        print(f"  Reassigned {module} as attention module")
            
            reassign_input = input("Enter indices to reassign as MLP modules (comma-separated): ")
            if reassign_input.strip():
                reassign_indices = [int(i.strip()) for i in reassign_input.split(',') if i.strip().isdigit()]
                for idx in reassign_indices:
                    if 0 <= idx < len(selected_other_modules):
                        module = selected_other_modules[idx]
                        selected_mlp_modules.append(module)
                        print(f"  Reassigned {module} as MLP module")
        except (EOFError, KeyboardInterrupt):
            print("\nSkipping reassignment...")
        except ValueError:
            print("Invalid input. Skipping reassignment...")
    
    # 4. Detect and categorize parameter patterns
    pattern_to_parameters = detect_parameter_patterns(model)
    logit_lens_patterns, norm_patterns, other_param_patterns = categorize_parameter_patterns(pattern_to_parameters)
    
    # 5. Get user selection for parameters (logit lens, norm, and other)
    logit_lens_parameter, norm_parameter_names, other_parameter_names = get_parameter_selection(
        logit_lens_patterns, norm_patterns, other_param_patterns, pattern_to_parameters
    )
    
    # 5.5. Optionally reassign other parameters for model-agnostic support
    if other_parameter_names:
        print("\n" + "="*80)
        print("REASSIGN OTHER PARAMETERS (for model-agnostic support)")
        print("="*80)
        print("You can reassign 'other' parameters:")
        for idx, param in enumerate(other_parameter_names):
            print(f"[{idx}] {param}")
        
        try:
            reassign_input = input("Enter index to use as logit lens parameter (single index): ")
            if reassign_input.strip() and reassign_input.strip().isdigit():
                idx = int(reassign_input.strip())
                if 0 <= idx < len(other_parameter_names):
                    if logit_lens_parameter is None:  # Only assign if none selected
                        logit_lens_parameter = other_parameter_names[idx]
                        print(f"  Reassigned {logit_lens_parameter} as logit lens parameter")
                    else:
                        print(f"  Logit lens parameter already selected: {logit_lens_parameter}")
            
            reassign_input = input("Enter indices to reassign as normalization parameters (comma-separated): ")
            if reassign_input.strip():
                reassign_indices = [int(i.strip()) for i in reassign_input.split(',') if i.strip().isdigit()]
                for idx in reassign_indices:
                    if 0 <= idx < len(other_parameter_names):
                        param = other_parameter_names[idx]
                        norm_parameter_names.append(param)
                        print(f"  Reassigned {param} as normalization parameter")
        except (EOFError, KeyboardInterrupt):
            print("\nSkipping parameter reassignment...")
        except ValueError:
            print("Invalid input. Skipping parameter reassignment...")
    
    # 6. Save selections in the specified format
    save_selections(
        selected_attn_modules, 
        selected_mlp_modules, 
        selected_other_modules,
        logit_lens_parameter,
        norm_parameter_names,
        model_name,
        prompt
    )


def main():
    """Main function for module and parameter selection"""
    parser = argparse.ArgumentParser(description="Module and parameter selection for activation capture")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B", help="Model name or path")
    parser.add_argument("--prompt", type=str, help="Prompt for activation capture")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation='eager')
    model.eval()
    
    # Get prompt from CLI or user input
    if args.prompt:
        prompt = args.prompt
        print(f"Using prompt: {prompt}")
    else:
        try:
            prompt = input("\nEnter prompt for activation capture: ")
        except (EOFError, KeyboardInterrupt):
            print("\nInput interrupted. Exiting.")
            sys.exit(0)
    
    select_modules(model, args.model, prompt)


if __name__ == "__main__":
    main() 