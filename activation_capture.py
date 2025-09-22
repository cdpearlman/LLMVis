import argparse
import json
import sys
from typing import Dict, List, Tuple, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer




def safe_to_serializable(obj: Any) -> Any:
    """Convert tensors to lists and tuples/lists/dicts recursively"""
    try:
        if torch.is_tensor(obj):
            return obj.detach().cpu().tolist()
        if isinstance(obj, (list, tuple)):
            return [safe_to_serializable(x) for x in obj]
        if isinstance(obj, dict):
            return {k: safe_to_serializable(v) for k, v in obj.items()}
        # For other types, stringify compactly
        return obj
    except Exception:
        return str(obj)


def register_hooks(
    model,
    module_names: List[str]
) -> Tuple[Dict[str, Any], List[Any]]:
    """
    Register hooks on specified modules to capture activations.
    Returns (captured_data, hooks_list)
    """
    captured: Dict[str, Any] = {}
    hooks: List[Any] = []
    name_to_module = dict(model.named_modules())

    def make_torch_hook(mod_name: str):
        def hook_fn(module, inputs, output):
            try:
                captured[mod_name] = {'output': safe_to_serializable(output)}
                # Debug print for first few captures
                if len(captured) <= 3:
                    print(f"  Captured from {mod_name}")
            except Exception as e:
                captured[mod_name] = {"error": f"{e}"}
                print(f"  Error capturing from {mod_name}: {e}")
        return hook_fn

    registered_count = 0
    for mod_name in module_names:
        module = name_to_module.get(mod_name)
        if module is None:
            print(f"Warning: module '{mod_name}' not found; skipping")
            continue
        try:
            hooks.append(module.register_forward_hook(make_torch_hook(mod_name)))
            registered_count += 1
        except Exception as e:
            print(f"Failed to register hook for {mod_name}: {e}")
    
    print(f"Registered {registered_count} forward hooks successfully.")
    return captured, hooks


def remove_hooks(hooks: List[Any]) -> None:
    """Remove all registered hooks"""
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass


def load_module_selections(
    selections_path: str
) -> Tuple[List[str], List[str], List[str], str, str]:
    """
    Load module selections from JSON file.
    Returns (selected_attn_modules, selected_mlp_modules, selected_other_modules, model_name, prompt)
    """
    try:
        with open(selections_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        selected_modules = data.get("selected_modules", {})
        selected_attn_modules = selected_modules.get("attention", [])
        selected_mlp_modules = selected_modules.get("mlp", [])
        selected_other_modules = selected_modules.get("other", [])
        model_name = data.get("model_name", "")
        prompt = data.get("prompt", "")
        
        print(f"Loaded module selections from {selections_path}:")
        print(f"  Model: {model_name}")
        print(f"  Prompt: {prompt}")
        print(f"  Attention modules: {len(selected_attn_modules)}")
        print(f"  MLP modules: {len(selected_mlp_modules)}")
        print(f"  Other modules: {len(selected_other_modules)}")
        print(f"  Total modules: {len(selected_attn_modules) + len(selected_mlp_modules) + len(selected_other_modules)}")
        
        return selected_attn_modules, selected_mlp_modules, selected_other_modules, model_name, prompt
        
    except FileNotFoundError:
        print(f"ERROR: Module selections file not found: {selections_path}")
        print("Please run module_selector.py first to generate the selections file.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in selections file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load module selections: {e}")
        sys.exit(1)


def capture_activations(
    model,
    selected_attn_modules: List[str],
    selected_mlp_modules: List[str], 
    selected_other_modules: List[str],
    prompt: str,
    tokenizer,
    device: torch.device
) -> Tuple[Dict[str, Any], torch.Tensor]:
    """
    Main orchestrator function for capturing activations.
    Returns (captured_data, input_ids)
    """
    all_modules = selected_attn_modules + selected_mlp_modules + selected_other_modules
    
    if not all_modules:
        print("No modules selected; nothing to capture.")
        return {}, torch.tensor([])
    
    print(f"\nRegistering hooks for {len(all_modules)} modules...")
    captured, hooks = register_hooks(model, all_modules)
    print(f"Successfully registered {len(hooks)} hooks")

    print(f"\nRunning forward pass with prompt: '{prompt}'")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model(**inputs, use_cache=False)
    print(f"Forward pass completed. Captured data from {len(captured)} modules.")

    # Clean up hooks
    remove_hooks(hooks)

    # Validate captured data
    if not captured:
        print("WARNING: No data was captured! This might indicate an issue with hook registration.")
    else:
        print(f"Successfully captured data from {len(captured)} modules.")

    return captured, inputs["input_ids"]


def save_simplified_json(
    captured_data: Dict[str, Any],
    selected_attn_modules: List[str],
    selected_mlp_modules: List[str],
    selected_other_modules: List[str],
    model_name: str,
    prompt: str,
    input_ids: torch.Tensor,
    output_path: str
) -> None:
    """
    Save captured activations to a simplified JSON format.
    """
    # Organize captured data by category
    attention_data = {k: v for k, v in captured_data.items() if k in selected_attn_modules}
    mlp_data = {k: v for k, v in captured_data.items() if k in selected_mlp_modules}
    other_data = {k: v for k, v in captured_data.items() if k in selected_other_modules}

    simplified_output = {
        "model": model_name,
        "prompt": prompt,
        "input_ids": input_ids.detach().cpu().tolist(),
        "selected_modules": {
            "attention": selected_attn_modules,
            "mlp": selected_mlp_modules,
            "other": selected_other_modules
        },
        "attention_data": attention_data,
        "mlp_data": mlp_data,
        "other_data": other_data
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(simplified_output, f, indent=2)
    print(f"Saved simplified activations to {output_path}")


def main():
    """Main function for activation capture using module selections from JSON file"""
    parser = argparse.ArgumentParser(description="Activation capture from specified modules")
    parser.add_argument("--selections", type=str, default="module_selections.json", help="Path to module selections JSON file (generated by module_selector.py)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu|cuda|mps)")
    parser.add_argument("--output", type=str, default="activations.json", help="Output JSON path")
    args = parser.parse_args()

    # Load module selections from JSON file
    selected_attn_modules, selected_mlp_modules, selected_other_modules, model_name, prompt = load_module_selections(args.selections)
        
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation='eager')
    model.eval()
    device = torch.device(args.device)
    model.to(device)

    captured_data, input_ids = capture_activations(
        model,
        selected_attn_modules,
        selected_mlp_modules,
        selected_other_modules,
        prompt,
        tokenizer,
        device
    )
    
    save_simplified_json(
        captured_data,
        selected_attn_modules,
        selected_mlp_modules,
        selected_other_modules,
        model_name,
        prompt,
        input_ids,
        args.output
    )


if __name__ == "__main__":
    main() 