import argparse
import json
import sys
from typing import Dict, List, Tuple, Any, Optional
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


def capture_norm_parameters(model, norm_parameter_names: List[str]) -> List[Any]:
    """Capture normalization parameters from model"""
    norm_data = []
    
    if not norm_parameter_names:
        return norm_data
    
    print(f"Capturing {len(norm_parameter_names)} normalization parameters...")
    all_params = dict(model.named_parameters())
    
    for param_name in norm_parameter_names:
        if param_name in all_params:
            param_tensor = all_params[param_name]
            norm_data.append(safe_to_serializable(param_tensor))
            print(f"  {param_name}: shape {param_tensor.shape}")
        else:
            print(f"  Warning: Parameter '{param_name}' not found")
    
    return norm_data


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


def load_selections(selections_path: str) -> Tuple[List[str], List[str], List[str], Optional[str], str, str]:
    """
    Load module and parameter selections from JSON file.
    Returns (attention_modules, mlp_modules, norm_parameters, logit_lens_parameter, model_name, prompt)
    """
    try:
        with open(selections_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        attention_modules = data.get("attention_modules", [])
        mlp_modules = data.get("mlp_modules", [])
        norm_parameters = data.get("norm_parameters", [])
        logit_lens_parameter = data.get("logit_lens_parameter")
        model_name = data.get("model", "")
        prompt = data.get("prompt", "")
        
        print(f"Loaded selections from {selections_path}:")
        print(f"  Model: {model_name}")
        print(f"  Prompt: {prompt}")
        print(f"  Attention modules: {len(attention_modules)}")
        print(f"  MLP modules: {len(mlp_modules)}")
        print(f"  Norm parameters: {len(norm_parameters)}")
        print(f"  Logit lens parameter: {logit_lens_parameter}")
        
        return attention_modules, mlp_modules, norm_parameters, logit_lens_parameter, model_name, prompt
        
    except FileNotFoundError:
        print(f"ERROR: Selections file not found: {selections_path}")
        print("Please run module_selector.py first to generate the selections file.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in selections file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load selections: {e}")
        sys.exit(1)


def capture_activations_and_data(
    model,
    attention_modules: List[str],
    mlp_modules: List[str], 
    norm_parameters: List[str],
    prompt: str,
    tokenizer,
    device: torch.device
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Any], torch.Tensor]:
    """
    Capture activations from selected modules and normalization parameters.
    Returns (attention_outputs, mlp_outputs, norm_data, input_ids)
    """
    all_modules = attention_modules + mlp_modules
    
    if not all_modules:
        print("No modules selected; nothing to capture.")
        return {}, {}, [], torch.tensor([])
    
    # Capture activations
    print(f"\nRegistering hooks for {len(all_modules)} modules...")
    captured_activations, hooks = register_hooks(model, all_modules)

    print(f"Running forward pass with prompt: '{prompt}'")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model(**inputs, use_cache=False)
    
    remove_hooks(hooks)
    print(f"Captured activations from {len(captured_activations)} modules")

    # Separate attention and MLP outputs
    attention_outputs = {k: v for k, v in captured_activations.items() if k in attention_modules}
    mlp_outputs = {k: v for k, v in captured_activations.items() if k in mlp_modules}

    # Capture normalization parameters
    norm_data = capture_norm_parameters(model, norm_parameters)

    return attention_outputs, mlp_outputs, norm_data, inputs["input_ids"]


def save_activations(
    attention_outputs: Dict[str, Any],
    mlp_outputs: Dict[str, Any],
    attention_modules: List[str],
    mlp_modules: List[str],
    norm_data: List[Any],
    logit_lens_parameter: Optional[str],
    model_name: str,
    prompt: str,
    output_path: str
) -> None:
    """Save captured data to JSON format matching the specified structure."""
    output = {
        "model": model_name,
        "prompt": prompt,
        "attention_modules": attention_modules,
        "attention_outputs": attention_outputs,
        "mlp_modules": mlp_modules,
        "mlp_outputs": mlp_outputs,
        "norm_parameter": norm_data,
        "logit_lens_parameter": logit_lens_parameter
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Saved data to {output_path}")
    print(f"  Attention outputs: {len(attention_outputs)}")
    print(f"  MLP outputs: {len(mlp_outputs)}")
    print(f"  Norm parameters: {len(norm_data)}")
    print(f"  Logit lens parameter: {logit_lens_parameter}")


def main():
    """Main function for activation and parameter capture using selections from JSON file"""
    parser = argparse.ArgumentParser(description="Capture activations and parameters from selected modules")
    parser.add_argument("--selections", type=str, default="module_selections.json", 
                       help="Path to selections JSON file (generated by module_selector.py)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu|cuda|mps)")
    parser.add_argument("--output", type=str, default="activations.json", help="Output JSON path")
    args = parser.parse_args()

    # Load selections from JSON file
    attention_modules, mlp_modules, norm_parameters, logit_lens_parameter, model_name, prompt = load_selections(args.selections)
        
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation='eager')
    model.eval()
    device = torch.device(args.device)
    model.to(device)

    attention_outputs, mlp_outputs, norm_data, input_ids = capture_activations_and_data(
        model,
        attention_modules,
        mlp_modules,
        norm_parameters,
        prompt,
        tokenizer,
        device
    )
    
    save_activations(
        attention_outputs,
        mlp_outputs,
        attention_modules,
        mlp_modules,
        norm_data,
        logit_lens_parameter,
        model_name,
        prompt,
        args.output
    )


if __name__ == "__main__":
    main() 