import argparse
import json
import sys
from typing import Dict, List, Tuple, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def try_import_pyvene():
    """Try to import PyVene, return None if not available"""
    try:
        import pyvene  # type: ignore
        return pyvene
    except Exception:
        return None


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
    module_names: List[str],
    store_inputs: bool,
    store_outputs: bool,
    use_pyvene: bool,
) -> Tuple[Dict[str, Any], List[Any]]:
    """
    Register hooks on specified modules to capture activations.
    Returns (captured_data, hooks_list)
    """
    captured: Dict[str, Any] = {}
    hooks: List[Any] = []

    name_to_module = dict(model.named_modules())

    if use_pyvene:
        pyvene = try_import_pyvene()
        if pyvene is None:
            print("PyVene not available; falling back to torch hooks.")
            use_pyvene = False

    if use_pyvene:
        # Minimal PyVene integration: attach to chosen modules; capture input/output
        from contextlib import ExitStack

        stack = ExitStack()
        # Store stack so we can close later via hooks list
        hooks.append(stack)

        def make_cb(mod_name: str):
            def callback(evt):
                # evt has: module, inputs, output (API may vary)
                try:
                    payload: Dict[str, Any] = {}
                    if store_inputs:
                        payload['inputs'] = safe_to_serializable(evt.inputs)
                    if store_outputs:
                        payload['output'] = safe_to_serializable(evt.output)
                    captured[mod_name] = payload
                except Exception as e:
                    captured[mod_name] = {"error": f"{e}"}
            return callback

        for mod_name in module_names:
            module = name_to_module.get(mod_name)
            if module is None:
                continue
            cb = make_cb(mod_name)
            # Using hypothetical pyvene API; wrap module forward
            handle = pyvene.hook(module, cb)
            stack.enter_context(handle)
        return captured, hooks

    # Torch hooks fallback
    def make_torch_hook(mod_name: str):
        def hook_fn(module, inputs, output):
            try:
                payload: Dict[str, Any] = {}
                if store_inputs:
                    payload['inputs'] = safe_to_serializable(inputs)
                if store_outputs:
                    payload['output'] = safe_to_serializable(output)
                captured[mod_name] = payload
                # Debug print for first few captures
                if len(captured) <= 3:
                    print(f"  Captured from {mod_name}: {list(payload.keys())}")
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
            # Also capture pre-forward inputs if requested
            if store_inputs:
                def make_pre_hook(name_inner: str):
                    def pre_hook_fn(module, inputs):
                        try:
                            entry = captured.get(name_inner, {})
                            entry['pre_inputs'] = safe_to_serializable(inputs)
                            captured[name_inner] = entry
                        except Exception as e:
                            captured[name_inner] = {"error_pre": f"{e}"}
                    return pre_hook_fn
                hooks.append(module.register_forward_pre_hook(make_pre_hook(mod_name)))
        except Exception as e:
            print(f"Failed to register hook for {mod_name}: {e}")
    
    print(f"Registered {registered_count} forward hooks successfully.")
    return captured, hooks


def remove_hooks(hooks: List[Any]) -> None:
    """Remove all registered hooks"""
    for h in hooks:
        try:
            # PyVene context manager vs torch handle
            close = getattr(h, 'close', None)
            if callable(close):
                close()
            else:
                h.remove()
        except Exception:
            pass


def capture_activations(
    model,
    selected_attn_modules: List[str],
    selected_mlp_modules: List[str], 
    selected_other_modules: List[str],
    prompt: str,
    tokenizer,
    device: torch.device,
    store_inputs: bool = False,
    store_outputs: bool = True,
    use_pyvene: bool = False
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
    captured, hooks = register_hooks(
        model,
        all_modules,
        store_inputs=store_inputs,
        store_outputs=store_outputs,
        use_pyvene=use_pyvene,
    )
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
        for name, data in list(captured.items())[:3]:  # Show first 3 for debugging
            if isinstance(data, dict) and 'output' in data:
                output_shape = "unknown"
                if isinstance(data['output'], list) and data['output']:
                    try:
                        # Try to infer shape from nested list structure
                        shape_info = f"list with {len(data['output'])} items"
                        if isinstance(data['output'][0], list):
                            shape_info += f" x {len(data['output'][0])}"
                        output_shape = shape_info
                    except:
                        pass
                print(f"  {name}: output shape = {output_shape}")

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
    """Main function for testing activation capture"""
    parser = argparse.ArgumentParser(description="Activation capture from specified modules")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B", help="Model name or path")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Input prompt")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu|cuda|mps)")
    parser.add_argument("--use-pyvene", action="store_true", help="Attempt to use PyVene for hooking")
    parser.add_argument("--output", type=str, default="simplified_activations.json", help="Output JSON path")
    parser.add_argument("--store-inputs", action="store_true", help="Store module inputs (pre and forward)")
    parser.add_argument("--store-outputs", action="store_true", default=True, help="Store module outputs")
    
    # For testing, we'll simulate some module selections
    parser.add_argument("--test-modules", type=str, nargs='+', 
                       help="Test with specific module names (format: attn:mod1,mod2 mlp:mod3,mod4 other:mod5)")
    
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation='eager')
    model.eval()
    device = torch.device(args.device)
    model.to(device)

    # For testing, use provided test modules or default selections
    if args.test_modules:
        selected_attn_modules = []
        selected_mlp_modules = []
        selected_other_modules = []
        
        for spec in args.test_modules:
            if ':' in spec:
                category, modules_str = spec.split(':', 1)
                modules = [m.strip() for m in modules_str.split(',') if m.strip()]
                if category.lower() in ['attn', 'attention']:
                    selected_attn_modules.extend(modules)
                elif category.lower() == 'mlp':
                    selected_mlp_modules.extend(modules)
                elif category.lower() == 'other':
                    selected_other_modules.extend(modules)
    else:
        # Default test: select first attention and mlp modules found
        all_modules = [name for name, _ in model.named_modules() if name]
        selected_attn_modules = [name for name in all_modules if 'attn' in name.lower()][:2]
        selected_mlp_modules = [name for name in all_modules if 'mlp' in name.lower()][:2]
        selected_other_modules = []
        
        print("No test modules specified, using default selection:")
        print(f"Attention: {selected_attn_modules}")
        print(f"MLP: {selected_mlp_modules}")

    captured_data, input_ids = capture_activations(
        model,
        selected_attn_modules,
        selected_mlp_modules,
        selected_other_modules,
        args.prompt,
        tokenizer,
        device,
        store_inputs=args.store_inputs,
        store_outputs=args.store_outputs,
        use_pyvene=args.use_pyvene
    )
    
    save_simplified_json(
        captured_data,
        selected_attn_modules,
        selected_mlp_modules,
        selected_other_modules,
        args.model,
        args.prompt,
        input_ids,
        args.output
    )


if __name__ == "__main__":
    main() 