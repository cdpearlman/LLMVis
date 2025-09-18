import argparse
import json
import sys
from typing import Dict, List, Tuple, Any
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def print_group(title: str, items: List[str], max_items: int = 200) -> None:
	print("=" * 80)
	print(title)
	print("=" * 80)
	if not items:
		print("(none)")
		return
	for idx, name in enumerate(items[:max_items]):
		print(f"[{idx}] {name}")
	if len(items) > max_items:
		print(f"... ({len(items) - max_items} more not shown)")


def categorize_modules(model) -> Tuple[List[str], List[str], List[str], Dict[str, List[str]]]:
	attn_patterns: Dict[str, List[str]] = {}
	mlp_patterns: Dict[str, List[str]] = {}
	other_patterns: Dict[str, List[str]] = {}
	
	for name, _ in model.named_modules():
		lower = name.lower()
		if not name:
			continue
		
		# Extract pattern (remove layer numbers)
		pattern = extract_pattern(name)
		
		if ("attn" in lower) or ("attention" in lower):
			if pattern not in attn_patterns:
				attn_patterns[pattern] = []
			attn_patterns[pattern].append(name)
		elif "mlp" in lower:
			if pattern not in mlp_patterns:
				mlp_patterns[pattern] = []
			mlp_patterns[pattern].append(name)
		else:
			if pattern not in other_patterns:
				other_patterns[pattern] = []
			other_patterns[pattern].append(name)
	
	# Return unique patterns and full mapping
	attn_unique = list(attn_patterns.keys())
	mlp_unique = list(mlp_patterns.keys())
	other_unique = list(other_patterns.keys())
	
	pattern_to_modules = {**attn_patterns, **mlp_patterns, **other_patterns}
	
	return attn_unique, mlp_unique, other_unique, pattern_to_modules


def extract_pattern(module_name: str) -> str:
	"""Extract pattern from module name by removing layer indices"""
	import re
	# Replace patterns like .h.0., .layers.0., .layer.0. with .{layer}.
	pattern = re.sub(r'\.(h|layers?)\.\d+\.', r'.{layer}.', module_name)
	return pattern


def parse_selection(user_input: str, options: List[str]) -> List[str]:
	selected: List[str] = []
	if not user_input.strip():
		return selected
	tokens = [t.strip() for t in user_input.split(',') if t.strip()]
	name_set = set(options)
	for tok in tokens:
		# allow index selection
		if tok.isdigit():
			idx = int(tok)
			if 0 <= idx < len(options):
				selected.append(options[idx])
			else:
				print(f"Warning: index {idx} out of range; skipping")
			continue
		# allow exact name match
		if tok in name_set:
			selected.append(tok)
			continue
		# allow suffix match for convenience (e.g., ".self_attn.o_proj")
		matches = [n for n in options if n.endswith(tok)]
		if len(matches) == 1:
			selected.append(matches[0])
		elif len(matches) > 1:
			print(f"Warning: ambiguous token '{tok}' matches {len(matches)} modules; skipping")
		else:
			print(f"Warning: token '{tok}' did not match any module; skipping")
	return list(dict.fromkeys(selected))  # dedupe, preserve order


def try_import_pyvene():
	try:
		import pyvene  # type: ignore
		return pyvene
	except Exception:
		return None


def register_hooks(
	model,
	module_names: List[str],
	store_inputs: bool,
	store_outputs: bool,
	use_pyvene: bool,
) -> Tuple[Dict[str, Any], List[Any]]:
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


def safe_to_serializable(obj: Any) -> Any:
	# Convert tensors to lists and tuples/lists/dicts recursively; cap very large tensors optionally
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


def main():
	parser = argparse.ArgumentParser(description="Agnostic activation capture via selectable module hooks")
	parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B", help="Model name or path")
	parser.add_argument("--prompt", type=str, default="Once upon a time", help="Input prompt")
	parser.add_argument("--device", type=str, default="cpu", help="Device (cpu|cuda|mps)")
	parser.add_argument("--use-pyvene", action="store_true", help="Attempt to use PyVene for hooking")
	parser.add_argument("--output", type=str, default="agnostic_activations.json", help="Output JSON path")
	parser.add_argument("--store-inputs", action="store_true", help="Store module inputs (pre and forward)")
	parser.add_argument("--store-outputs", action="store_true", default=True, help="Store module outputs")
	parser.add_argument("--max-print", type=int, default=200, help="Max items to print per group")
	parser.add_argument("--auto-select", type=str, help="Auto-select patterns (format: 'attn:1,2;mlp:1,2;other:')")
	args = parser.parse_args()

	print(f"Loading model: {args.model}")
	tokenizer = AutoTokenizer.from_pretrained(args.model)
	model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation='eager')
	model.eval()
	device = torch.device(args.device)
	model.to(device)

	attn_patterns, mlp_patterns, other_patterns, pattern_to_modules = categorize_modules(model)
	print_group("Attention-like module patterns (will apply to all layers)", attn_patterns, args.max_print)
	print_group("MLP-like module patterns (will apply to all layers)", mlp_patterns, args.max_print)
	print_group("Other module patterns", other_patterns, args.max_print)

	if args.auto_select:
		# Parse auto-select format: "attn:1,2;mlp:1,2;other:"
		parts = args.auto_select.split(';')
		attn_sel = mlp_sel = other_sel = ""
		for part in parts:
			if ':' in part:
				group, indices = part.split(':', 1)
				if group.strip().lower() in ['attn', 'attention']:
					attn_sel = indices
				elif group.strip().lower() == 'mlp':
					mlp_sel = indices
				elif group.strip().lower() == 'other':
					other_sel = indices
		print(f"Auto-selected: attn='{attn_sel}', mlp='{mlp_sel}', other='{other_sel}'")
	else:
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
	selected_patterns = sel_attn_patterns + sel_mlp_patterns + sel_other_patterns

	# Expand patterns to actual module names while keeping track of categories
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
	
	selected = selected_attn_modules + selected_mlp_modules + selected_other_modules

	print(f"\nSelected {len(selected_patterns)} patterns expanding to {len(selected)} actual modules:")
	for pattern in selected_patterns:
		count = len(pattern_to_modules.get(pattern, []))
		print(f" - {pattern} ({count} instances)")

	if not selected:
		print("No modules selected; exiting.")
		sys.exit(0)

	print(f"\nRegistering hooks for {len(selected)} modules...")
	captured, hooks = register_hooks(
		model,
		selected,
		store_inputs=args.store_inputs,
		store_outputs=args.store_outputs,
		use_pyvene=args.use_pyvene,
	)
	print(f"Successfully registered {len(hooks)} hooks")

	print(f"\nRunning forward pass with prompt: '{args.prompt}'")
	inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
	with torch.no_grad():
		_ = model(**inputs, use_cache=False)
	print(f"Forward pass completed. Captured data from {len(captured)} modules.")

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

	# Organize captured data by category
	attention_outputs = {k: v for k, v in captured.items() if k in selected_attn_modules}
	mlp_outputs = {k: v for k, v in captured.items() if k in selected_mlp_modules}
	other_outputs = {k: v for k, v in captured.items() if k in selected_other_modules}

	out = {
		"model": args.model,
		"prompt": args.prompt,
		"input_ids": inputs["input_ids"].detach().cpu().tolist(),
		"selected_patterns": {
			"attention": sel_attn_patterns,
			"mlp": sel_mlp_patterns,
			"other": sel_other_patterns
		},
		"selected_modules": {
			"attention": selected_attn_modules,
			"mlp": selected_mlp_modules,
			"other": selected_other_modules
		},
		"captured": {
			"attention_outputs": attention_outputs,
			"mlp_outputs": mlp_outputs,
			"other_outputs": other_outputs
		},
		"capture_summary": {
			"num_modules_selected": len(selected),
			"num_modules_captured": len(captured),
			"attention_captured": len(attention_outputs),
			"mlp_captured": len(mlp_outputs),
			"other_captured": len(other_outputs),
			"capture_keys": list(captured.keys())[:10] if captured else []
		}
	}
	with open(args.output, "w", encoding="utf-8") as f:
		json.dump(out, f, indent=2)
	print(f"Saved activations to {args.output}")


if __name__ == "__main__":
	main()
