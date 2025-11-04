# Fix Summary: Token Predictions Not Being Collected

## Problem
The "No predictions available" message was appearing because token predictions were not being collected in the `extract_layer_data` function.

## Root Cause
The code was checking if `logit_lens_parameter` was set before computing predictions:

```python
logit_lens_enabled = activation_data.get('logit_lens_parameter') is not None
# ...
top_tokens = _get_top_tokens(...) if logit_lens_enabled else None
```

However, `logit_lens_parameter` is **not actually required** for computing predictions. The `_get_top_tokens` function only needs:
1. `block_outputs` (layer outputs)
2. `norm_parameters` (final layer normalization)

## Investigation Process
1. Created test files to diagnose the issue
2. Ran tests showing predictions were not collected when `logit_lens_parameter` was `None`
3. Confirmed predictions WERE collected when `logit_lens_parameter` was set to any value
4. Identified that the condition was incorrect

## Solution
Changed the condition in `utils/model_patterns.py` (line 1054-1060) from checking `logit_lens_parameter` to checking for the actual required components:

**Before:**
```python
logit_lens_enabled = activation_data.get('logit_lens_parameter') is not None
# ...
top_tokens = _get_top_tokens(activation_data, module_name, model, tokenizer, top_k=5) if logit_lens_enabled else None
```

**After:**
```python
# Check if we can compute token predictions (requires block_outputs and norm_parameters)
# Note: Previously, this checked for logit_lens_parameter, but that parameter is not actually
# needed for computing predictions. The _get_top_tokens function only needs block_outputs
# and norm_parameters to work correctly.
has_block_outputs = bool(activation_data.get('block_outputs', {}))
has_norm_params = bool(activation_data.get('norm_parameters', []))
can_compute_predictions = has_block_outputs and has_norm_params
# ...
top_tokens = _get_top_tokens(activation_data, module_name, model, tokenizer, top_k=5) if can_compute_predictions else None
```

Also updated the condition for computing `global_top5_probs` on line 1076:
```python
if can_compute_predictions and global_top5_token_names:
```

## Testing
Created and ran test files that confirmed:
- **Before fix**: Predictions were NOT collected when `logit_lens_parameter` was `None`
- **After fix**: Predictions ARE collected correctly with just `block_outputs` and `norm_parameters`

Example test output (after fix):
```
First layer (Layer 0):
  - top_token: not
  - top_prob: 0.3322276473045349
  - top_5_tokens: [('not', 0.3322), ('now', 0.1279), ('still', 0.0824), ...]

[SUCCESS] FIX VERIFIED!
  Predictions ARE now collected WITHOUT logit_lens_parameter!
```

## Files Changed
- `utils/model_patterns.py`: Updated condition in `extract_layer_data` function
- `todo.md`: Added fix to recent fixes section

## Impact
This fix ensures that token predictions are always collected when the necessary components (`block_outputs` and `norm_parameters`) are available, regardless of whether `logit_lens_parameter` is set.

