# Lessons Learned

<!-- Append-only. Record what the team learned the hard way. -->
<!-- Format:
## YYYY-MM-DD — [Brief title]
**What happened**: What went wrong or what was discovered
**Root cause**: Why it happened
**Fix**: What was done about it
**Rule going forward**: What to do (or avoid) in the future
-->

## 2026-03-02 — Dead code accumulation during refactors
**What happened**: Large component changes left hundreds of lines of orphaned code from deprecated or deleted components
**Root cause**: Refactors focused only on building the new thing without cleaning up what the old thing left behind
**Fix**: Manual cleanup after discovering the bloat
**Rule going forward**: Every refactor must include a dead code sweep. This is a first-class concern, not an afterthought.

## 2026-03-02 — Tunnel vision on implementation details
**What happened**: Going deep on implementation rabbit holes produced outputs that weren't actually useful for the educational goal
**Root cause**: Losing sight of the "is this useful for teaching?" question while focused on technical correctness
**Fix**: Stepped back and re-evaluated against the educational mission
**Rule going forward**: Sanity check every significant change: (1) Does this help someone understand transformers? (2) Is this accurate enough for correct intuition? (3) Am I in a rabbit hole?

## 2026-03-19 — Missing torch_dtype causes silent gibberish on CPU
**What happened**: Beam search produced gibberish (exclamation points, partial words) for Pythia/OPT/GPT-2 Medium on HF Space but worked locally
**Root cause**: `from_pretrained()` without `torch_dtype=torch.float32` loads models in native dtype (float16/bfloat16). On CPU, these dtypes cause numerical instability and dtype mismatches in logit lens. GPT-2 Small happened to be natively float32, masking the bug.
**Fix**: Created centralized `load_model_for_inference()` with forced float32 + weight-tying check
**Rule going forward**: Always specify `torch_dtype=torch.float32` when loading models for CPU inference. Never scatter `from_pretrained` across multiple call sites — use a single loader.

## 2026-03-21 — Key mismatch between raw JSON and enriched helper return values
**What happened**: Category buttons all showed "(0)" heads despite data existing
**Root cause**: `get_active_head_summary()` returns categories with key `'heads'`, but the raw JSON file (`head_categories.json`) uses `'top_heads'`. Code in app.py used the raw key against the enriched object.
**Fix**: Changed `cat_data.get('top_heads', [])` to `cat_data.get('heads', [])` in app.py
**Rule going forward**: When consuming data from a helper function, check the helper's return schema — don't assume it mirrors the raw data file's keys.
