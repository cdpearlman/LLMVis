# Todo

## Completed: Test Suite Setup (Done)
- [x] Create `tests/` folder with `__init__.py` and `conftest.py` (shared fixtures)
- [x] Create `test_model_config.py` - 15 tests for model family lookups
- [x] Create `test_ablation_metrics.py` - 8 tests for KL divergence and probability deltas
- [x] Create `test_head_detection.py` - 20 tests for attention head categorization
- [x] Create `test_model_patterns.py` - 16 tests for merge_token_probabilities, safe_to_serializable
- [x] Create `test_token_attribution.py` - 11 tests for visualization data formatting
- [x] Verify all 73 tests pass with `pytest tests/ -v`

## Completed: Pipeline Explanation Refactor

### Phase 1: New Components (Done)
- [x] Create `components/pipeline.py` with 5 expandable stages
- [x] Create `utils/token_attribution.py` with Integrated Gradients
- [x] Create `components/investigation_panel.py` (ablation + attribution)

### Phase 2: Simplifications (Done)
- [x] Remove comparison UI from `model_selector.py`
- [x] Refactor `app.py`: wire pipeline, remove heatmap/comparison callbacks

### Phase 3: Cleanup (Done)
- [x] Delete `main_panel.py`
- [x] Delete `prompt_comparison.py`
- [x] Update `utils/__init__.py` exports
- [x] Add pipeline CSS styles to `assets/style.css`
