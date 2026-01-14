# Todo

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

---

## Next Steps

### Testing
- [ ] Run the dashboard and verify all pipeline stages render correctly
- [ ] Test ablation experiment workflow
- [ ] Test token attribution (both methods)
- [ ] Verify beam search still works with multi-token generation

### Enhancements (Optional)
- [ ] Add loading spinners to investigation tools
- [ ] Improve attention visualization formatting
- [ ] Add more detailed MLP stage visualization
- [ ] Consider adding "copy to clipboard" for token data
