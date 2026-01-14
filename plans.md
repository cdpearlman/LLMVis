## Current Plan

### Pipeline Explanation Refactor - COMPLETED

The dashboard has been refactored from a testing/analysis tool into an explanation-first interface:

1. **New Pipeline Visualization**: Linear flow (Input → Tokens → Embed → Attention → MLP → Output) with click-to-expand stages
2. **Investigation Panel**: Consolidated ablation and token attribution tools
3. **Simplified Codebase**: Removed heatmap, comparison mode, and ~900 lines of code
4. **Token Attribution**: New gradient-based feature importance analysis

#### File Changes
- `app.py`: Reduced from 1781 to ~750 lines
- `components/pipeline.py`: NEW - Main explanation flow
- `components/investigation_panel.py`: NEW - Ablation + Attribution
- `utils/token_attribution.py`: NEW - Integrated Gradients
- `model_selector.py`: Simplified (removed comparison UI)
- `main_panel.py`: DELETED
- `prompt_comparison.py`: DELETED
