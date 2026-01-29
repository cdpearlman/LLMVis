# Implementation Plan - Interactive Attention Head Ablation

## Phase 1: Backend Support for Ablation
- [ ] Task: Create a reproduction script to test manual PyVene interventions for head ablation.
    - [ ] Sub-task: Write a standalone script that loads a small model (e.g., GPT-2) and uses PyVene to zero out a specific head (e.g., L0H0).
    - [ ] Sub-task: Verify that the output logits change compared to the baseline run.
- [ ] Task: Extend `utils/model_patterns.py` (or creating `utils/ablation.py`) to support dynamic head masking.
    - [ ] Sub-task: Write tests for the new ablation utility function.
    - [ ] Sub-task: Implement a function `apply_ablation_mask(model, heads_to_ablate)` that registers the necessary PyVene hooks.
- [ ] Task: Update the main inference pipeline to accept an ablation configuration.
    - [ ] Sub-task: Modify the capture logic to check for an "ablation list" in the request.
    - [ ] Sub-task: Ensure the pipeline correctly applies the mask before running the forward pass.
- [ ] Task: Conductor - User Manual Verification 'Backend Support for Ablation' (Protocol in workflow.md)

## Phase 2: Frontend Control Panel
- [ ] Task: Create an `AblationPanel` component in `components/`.
    - [ ] Sub-task: Design a layout (e.g., Heatmap or Grid) that displays all heads (Layers rows x Heads columns).
    - [ ] Sub-task: Implement the callback to handle clicks on the grid and update a `dcc.Store` with the list of disabled heads.
- [ ] Task: Integrate the `AblationPanel` into `app.py`.
    - [ ] Sub-task: Add the panel to the main layout (likely in a new "Experiments" tab or collapsible sidebar).
    - [ ] Sub-task: Connect the global "Run" or "Update" callback to include the ablation state from the store.
- [ ] Task: Conductor - User Manual Verification 'Frontend Control Panel' (Protocol in workflow.md)

## Phase 3: Visualization & Feedback Loop
- [ ] Task: Connect the Frontend Ablation State to the Backend Inference.
    - [ ] Sub-task: Update the main `app.py` callback to pass the `disabled_heads` list to the backend capture function.
    - [ ] Sub-task: Verify that toggling a head in the UI updates the Logit Lens/Output display.
- [ ] Task: Visual Polish for Ablated State.
    - [ ] Sub-task: Ensure the Attention Map visualization shows disabled heads as blank or "inactive".
    - [ ] Sub-task: Add a "Reset Ablations" button to quickly restore the original model state.
- [ ] Task: Conductor - User Manual Verification 'Visualization & Feedback Loop' (Protocol in workflow.md)
