# Track Specification: Interactive Attention Head Ablation

## Overview
This track introduces interactive ablation capabilities to the Dash dashboard. Users will be able to selectively disable (zero out) specific attention heads in the transformer model and observe the resulting changes in model output (logits/probabilities) and attention patterns. This directly supports the "Interactive Experimentation" core value proposition.

## Goals
- Enable users to toggle specific attention heads on/off via the UI.
- Update the model's forward pass to respect these ablation masks.
- Visualize the "ablated" state compared to the "original" state (if feasible) or simply show the new state.
- Provide immediate feedback on how head removal affects token prediction.

## User Stories
- As a student, I want to turn off a specific attention head to see if it is responsible for a particular grammatical dependency (e.g., matching plural subjects to verbs).
- As a researcher, I want to ablate a group of heads to test a hypothesis about distributed representations.
- As a user, I want clear visual indicators of which heads are currently active or disabled.

## Requirements

### Frontend (Dash)
- **Ablation Control Panel:** A UI component (e.g., a grid of toggles or a heatmap with clickable cells) representing all attention heads in the model (Layers x Heads).
- **State Management:** Store the set of "disabled heads" in the Dash app state (`dcc.Store`).
- **Visual Feedback:** 
    - Disabled heads should be visually distinct (e.g., grayed out) in the visualization.
    - The output (Logit Lens or Top-K tokens) must update dynamically when heads are toggled.

### Backend (Model Logic)
- **Intervention Mechanism:** Modify the `model_patterns.py` or `agnostic_capture.py` logic to accept an "ablation mask".
- **PyVene Integration:** Use PyVene's intervention capabilities to zero out the activations of specific heads during the forward pass.
    - *Technical Note:* This might require defining a specific intervention function that takes the head output and multiplies it by 0 if the index matches the ablated head.

### Visualization
- Update the attention map visualization to reflect that the ablated head is contributing nothing (blank map or "Disabled" overlay).

## Non-Functional Requirements
- **Latency:** The update loop (Toggle -> Inference -> Update UI) should be fast enough for interactive exploration (< 2-3 seconds for small/medium models).
- **Clarity:** It must be obvious to the user that they have modified the model. A "Reset All" button is essential.
