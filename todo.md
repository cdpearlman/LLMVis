# Todo

## Simple Fixes

### Glossary Layout
- [x] Add padding/margin to `.glossary-content` in `components/glossary.py`
- [x] Test that text no longer reaches screen edges

### Collapsible Tokenization Example  
- [x] Wrap static diagram in `html.Details()` in `components/tokenization_panel.py`
- [x] Add `html.Summary("View example tokenization flow")`
- [x] Default to collapsed state

### Tokenization Visual Toggle
- [x] Show example diagram always (no collapse)
- [x] Collapse prompt tokenization to first token + ellipsis

### Fix Double Arrow (Transformer Layers)
- [x] Add `.transformer-layers-summary::-webkit-details-marker { display: none; }` to `assets/style.css`
- [x] Add `.transformer-layers-summary::marker { display: none; }` for Firefox

### Reorder Tokenization Section
- [x] Move `create_tokenization_panel()` above heatmap container in `components/main_panel.py`

---

## Complex: Position x Layer Heatmap

### Design Spec
- X-axis: Token positions | Y-axis: Layers (0 at bottom)
- Color: Light→Dark blue | Metric: Top-token probability delta (layer-to-layer)
- Click cell → Modal with layer details
- Toggle support for comparison mode (Prompt 1/2) and ablation mode (Original/Ablated)

### Data Layer
- [x] Create `compute_position_layer_matrix()` in `utils/model_patterns.py`
- [x] Reuse existing `slice_data()` logic for per-position slicing
- [x] Return 2D array: `[num_layers, seq_len]` of delta values

### Heatmap Component
- [x] Create Plotly `go.Heatmap` with `Blues` colorscale
- [x] Set X-axis labels to token strings
- [x] Set Y-axis labels to layer numbers (reversed for bottom-up)
- [x] Add hover template showing token, layer, delta

### UI Integration
- [x] Remove scrubber container from `components/main_panel.py`
- [x] Add `html.Div(id='heatmap-container')` in its place
- [x] Create callback to render heatmap from activation data
- [x] Add toggle buttons for comparison/ablation modes

### Modal Interaction
- [x] Create modal component (reuse accordion content structure)
- [x] Add callback on heatmap `clickData` to extract (layer, position)
- [x] Populate modal with top-5 chart, attention viz, deltas
- [x] Add close button and context header