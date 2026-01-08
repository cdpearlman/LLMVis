# Todo

## Simple Fixes

### Glossary Layout
- [ ] Add padding/margin to `.glossary-content` in `components/glossary.py`
- [ ] Test that text no longer reaches screen edges

### Collapsible Tokenization Example  
- [ ] Wrap static diagram in `html.Details()` in `components/tokenization_panel.py`
- [ ] Add `html.Summary("View example tokenization flow")`
- [ ] Default to collapsed state

### Fix Double Arrow (Transformer Layers)
- [ ] Add `.transformer-layers-summary::-webkit-details-marker { display: none; }` to `assets/style.css`
- [ ] Add `.transformer-layers-summary::marker { display: none; }` for Firefox

### Reorder Tokenization Section
- [ ] Move `create_tokenization_panel()` above scrubber container in `components/main_panel.py`

---

## Complex: Position x Layer Heatmap

### Design Spec
- X-axis: Token positions | Y-axis: Layers (0 at bottom)
- Color: Light→Dark blue | Metric: Top-token probability delta
- Click cell → Modal with layer details

### Data Layer
- [ ] Create `compute_position_layer_matrix()` in `utils/model_patterns.py`
- [ ] Reuse existing `slice_data()` logic for per-position slicing
- [ ] Return 2D array: `[num_layers, seq_len]` of delta values

### Heatmap Component
- [ ] Create Plotly `go.Heatmap` with `Blues` colorscale
- [ ] Set X-axis labels to token strings
- [ ] Set Y-axis labels to layer numbers (reversed for bottom-up)
- [ ] Add hover template showing token, layer, delta

### UI Integration
- [ ] Remove scrubber container from `components/main_panel.py`
- [ ] Add `html.Div(id='heatmap-container')` in its place
- [ ] Create callback to render heatmap from activation data

### Modal Interaction
- [ ] Create modal component (reuse accordion content structure)
- [ ] Add callback on heatmap `clickData` to extract (layer, position)
- [ ] Populate modal with top-5 chart, attention viz, deltas
- [ ] Add close button and context header