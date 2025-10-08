# UI Refactor: Panel-Based Layers (Accordion) — Actionable Tasks

Note: Minimal-change approach. Reuse existing files (`app.py`, `components/main_panel.py`, `utils/*`). Avoid new dependencies; use native `html.Details`/`html.Summary`, existing `dcc.Graph`, and current BertViz integration.

## Feature: Switch to panels with plain-language headers
- [x] Replace per-layer node display with `html.Details` (accordion) per layer in `components/main_panel.py`
- [x] Use `html.Summary` as header: `Layer L{N}: likely '{token}' (p={prob})`
- [x] Truncate long tokens in header with CSS (ellipsis); keep one-line summary
- [x] Add lightweight top-3 tokens "chips/arrows" between adjacent panel headers (no Cytoscape)
- [x] Gate old Cytoscape graph behind a feature flag (keep code path but hidden by default)

## Feature: Keep initial panels small to preserve flow view
- [ ] Style summary rows compactly (single-line, small font, consistent height)
- [ ] Ensure `html.Details` closed by default for all layers
- [ ] Add CSS utility classes for compact header + tokens chips row

## Feature: Per-layer predictions (top-5), deltas, certainty meter
- [x] Extend forward pass outputs to include per-layer top-5 tokens + probs (reusing logit lens) in `utils/model_patterns.py`
- [x] Compute delta vs previous layer for overlapping tokens (prob change, signed)
- [x] Compute certainty meter using normalized entropy over top-5 probs (0–1)
- [x] Render a `dcc.Graph` horizontal bar chart (top-5) inside each panel body
- [x] Show per-token delta as small ▲/▼ with color next to bars
- [x] Add tooltip explaining certainty: "certainty = 1 − H(p_top5)/log(5)"
- [x] Add a spinning "Loading visuals..." after loading data until all the visualizations are loaded

## Feature: Simplified attention view + open full interactive view
- [x] From `activation_data['attention_outputs']`, compute top-3 attended input tokens for current position (per layer)
- [x] Render a simple list: token text + attention weight (rounded), most-to-least
- [x] Add button/link: "Open full interactive view" → shows existing BertViz `model_view` for that layer
- [x] Keep current BertViz plumbing; reuse callbacks to load the selected layer

## Feature: Tokenization example of initial prompt (top of page)
- [ ] Add a new section above panels: "How the tokenizer splits your prompt"
- [ ] Tokenize only the main prompt; render chips with token text and hover for token id
- [ ] Add one-line explainer tooltip: what a token is; note about spaces/subwords

## Feature: Second prompt comparison (bar chart becomes comparison)
- [ ] When prompt 2 is present, render grouped bars (Prompt 1 vs Prompt 2) for top-5 in each layer
- [ ] Reuse existing comparison utilities where possible (`utils/prompt_comparison.py`)
- [ ] Add legend and consistent colors for the two prompts

## Feature: Divergence badge in headers (if 2nd prompt)
- [ ] Surface divergence result as a small badge in `html.Summary` (e.g., "Diverges")
- [ ] Reuse existing divergent-layer detection; no new heuristics

## Feature: Tooltips + collapsible "How to read this" help
- [ ] Add `title` tooltips for charts, certainty meter, deltas, and attention list
- [ ] Add small `html.Details` blocks inside panel body: "How to read this section"
- [ ] Keep content plain-language; 1–3 sentences each

## Feature: Expand/Collapse all controls
- [ ] Add two buttons above panels: "Expand all" and "Collapse all"
- [ ] Add callback to set all `html.Details` components `open` state accordingly

## Wiring & data plumbing
- [ ] Add per-layer data structure to `activation_data`: {layer -> top5, deltas, certainty}
- [ ] Ensure existing stores/callbacks pass required data to `components/main_panel.py`
- [ ] Keep performance: avoid heavy per-layer recalcs in callbacks; precompute on forward pass

## Styling & accessibility
- [ ] Use colorblind-safe palette; avoid color-only encodings (use ▲/▼ icons)
- [ ] Keep consistent axis scales across layers; show exact values on hover
- [ ] Ensure layout works on narrow screens (no horizontal scroll in headers)

## QA checklist
- [ ] Single prompt: panels render, bar charts show, certainty tooltips present
- [ ] Two prompts: grouped bars render; divergence badges show on appropriate layers
- [ ] "Open full interactive view" loads correct BertViz layer
- [ ] Expand/Collapse all works; individual panel state remains clickable
- [ ] Tokenization section matches tokenizer output; long tokens truncated safely


