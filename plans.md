# Transformer Visualization Dashboard — Implementation Plan

## Objective
Implement UX changes and new analysis features for the Dash + Cytoscape transformer visualization app with minimal, focused edits. Maintain a running to-do plan and strict git hygiene.

## Changes to Existing Workflow

### 1) Collapsible Sidebar (default collapsed)
- Description: Make the left sidebar collapsible. Default state is collapsed since modules/params are auto-detected.
- Steps:
  1. Add a small toggle control (icon button) to collapse/expand the sidebar container.
  2. Persist state for the session (e.g., in a `dcc.Store`).
  3. Ensure dropdowns still function while expanded; disabled/hidden while collapsed.
- Acceptance criteria:
  - Sidebar renders collapsed on initial page load.
  - Toggle expands/collapses without page refresh.
  - No layout shift breaks Cytoscape sizing.

### 2) Compare With Another Prompt
- Description: Add a button next to “Enter Prompt:” to open a second prompt input. When a second prompt is given, render an identical graph for the second prompt below the first.
- Steps:
  1. Add a “Compare With Another Prompt” button near the primary prompt input.
  2. When clicked, reveal a second `dcc.Textarea`/`dcc.Input`.
  3. On “Run Analysis”, if prompt 2 is present, run a second forward pass and render a second Cytoscape graph underneath the first.
- Acceptance criteria:
  - Button reveals a second prompt input.
  - Two independent graphs appear (top: prompt 1, bottom: prompt 2) with consistent styling and controls.
  - No interference between graphs’ hover/click callbacks.

### 3) “Check Token” input (adds 4th edge)
- Description: Above “Model Flow Visualization”, add “Check Token:” with a text input. If a word/token is provided, add a 4th edge between consecutive nodes representing that token’s probability.
- Steps:
  1. Add “Check Token:” label + input above the graph.
  2. On render, if input is non-empty, compute the probability of the provided token at each layer (using current logit lens path). 
  3. Draw the token-specific edge as the 4th edge between each pair of consecutive nodes. Enforce a minimum opacity for 0 probability.
- Acceptance criteria:
  - A distinct 4th edge for the user-specified token appears between each consecutive layer node pair.
  - Opacity has a minimum floor (visible even at 0 probability).
  - Works with single- and two-prompt modes.

## New Features

### 4) Replace BertViz head_view with model_view (with minor UI mods)
- Description: Use BertViz `model_view` instead of `head_view` for per-layer inspection when clicking a node; apply light styling tweaks.
- Steps:
  1. Replace `head_view` usage in `generate_bertviz_html` with `model_view`.
  2. Keep current tokenization pipeline; ensure tokens passed correctly to `model_view`.
  3. Embed returned HTML in the iframe as done currently.
- Acceptance criteria:
  - Node click shows `model_view` visualization for the correct layer.
  - No errors for GPT-2/Qwen2.5 0.5B models listed in app.

### 5) Attention head detection and categorization
- Description: Implement a detection function to categorize attention heads as: Previous-Token Heads, First/Positional-Token Heads, Bag-of-Words Heads, Syntactic Heads (plus “Other”). Show model_view visualizations grouped by category.
- Detection outline (initial heuristics, adjustable):
  - Previous-Token: high mass on (i → i-1).
  - First/Positional: high mass on BOS/first token positions or strong positional patterning.
  - Bag-of-Words: diffuse attention focused on content-bearing tokens, weak positional bias.
  - Syntactic: heads with consistent dependency-like patterns (approximate by distance patterns or punctuation/function-word anchors).
- Steps:
  1. Add a utility to compute per-head attention statistics per layer for a given prompt.
  2. Implement heuristic rules to assign categories.
  3. In the UI, present sections/columns for each category and list heads with labels like “L{layer}-H{head}”.
- Acceptance criteria:
  - Each head is assigned exactly one category; unclassified defaults to “Other”.
  - UI shows grouped model_view previews or links by category.
  - Heuristics are parameterized for later tuning.

### 6) Two-prompt difference analysis (heads + output probs)
- Description: If two prompts are given, compute differences across attention heads and output token probabilities, highlight layers with significant differences.
- Steps:
  1. For each layer/head, compute a similarity/distance (cosine similarity or L2 norm) between prompts’ attention distributions.
  2. For output token probabilities (logit lens), compute difference at each layer.
  3. Highlight layers with differences exceeding thresholds (default tunables: cosine distance ≥ 0.2 or normalized L2 ≥ 1.0).
  4. Indicate differences visually (e.g., red border/label) and provide a summary panel.
- Acceptance criteria:
  - When two prompts are provided, differing layers are highlighted in red in both graphs.
  - A concise summary lists top-N most divergent layers/heads.

## Constraints / Non-Goals
- Keep changes minimal; reuse existing components (`sidebar`, `main_panel`, utils). Avoid large refactors.
- No database or persistent storage changes.
- No new dependencies unless strictly required.

## Implementation Notes
- Graph edges: current top-3 token edges remain. The “Check Token” edge is additive (4th edge) with minimum opacity.
- Token input: treat user input as raw text; tokenize with the model tokenizer. If multiple sub-tokens arise, use the last sub-token for next-token prediction comparison (configurable later).

## Acceptance Test Checklist
- Sidebar toggles and defaults to collapsed.
- “Compare With Another Prompt” reveals input and renders two graphs.
- “Check Token” adds a visible 4th edge with minimum opacity.
- Node-click opens BertViz `model_view` without errors.
- Head categories rendered with labels by section.
- With two prompts, layers with large differences are highlighted.

## Process & Workflow
- Maintain `todo.md` as the single source of truth for step-by-step execution.
- After each cohesive change set: `git commit -am "[what changed]"`.
- Between features: push, then `git checkout -b [branch-name]`. Branches are never merged.
- The agent may propose small improvements and update this file and `todo.md`.
