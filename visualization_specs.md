# Implementation Specification: Transformer Activation Visualization Dashboard

## 1. High-Level Overview & Goals

A Plotly Dash application for model-agnostic visualization of transformer internals. Users select a model and prompt, discover model-specific module/parameter names, and then visualize: (a) per-layer mini attention views (BertViz-style) and (b) connections between adjacent layers representing the top-3 predicted next tokens per layer from a logit-lens projection. The app runs activation capture and logit-lens analysis inside the server, caches heavy results per session using dash-extensions ServersideOutputTransform, supports CPU-only execution, and scales the layout horizontally (up to 4 rows). Success is a responsive, understandable UI that enables exploration without manual pre-processing or code edits.

## 2. System Architecture & Workflow

```mermaid
graph TD
A[User: Top Bar selects Model + Prompt] --> B[Click: Find Module Names]
B --> C[Server: Load/Reuse Model + Tokenizer (CPU)]
C --> D[Server: Discover module patterns & parameter names]
D --> E[Populate Sidebar Dropdowns<br/>(Attention Pattern, MLP Pattern, Norm Param, Logit-Lens Param)]
E --> F[User: Select 1 Attention Pattern + 1 MLP Pattern + 1 Norm Param + 1 Logit-Lens Param]
F --> G[Click: Visualize]
G --> H[Server: Expand patterns -> per-layer module names]
H --> I[Server: Activation Capture (hooks) on selected modules]
I --> J[Server: Prepare attention weights + MLP outputs + norm param + selected logit-lens weight]
J --> K[Server: Logit-Lens compute top-3 tokens per layer (last token position)]
K --> L[Cache Results (ServersideOutputTransform)<br/>Key=model+prompt+selections+top_k, TTL=session]
L --> M[Main Dashboard: Render layer tiles + edges]
M --> N[User: Click layer tile]
N --> O[Server: On-demand BertViz renders<br/>(Tab 1: Layer head_view, Tab 2: Full model_view)]
O --> P[Cache per-session BertViz HTML for reuse]
```

3. File Breakdown

app/app.py
• Objective: Dash entrypoint. Configures dash-extensions ServersideOutputTransform, builds layout, registers callbacks, initializes in-memory session cache, and manages model registry.

app/components/top_bar.py
• Objective: Top bar controls for model selection ("gpt2", "Qwen/Qwen2.5-0.5B"), prompt input, and "Find Module Names" button.

app/components/sidebar.py
• Objective: Left sidebar with dropdowns for single selections: Attention Pattern, MLP Pattern, Normalization Parameter, Logit-Lens Parameter, and the "Visualize" button.

app/components/dashboard.py
• Objective: Main dashboard container: responsive horizontal grid (up to 4 rows) of layer tiles and curved inter-layer edges; handles scrolling for overflow.

app/components/layer_tile.py
• Objective: Renders a single layer tile (300x200) with a lightweight thumbnail mini-view of that layer’s attention; clickable to open the modal.

app/components/modals.py
• Objective: Modal with tabs: Tab 1 shows full head_view for the selected layer; Tab 2 shows the full multi-layer model_view.

app/services/model_registry.py
• Objective: Manage per-model instances (model/tokenizer) in memory on CPU; reuse across callbacks; lazy-load on first use.

app/services/module_discovery.py
• Objective: Discover module numeric patterns (generalized with {N}), categorize patterns (attention/MLP/other), discover parameter names and categorize (logit-lens/norm/other); produce sorted dropdown options.

app/services/activation_pipeline.py
• Objective: Expand selected patterns to all layer instances; register hooks; run forward pass; capture attention and MLP outputs; capture selected norm parameter; compute top-3 tokens via logit-lens using selected parameter; structure results per the activation_capture.py schema.

app/services/bertviz_renderer.py
• Objective: Extract attention weights for a single layer; generate: (a) lightweight thumbnail (static/simplified), (b) full interactive BertViz head_view for a layer, and (c) full model_view; returns HTML strings or image data.

app/services/caching.py
• Objective: Compose cache keys; wrap ServersideOutput outputs; session-scoped storage of heavy results (activations, per-layer attentions, top-k tokens, BertViz HTML).

app/utils/data_contract.py
• Objective: Validate and normalize the activation_capture-compatible data structures; provide robust read helpers (e.g., norm_parameter as a list with first element used).

app/utils/layout_math.py
• Objective: Compute even distribution of N layers across up to 4 rows; calculate tile positions and edge paths (curved).

app/utils/token_formatting.py
• Objective: Format tokens and probabilities for hover/labels consistent with current logit_lens_analysis printing.

app/constants.py
• Objective: Centralize constants: top_k=3, min_edge_opacity=0.15, curve_factor=0.5, tile_size=(300,200), max_prompt_words=50.

assets/styles.css
• Objective: CSS for sizing, grid rows, scroll behavior, tile hover, modal sizing, and BertViz embed styling.

4. Function Definitions

app/services/model_registry.py
get_model_and_tokenizer(model_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]
• Purpose: Load or reuse a CPU model and tokenizer for the given model name.
• Inputs:
    ◦ model_name: str: "gpt2" or "Qwen/Qwen2.5-0.5B".
• Output: (model, tokenizer)
• Error Handling: Raises on HF load errors; caches successful instances in memory keyed by model_name.

app/services/module_discovery.py
detect_numeric_patterns(model) -> dict[str, list[str]]
• Purpose: Map generalized pattern strings with {N} to concrete module names.
• Inputs:
    ◦ model: AutoModelForCausalLM
• Imports:
    ◦ torch / transformers typing only
• Output: pattern_to_modules
• Error Handling: Skips empty names; robust to nonstandard module names.

categorize_module_patterns(pattern_to_modules: dict[str, list[str]]) -> tuple[list[str], list[str], list[str]]
• Purpose: Categorize generalized module patterns into attention, mlp, and other (by heuristics).
• Inputs:
    ◦ pattern_to_modules
• Output: (attn_patterns, mlp_patterns, other_patterns)

detect_parameter_names(model) -> dict[str, list[str]]
• Purpose: Map parameter patterns (generalized {N}) to parameter names.
• Inputs:
    ◦ model
• Output: pattern_to_parameters

categorize_parameter_patterns(pattern_to_parameters: dict[str, list[str]]) -> tuple[list[str], list[str], list[str]]
• Purpose: Categorize parameter patterns into logit-lens, norm, and other (heuristics).
• Output: (logit_lens_patterns, norm_patterns, other_patterns)

build_dropdown_options(attn_patterns: list[str], mlp_patterns: list[str], other_patterns: list[str], pattern_to_parameters: dict[str, list[str]]) -> dict
• Purpose: Build UI-ready options.
• Inputs:
    ◦ attn_patterns, mlp_patterns, other_patterns
    ◦ pattern_to_parameters
• Output: {
    attention_pattern_options: list[str],   // generalized with {N}, attention-first then others
    mlp_pattern_options: list[str],         // generalized with {N}, mlp-first then others
    norm_param_options: list[str],          // fully-qualified parameter names (flattened)
    logit_param_options: list[str]          // fully-qualified parameter names (flattened)
}
• Error Handling: Ensures non-empty sets; otherwise returns empty lists with user-facing warnings.

app/services/activation_pipeline.py
expand_pattern_to_modules(selected_pattern: str, pattern_to_modules: dict[str, list[str]]) -> list[str]
• Purpose: Expand a single generalized pattern to all concrete per-layer module names.

capture_activations(model, tokenizer, prompt: str, attention_modules: list[str], mlp_modules: list[str]) -> tuple[dict, dict, torch.Tensor]
• Purpose: Register forward hooks on given modules, run a no-cache forward (use_cache=False), and capture outputs.
• Inputs:
    ◦ model, tokenizer, prompt, attention_modules, mlp_modules
• Output: (attention_outputs, mlp_outputs, input_ids)
• Error Handling: Nonexistent module names are logged and skipped; capture errors stored per-module.

capture_norm_parameter(model, param_name: str) -> list[Any]
• Purpose: Capture the single selected norm parameter; returned as a list with exactly one serialized tensor to match current schema.

build_capture_payload(model_name: str, prompt: str, attention_modules: list[str], attention_outputs: dict, mlp_modules: list[str], mlp_outputs: dict, norm_param: list[Any], logit_lens_param_name: str) -> dict
• Purpose: Create the activation_capture-compatible JSON object (in-memory).

load_logit_lens_weights(model_name: str, param_name: str) -> torch.Tensor
• Purpose: Load the selected projection parameter tensor from the model by name (model-agnostic; user-selected).

extract_mlp_outputs_for_lens(data: dict) -> list[torch.Tensor]
• Purpose: Convert stored MLP outputs into a list of tensors (layer order aligned with mlp_modules).

extract_norm_weight(data: dict) -> torch.Tensor
• Purpose: Read norm_parameter list and return the first (only) element as a tensor.

apply_logit_lens(mlp_outputs: list[torch.Tensor], logit_lens_weight: torch.Tensor, norm_weight: torch.Tensor, tokenizer: AutoTokenizer, top_k: int = 3) -> list[list[tuple[str, float]]]
• Purpose: Normalize hidden states with the selected norm weight, project via selected logit-lens weight, softmax, and take top-k for last token position.
• Output: For each layer: [(token_str, probability), ...] length top_k.
• Error Handling: Validates dimensions; raises if tensors mismatched.

format_topk_for_hover(layer_results: list[tuple[str, float]]) -> list[str]
• Purpose: Return strings formatted like current logit_lens_analysis:
    - prob < 0.001 => 6 decimals; else 3 decimals.
• Output: ["token(0.123)", "▁the(0.045)"] etc.

compute_edge_opacity(prob: float, top1_prob: float, min_opacity: float = 0.15) -> float
• Purpose: Map probability to opacity: opacity = max(min_opacity, 0.15 + 0.85 * (prob / top1_prob)).
• Output: float in [min_opacity, 1.0].

app/services/bertviz_renderer.py
get_tokens_from_input_ids(tokenizer: AutoTokenizer, input_ids: torch.Tensor) -> list[str]
• Purpose: Convert input_ids to token strings for visualization; preserve printable formatting.

extract_single_layer_attention(attention_outputs: dict, module_name: str) -> torch.Tensor
• Purpose: Extract attention weights for a given layer module (expects tuple outputs where weights are element 1).

render_layer_thumbnail(attention_weights: torch.Tensor, tokens: list[str], size: tuple[int, int] = (300, 200)) -> str
• Purpose: Produce a lightweight thumbnail (e.g., base64 image or simplified Plotly figure HTML) to embed in the tile.

render_layer_head_view(attention_weights: torch.Tensor, tokens: list[str]) -> str
• Purpose: Produce the full interactive BertViz head_view HTML for a single layer.

render_full_model_view(all_attention_weights: list[torch.Tensor], tokens: list[str]) -> str
• Purpose: Produce the full interactive BertViz model_view HTML across all layers.

app/services/caching.py
compose_cache_key(params: dict) -> str
• Purpose: Deterministically hash the tuple:
    (model_name, normalized_prompt_text, attention_pattern, mlp_pattern, norm_param_name, logit_lens_param_name, tokenizer_version, top_k=3)

get_session_id() -> str
• Purpose: Retrieve or assign a per-session identifier for session-scoped caching.

app/components/dashboard.py
compute_grid_positions(num_layers: int, max_rows: int = 4) -> list[tuple[int, int]]
• Purpose: Evenly distribute layer indices across up to 4 rows, row-major left→right; returns (row, col) per layer index.

compute_edge_paths(positions: list[tuple[int, int]], curve_factor: float = 0.5) -> list[dict]
• Purpose: For each adjacent pair (k→k+1), compute 3 curved edge paths (Bezier) corresponding to top-3 tokens per layer; map opacity via compute_edge_opacity.

build_dashboard_figure(tiles: list[dict], edges: list[dict]) -> plotly.graph_objs.Figure
• Purpose: Assemble layer tiles and edges into a Plotly figure with hover tooltips.

app/components/top_bar.py
validate_prompt_length(raw_prompt: str, max_words: int = 50) -> tuple[str, bool, str]
• Purpose: Enforce word-count limit by truncation to first 50 words (with a non-blocking notice).
• Output: (possibly_truncated_prompt, truncated: bool, message)

app/app.py (callbacks)
on_find_module_names(model_name: str, prompt: str) -> dict
• Purpose: Load/reuse model+tokenizer, discover patterns and parameter names, and return dropdown options plus prompt info.

on_visualize(model_name: str, prompt: str, selections: dict) -> ServersideOutput
• Purpose: Expand patterns, run activation capture, prepare data, run logit-lens, cache results; return a handle for server-side stored data to render main dashboard.

on_open_layer_modal(layer_index: int) -> dict
• Purpose: Generate on-demand BertViz for the selected layer (tab 1) and full model_view (tab 2); cache HTML strings per session.

5. Execution Workflow

1. Input
- User selects:
  - Model: "gpt2" or "Qwen/Qwen2.5-0.5B"
  - Prompt: free text; enforce 50-word limit by truncating to first 50 words with a notice
- User clicks “Find Module Names”

2. Discovery
- Server loads or reuses model/tokenizer on CPU.
- Detect generalized module patterns with {N}, categorize into attention/MLP/other.
- Detect parameter names; categorize into logit-lens/norm/other.
- Populate sidebar:
  - Attention Pattern: single-select, attention-first then others, display generalized pattern with {N}
  - MLP Pattern: single-select, mlp-first then others, display generalized pattern with {N}
  - Normalization Parameter: single-select, fully-qualified parameter names
  - Logit-Lens Parameter: single-select, fully-qualified parameter names

3. Visualization
- User chooses one attention pattern, one mlp pattern, one norm parameter, and one logit-lens parameter, then clicks “Visualize”.
- Server expands patterns to all per-layer module names.
- Activation capture (hooks) runs a forward pass with use_cache=False; captures:
  - attention_outputs: per-module outputs, from which attention weights are extracted
  - mlp_outputs: per-module outputs as serialized tensors
  - norm_parameter: list containing the single selected parameter (unchanged schema)
  - logit_lens_parameter: stored as the selected parameter name
- Server builds in-memory activation_capture-compatible payload.
- Server loads the selected logit-lens weight tensor by name.
- Server extracts mlp_outputs as tensors, extracts norm weight (first element of list), applies normalization and projection, computes top-3 tokens at the last token position for each layer.
- Compute per-layer opacities for the 3 edges:
  - opacity = max(0.15, 0.15 + 0.85 * (prob / top1_prob_for_that_layer))
- Compose hover labels using current logit_lens_analysis formatting:
  - prob < 0.001 → 6 decimals; else 3 decimals
- Cache results via ServersideOutputTransform (in-memory, session-scoped).
- Render main dashboard:
  - Grid: horizontal left→right tiles (300×200), evenly distributed across up to 4 rows; horizontal/vertical scrolling if needed
  - Each tile: thumbnail of that layer’s attention
  - Edges: for each adjacent layer pair, 3 curved lines (Bezier) with curvature 0.5, grayscale opacity as above; hover shows token and probability

4. Drill-down
- Clicking a layer tile opens a modal:
  - Tab 1: Full interactive head_view for selected layer
  - Tab 2: Full interactive model_view across all layers
- Full model_view is generated on demand, cached per session.

5. Final Output
- The dashboard shows:
  - Tiled per-layer mini-views (attention)
  - Curved edges between adjacent layers (top-3 tokens), grayscale with min opacity 0.15
  - Hover tooltips: token and probability formatted as in current logit_lens_analysis
- The modal provides deep inspection via BertViz tabs.

6. Caching & Keys
- Caching: dash-extensions ServersideOutputTransform, in-memory, session-scoped TTL (cleared when session ends or app restarts).
- Key: Hash of
  - model_name
  - normalized_prompt_text
  - attention_pattern
  - mlp_pattern
  - norm_param_name
  - logit_lens_param_name
  - tokenizer_version
  - top_k=3
