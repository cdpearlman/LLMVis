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

## Completed: Pipeline Clarity Improvements (Agent A)

- [x] Rename "Max New Tokens:" to "Number of New Tokens:" in app.py
- [x] Rename "Beam Width:" to "Number of Generation Choices:" in app.py
- [x] Remove score display from generated sequences in app.py
- [x] Update glossary to clarify "Number of Generation Choices" relates to Beam Search

## Completed: Pipeline Clarity Improvements (Agent E)

- [x] Modified output display to show full prompt with predicted token highlighted
- [x] Fixed top-5 tokens hover to show "Token (X%)" format instead of long decimals
- [x] Added Plotly hovertemplate for cleaner hover formatting

## Completed: Pipeline Clarity Improvements (Agent C)

- [x] Add educational explanation for embedding stage (pre-learned lookup table concept)
- [x] Add educational explanation for MLP stage (knowledge storage during training)
- [x] Add educational explanation for attention stage (how to interpret BertViz visualization)

## Completed: Pipeline Clarity Improvements (Agent B)

- [x] Convert tokenization from horizontal three-column layout to vertical rows
- [x] Each token row shows: [token] → [ID] → [embedding placeholder]
- [x] Maintain existing color scheme and educational tooltips
- [x] Update CSS styles for .tokenization-rows and .token-row layout
- [x] Add responsive styles for mobile (stack on small screens)

## Completed: Pipeline Clarity Improvements (Agent D)

- [x] Switch generate_bertviz_html from model_view to head_view in model_patterns.py
- [x] Deprecate _get_top_attended_tokens function (remove usage in extract_layer_data)
- [x] Add generate_head_view_with_categories function for categorized attention heads
- [x] Add get_head_category_counts helper function for UI display
- [x] Run tests to verify no regressions (73 tests pass)

## Completed: Pipeline Clarity Improvements (Agent F - Analysis Scope)

- [x] Add session-original-prompt-store and session-selected-beam-store to app.py
- [x] Modify run_generation to analyze ORIGINAL PROMPT only (not generated beam)
- [x] Store beam generation results separately for post-experiment comparison
- [x] Update analyze_selected_sequence to store beam for comparison instead of re-analyzing
- [x] Update ablation experiment to show selected beam context in results

## Completed: Pipeline Clarity Improvements (Agent G - Integration & Attention UI)

- [x] Remove deprecated "Most attended tokens" section from attention stage
- [x] Wire head categorization into attention stage UI (shows category counts)
- [x] Add enhanced navigation instructions for BertViz head view
- [x] Verify all 73 tests pass

## Completed: UI/UX Fixes (5 Issues)

### Issue 1: "Select for Comparison" Button
- [x] Update store_selected_beam callback in app.py to update UI
- [x] Clear all other generated sequences when one is selected
- [x] Display selected sequence with "Selected for Comparison" badge

### Issue 2: Tokenization Vertical Layout
- [x] Modify create_tokenization_content in pipeline.py to use vertical layout
- [x] Each row displays: [token] → [ID] with header row

### Issue 3: Expandable Attention Categories
- [x] Convert category chips to expandable `<details>` elements in pipeline.py
- [x] Update app.py to pass full categorize_all_heads() data instead of counts
- [x] Show list of heads (L0-H3, L2-H5, etc.) when category is expanded

### Issue 4: BertViz Navigation Instructions
- [x] Add single-click explanation: selects/deselects that head
- [x] Add double-click explanation: selects only that head (deselects others)

### Issue 5: Multi-Layer Ablation Head Selection
- [x] Change ablation-selected-heads store to hold [{layer, head}, ...] objects
- [x] Add create_selected_heads_display function in investigation_panel.py
- [x] Show selected heads as chips with "x" buttons to remove
- [x] Update head buttons to show visual selection state per layer
- [x] Preserve selections across layer dropdown changes
- [x] Update run_ablation_experiment to handle multi-layer ablation
- [x] Verify all 73 tests pass

## Completed: Fix Multi-Layer Ablation Bug

- [x] Create `execute_forward_pass_with_multi_layer_head_ablation` in model_patterns.py
- [x] Export new function in utils/__init__.py
- [x] Replace per-layer ablation loop in app.py with single call to new function
- [x] Add 5 tests for multi-layer ablation in test_model_patterns.py
- [x] Verify all 78 tests pass

## Completed: Codebase Cleanup

- [x] Delete unused file: `components/tokenization_panel.py` (302 lines, 6 functions)
- [x] Remove 6 unused imports from `app.py`
- [x] Remove deprecated `_get_top_attended_tokens()` function from model_patterns.py
- [x] Remove `top_attended_tokens` field from extract_layer_data() return values
- [x] Remove unused `create_stage_summary()` function from pipeline.py
- [x] Remove 7 unused utility functions from utils/:
  - `get_check_token_probabilities`
  - `execute_forward_pass_with_layer_ablation`
  - `generate_category_bertviz_html`
  - `generate_head_view_with_categories`
  - `compute_sequence_trajectory`
  - `compute_layer_wise_summaries`
  - `compute_position_layer_matrix`
- [x] Update `utils/__init__.py` exports
- [x] Update README.md to remove reference to deleted file

## Completed: AI Chatbot Integration

- [x] Create `rag_docs/` folder with placeholder README for RAG documents
- [x] Create `utils/gemini_client.py` with Gemini API wrapper (generate + embed)
- [x] Create `utils/rag_utils.py` with document loading, chunking, and retrieval
- [x] Create `components/chatbot.py` with UI components (icon, window, messages)
- [x] Add chatbot CSS to `assets/style.css` (floating button, chat window, message bubbles)
- [x] Modify `app.py` to add chat layout and callbacks
- [x] Add `google-generativeai` to `requirements.txt`
- [x] Test end-to-end: toggle, message send, context awareness
- [x] Verify all 81 tests pass

## Completed: Hugging Face Deployment Prep

- [x] Create `.gitignore` to exclude `.env`, `__pycache__/`, etc.
- [x] Add `load_dotenv()` to `app.py` for local development
- [x] Create `tests/test_gemini_connection.py` to verify API key connectivity
- [x] Tests verify: API key is set, can list models, flash model available
- Note: On Hugging Face Spaces, set `GEMINI_API_KEY` in Repository Secrets

## Completed: Migrate to New Google GenAI SDK (Superseded)

- [x] Update `requirements.txt`: `google-generativeai` → `google-genai>=1.0.0`
- [x] Rewrite `utils/gemini_client.py` using new centralized Client architecture
- [x] All 4 connection tests pass
- [x] Verified: embeddings work (3072 dimensions), chat generation works

## Completed: Migrate from Gemini to OpenRouter

- [x] Create `utils/openrouter_client.py` with OpenAI-compatible API
  - Global model config: `DEFAULT_CHAT_MODEL` and `DEFAULT_EMBEDDING_MODEL`
  - Chat via: `POST /api/v1/chat/completions`
  - Embeddings via: `POST /api/v1/embeddings`
- [x] Update `utils/rag_utils.py` imports to use openrouter_client
- [x] Update `app.py` imports to use openrouter_client
- [x] Create `tests/test_openrouter_connection.py` for API connectivity tests
- [x] Delete old `utils/gemini_client.py` and `tests/test_gemini_connection.py`
- [x] Update `requirements.txt`: remove `google-genai`, add `requests>=2.28.0`
- [x] Environment variable: `GEMINI_API_KEY` → `OPENROUTER_API_KEY`

## Completed: Switch to Paid OpenRouter Models (Cost-Optimized)

- [x] Evaluate OpenRouter models for chatbot use case (cost vs quality)
- [x] Switch chat model: `google/gemini-2.5-flash-lite`
  - $0.10/$0.40 per 1M tokens (input/output)
  - 1M context window, 318 tok/s, multimodal
- [x] Switch embedding model: `openai/text-embedding-3-small`
  - $0.02 per 1M tokens
  - 1536 dimensions, high quality
- [x] Remove local `sentence-transformers` dependency (simpler, no TF conflicts)
- [x] Estimated cost: ~$1.50/month for moderate usage

## Completed: Enhance RAG Documents for Chatbot

- [x] Category 1: 8 general LLM/Transformer knowledge files (what_is_an_llm.md through key_terminology.md)
- [x] Category 2: 7 dashboard component documentation files (dashboard_overview.md through model_selector_guide.md)
- [x] Category 3: 3 model-specific documentation files (gpt2_overview.md, llama_overview.md, opt_overview.md)
- [x] Category 4: 6 step-by-step guided experiment files (experiment_first_analysis.md through experiment_beam_search.md)
- [x] Category 5: 6 interpretation/troubleshooting/research files (interpreting_*.md, troubleshooting_and_faq.md, recommended_starting_points.md, mechanistic_interpretability_intro.md)
- [x] Delete embeddings_cache.json, update rag_docs/README.md with full inventory
- [x] Update todo.md and conductor docs
- Total: 30 RAG documents covering transformer concepts, dashboard usage, guided experiments, interpretation, troubleshooting, and research context
