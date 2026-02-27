# RAG Documents

This folder contains documents used by the AI chatbot for Retrieval-Augmented Generation (RAG).

## Supported File Types

- `.txt` - Plain text files
- `.md` - Markdown files

## How to Add Documents

1. Place your documentation files in this folder
2. Delete `embeddings_cache.json` if it exists (to force re-indexing)
3. The chatbot will automatically index new documents on startup
4. Documents are chunked and embedded for semantic search

## Document Inventory

### Category 1: General LLM/Transformer Knowledge
- `what_is_an_llm.md` - Neural networks, language models, next-token prediction
- `transformer_architecture.md` - Layers, encoder/decoder, residual stream
- `tokenization_explained.md` - Subword tokenization, BPE, token IDs
- `embeddings_explained.md` - Lookup tables, vector spaces, positional encodings
- `attention_mechanism.md` - Q/K/V, multi-head attention, intuitive explanations
- `mlp_layers_explained.md` - Feed-forward networks, knowledge storage, expand-compress
- `output_and_prediction.md` - Logits, softmax, temperature, greedy vs. sampling
- `key_terminology.md` - Extended glossary of ML/transformer terms

### Category 2: Dashboard Components
- `dashboard_overview.md` - Layout tour, navigation, typical workflow
- `pipeline_stages.md` - What each of the 5 pipeline stages shows
- `ablation_panel_guide.md` - How to use the ablation experiment panel
- `attribution_panel_guide.md` - How to use the token attribution panel
- `beam_search_and_generation.md` - Beam search, generation controls
- `head_categories_explained.md` - Previous-Token, Positional, BoW, Syntactic, Other
- `model_selector_guide.md` - Choosing models, auto-detection, generation settings

### Category 3: Model-Specific Documentation
- `gpt2_overview.md` - GPT-2 architecture, why it's a good starter, variants
- `gpt_neo_overview.md` - GPT-Neo architecture, local attention, comparison with GPT-2
- `pythia_overview.md` - Pythia architecture, RoPE, parallel attn+MLP, interpretability focus
- `opt_overview.md` - OPT architecture, ReLU activation, comparison with GPT-2
- `qwen_overview.md` - Qwen2.5 (LLaMA-like) architecture, RMSNorm, SiLU, GQA

### Category 4: Guided Experiments (Step-by-Step)
- `experiment_first_analysis.md` - Your first analysis with GPT-2
- `experiment_exploring_attention.md` - Reading attention patterns and head categories
- `experiment_first_ablation.md` - Removing a head and observing the effect
- `experiment_token_attribution.md` - Measuring token influence with gradients
- `experiment_comparing_heads.md` - Systematic comparison across head categories
- `experiment_beam_search.md` - Exploring alternative generation paths

### Category 5: Interpretation, Troubleshooting, and Research
- `interpreting_ablation_results.md` - How to read ablation probability changes
- `interpreting_attribution_scores.md` - Understanding attribution score values
- `interpreting_attention_maps.md` - Reading BertViz patterns visually
- `troubleshooting_and_faq.md` - Common issues and frequently asked questions
- `recommended_starting_points.md` - Best models, prompts, and experiment order
- `mechanistic_interpretability_intro.md` - Mech interp research context

## Notes

- Large files will be automatically chunked (~500 tokens per chunk)
- Embeddings are cached in `embeddings_cache.json` for faster subsequent loads
- Delete `embeddings_cache.json` to force re-indexing of all documents
