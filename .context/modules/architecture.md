# Architecture

## System Overview

A Plotly Dash single-page application that visualizes transformer LLM internals and enables interactive experimentation. Users select a model, enter a prompt, and explore a five-stage pipeline: Tokenization → Embedding → Attention → MLP → Output.

## Component Map

```
app.py                    # Entry point. Dash layout + all callbacks (~1450 lines)
├── components/
│   ├── sidebar.py        # Collapsible left panel: glossary, attention/block/norm dropdowns
│   ├── model_selector.py # Model dropdown + prompt textarea + generation settings
│   ├── pipeline.py       # 5-stage expandable pipeline with flow indicators
│   ├── investigation_panel.py  # Tabs: Ablation and Token Attribution
│   ├── ablation_panel.py      # Head selection, run ablation, original vs ablated comparison
│   ├── chatbot.py        # Floating chat icon + window + RAG-aware conversation
│   └── glossary.py       # Modal with transformer terms and video links
├── utils/
│   ├── model_patterns.py     # Model loading, forward pass, head ablation, bertviz, logit lens
│   ├── model_config.py       # Model family definitions, module templates, auto-selections
│   ├── head_detection.py     # Categorize heads (Previous Token, Induction, etc.)
│   ├── beam_search.py        # Beam search with optional ablation hooks
│   ├── token_attribution.py  # Integrated Gradients and simple gradient attribution
│   ├── ablation_metrics.py   # KL divergence, sequence scoring, token probability deltas
│   ├── openrouter_client.py  # OpenRouter API client for chat + embeddings
│   ├── rag_utils.py          # RAG: load/chunk rag_docs/, embed, retrieve
│   └── head_categories.json  # Static head category definitions
├── assets/
│   ├── style.css         # Custom styling (Bootstrap-compatible)
│   └── chat_resize.js    # Client-side chat window resize
├── rag_docs/             # ~30 markdown files: chatbot knowledge base
├── tests/                # pytest suite (~12 test files)
└── scripts/
    └── analyze_heads.py  # One-off analysis script
```

## Data Flow

1. **User selects model** → `model_patterns.load_model()` downloads/caches HF model
2. **User enters prompt** → Forward pass captures activations at each pipeline stage
3. **Pipeline renders** → Each stage shows its visualization (tokens, embeddings, attention maps, MLP, logits)
4. **Beam search** → `beam_search.perform_beam_search()` generates continuations with top-k display
5. **Experiments** → Ablation disables selected heads and re-runs; Attribution computes token importance via gradients
6. **Chatbot** → User question → RAG retrieval from `rag_docs/` → OpenRouter API → streamed response

## State Management

Dash `dcc.Store` components hold session state: activations, patterns, beam results, ablation state. No server-side session persistence — everything is per-page-load.

## Deployment

- Dockerfile targeting Hugging Face Spaces (port 7860)
- `.env` for `OPENROUTER_API_KEY` (not committed)
- Models cached locally on first load

## Key Boundaries

- **components/** only builds Dash layout — no ML logic
- **utils/** handles all computation — no Dash imports
- **app.py** is the glue: callbacks wire components to utils
- **rag_docs/** is the chatbot's knowledge base — edit these to change what the chatbot knows
