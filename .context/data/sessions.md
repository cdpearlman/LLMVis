# Session Log

<!-- Append-only. Add a new entry after each substantive work session. -->

## 2026-03-02 — Bootstrap
**Area**: Project setup
**Work done**: Ran ContextKit bootstrap interview, generated memory system
**Decisions made**: Established educational philosophy (conceptual understanding over math rigor), dead code cleanup as mandatory during refactors, agent behavior (push back on bad ideas, no yes-man behavior)
**Memory created**: architecture.md, conventions.md, education.md, testing.md, sessions.md, decisions.md, lessons.md
**Open threads**: None — ready for feature work

## 2026-03-17 — Jargon Reduction (UI Accessibility)
**Area**: UI text, educational content
**Work done**: Verified RESEARCH-jargon-reduction.md claims against actual code. Replaced technical jargon across 6 files (app.py, ablation_panel.py, chatbot.py, glossary.py, investigation_panel.py, pipeline.py) + head_detection.py. Pushed back on inaccurate proposals ("Feature Spotter" for heads, "Number Dictionary" for embedding). Changed L#-H# notation to L#-D# for consistency.
**Decisions made**: "Detector" chosen over "Feature Spotter" for attention heads; technical terms kept as parentheticals rather than removed entirely.
**Commits**: abf6a1c, deb2071 (on `jargon-reduction` branch)
**Open threads**: Branch not yet merged to main. Chatbot suggestion text updated but RAG docs in rag_docs/ still use old terminology (deferred per research doc).

## 2026-03-19 — Fix dtype Mismatch (Gibberish Output & None Crashes)
**Area**: Model loading, numerical stability, error handling
**Work done**: Diagnosed and fixed chain of dtype-related failures causing gibberish output on HF Space for non-GPT-2 models. Created `load_model_for_inference()` helper forcing float32 + weight-tying verification. Replaced all 6 `from_pretrained` call sites (5 in app.py, 1 in model_patterns.py). Added None guards for probabilities and attention weights. Added dtype cast in logit_lens. 5 new tests, all 113 pass.
**Commits**: 5528a77
**Open threads**: Needs HF Space deployment verification (Pythia, Qwen, OPT).
