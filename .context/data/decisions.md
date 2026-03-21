# Decision Record

<!-- Append-only. Record significant decisions with reasoning. -->
<!-- Format:
## [Decision title]
**Date**: YYYY-MM-DD
**Context**: What prompted this decision
**Options considered**: What alternatives were evaluated
**Decision**: What was chosen
**Reasoning**: Why
**Revisit if**: Conditions that would warrant reconsidering
-->

## Educational depth: conceptual over mathematical
**Date**: 2026-03-02
**Context**: Target audience includes people without full college-level CS/math education
**Options considered**: (1) Full mathematical rigor, (2) Conceptual understanding with simplified math, (3) No math at all
**Decision**: Conceptual understanding with simplified math — skip complex derivations, focus on motivation and intuition
**Reasoning**: The goal is building correct mental models, not producing textbook-ready proofs. Accurate simplification serves the audience better than intimidating formalism.
**Revisit if**: Audience shifts to researchers or grad students who need full rigor

## Chatbot backend: OpenRouter
**Date**: 2026-03-02
**Context**: Chatbot needed an LLM backend; previously used Gemini
**Options considered**: Gemini API, OpenRouter (multi-model), direct OpenAI
**Decision**: OpenRouter — provides access to multiple models through a single API
**Reasoning**: Flexibility to switch underlying models without code changes; single API key
**Revisit if**: OpenRouter pricing becomes prohibitive or a specific model provider offers significantly better educational responses

## Jargon replacement strategy: friendly primary + technical parenthetical
**Date**: 2026-03-17
**Context**: Research identified 6 categories of jargon that confuse younger audiences
**Options considered**: (1) Remove all technical terms, (2) Keep technical terms as primary with tooltips, (3) Lead with plain English, keep technical terms in parentheses
**Decision**: Option 3 — plain English headings with technical terms as parentheticals (e.g., "Test by Removing (Ablation)")
**Reasoning**: Completely removing terms loses educational value. Leading with jargon intimidates. Parentheticals let curious users learn the real term while keeping the UI approachable.
**Revisit if**: User testing shows parentheticals are confusing or ignored

## "Detector" chosen over "Feature Spotter" for attention heads
**Date**: 2026-03-17
**Context**: Research proposed "Feature Spotter" or "Context Detector" as head replacements
**Options considered**: (1) Feature Spotter, (2) Context Detector, (3) Detector, (4) Information Filter
**Decision**: "Detector" — short, accurate enough, pairs well with "L#-D#" notation
**Reasoning**: "Feature Spotter" is misleading (heads compute relationships between tokens, not features). "Detector" captures the idea of detecting patterns/relationships without implying feature extraction.
**Revisit if**: User testing shows "detector" is still confusing

## Verb-phrase chip labels completing a sentence stem
**Date**: 2026-03-21
**Context**: Example prompt chips needed labels that are immediately meaningful to non-technical users
**Options considered**: (1) Noun-phrase labels ("Object Tracking", "Word Agreement"), (2) Verb-phrase labels completing "See how models: ___"
**Decision**: Option 2 — verb phrases ("Track indirect objects", "Resolve ambiguity", etc.) with a sentence-stem intro
**Reasoning**: Noun labels read like category names and don't tell users what they'll learn. Verb phrases complete a natural sentence, making the purpose self-evident before clicking.
**Revisit if**: More than ~6 chips are added and the sentence-stem pattern becomes awkward

## Replace "Word Agreement" chip with "Resolve ambiguity"
**Date**: 2026-03-21
**Context**: "Word Agreement" tested subject-verb number agreement — a grammar concept most users won't connect to transformer behavior
**Options considered**: Keep with better label, replace with ambiguity resolution, replace with negation handling
**Decision**: Replace with "Resolve ambiguity" — prompt "The bat flew over the" tests how context disambiguates polysemous words
**Reasoning**: Ambiguity resolution is immediately intuitive (bat = animal or sports equipment), produces visible attention patterns (model attends back to "flew"), and the label completes the sentence stem naturally
**Revisit if**: User testing shows the prompt doesn't produce interesting enough patterns across models

## Centralized model loading with forced float32
**Date**: 2026-03-19
**Context**: Models like Pythia (float16) and Qwen (bfloat16) produced gibberish on CPU-only HF Space due to dtype instability; GPT-2 worked because it's natively float32
**Options considered**: (1) Add torch_dtype at each call site, (2) Centralize into a helper function, (3) Convert only at inference time
**Decision**: Option 2 — `load_model_for_inference()` in model_patterns.py, forces float32 and verifies weight tying
**Reasoning**: 6 call sites meant high risk of missing one; central helper is DRY and adds weight-tying safety net for tied-weight models
**Revisit if**: GPU deployment makes float16/bfloat16 desirable for performance
