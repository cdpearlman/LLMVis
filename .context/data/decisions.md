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
