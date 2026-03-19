# Educational Philosophy

## Audience

Primary: ML students and AI enthusiasts — people who are curious but may not have a full college-level CS/math background.

Secondary: Educators looking for interactive tools to teach transformer concepts.

## Core Principles

### Conceptual Understanding Over Mathematical Rigor
It is acceptable to skip complex math (e.g., full derivations of scaled dot-product attention) as long as the motivation and intuition are clearly communicated. The goal is "I understand what this does and why" not "I can derive this from scratch."

### Action-Oriented Learning
Every architectural explanation should be paired with an interactive element. Don't just tell — let users poke at things and see what happens.

### Progressive Disclosure
- **Surface level**: Clean interface, minimal jargon, tooltips for technical terms
- **Mid level**: In-situ descriptions followed by interactive examples
- **Deep level**: Glossary entries, video links, chatbot for open-ended questions

### Terminology Convention (Jargon Reduction)
- UI headings and button labels use plain English as the primary term
- Technical terms appear in parentheses or as italicized secondary notes: e.g., "Test by Removing (Ablation)"
- Key renames: Tokenization → Text Splitting, Embedding → Meaning Encoding, MLP → Knowledge Retrieval, Ablation → Test by Removing, Token Attribution → Word Influence, Attention Heads → Detectors
- Identifier notation: L#-D# (Layer-Detector) instead of L#-H# (Layer-Head)
- RAG docs (rag_docs/) still use original terminology — a future pass may update those

### Framing
- Speak to curiosity: "What happens if...?" and "How does this work?"
- Tone is enthusiastic and accessible but concise — no walls of text
- Frame experiments as hypothesis testing: "What if I disable this head?"

## Sanity Check Rule

Before shipping any educational content or visualization change, ask:
1. Does this actually help someone understand transformers better?
2. Is this accurate enough to build correct intuition (even if simplified)?
3. Am I going down a rabbit hole, or is this genuinely useful?

Tunnel vision leads to bad outputs. Step back regularly.

## Visual Consistency

- Attention head indices, layer numbers, and token highlights must be consistent across all panels
- Use consistent color language for different components (attention vs MLP)
- Support both light and dark modes with high contrast for data visualizations
