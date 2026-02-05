# Introduction to Mechanistic Interpretability

## What Is Mechanistic Interpretability?

Mechanistic interpretability (often called "mech interp") is a field of AI research that aims to understand **how** neural networks work internally -- not just what they predict, but why. Instead of treating models as black boxes, researchers open them up and study the individual components (neurons, attention heads, layers) to figure out what each one does.

This dashboard is a tool for doing exactly that kind of investigation.

## How This Dashboard Relates

The experiments available in this dashboard are real techniques used in mechanistic interpretability research:

- **Ablation** (removing heads to test their importance) is a standard tool for identifying which components are responsible for specific behaviors
- **Token attribution** (measuring input influence via gradients) is used to trace how information flows from inputs to outputs
- **Attention pattern analysis** (categorizing heads by behavior) helps researchers build a map of what each head does
- **Head categorization** (Previous-Token, BoW, Syntactic, etc.) builds on research that has identified recurring head types across models

## Key Concepts in the Field

### Circuits

A **circuit** is a small subnetwork within the model that performs a specific function. For example, researchers have found "induction circuits" -- combinations of attention heads across layers that work together to complete patterns like "A B ... A" → "B" (if the model has seen "A B" before, when it sees "A" again, it predicts "B").

In the dashboard, you can start to identify circuits by ablating combinations of heads and seeing which combinations have outsized effects.

### Superposition

**Superposition** is the idea that neural networks represent more features than they have dimensions. A 768-dimensional embedding might encode thousands of different concepts by overlapping them. This makes interpretation challenging because a single neuron can participate in many features.

### Induction Heads

**Induction heads** are one of the best-understood circuits. They are pairs of attention heads (typically one in an early layer and one in a later layer) that work together to copy patterns from context. If the model has seen "Harry Potter" earlier in the text and encounters "Harry" again, induction heads help it predict "Potter."

You might observe induction-like behavior in the dashboard when using prompts with repeated patterns.

### Polysemanticity

Neurons and heads are often **polysemantic** -- they respond to multiple unrelated features. An attention head might handle both pronoun resolution and list formatting, depending on the input. This is why head categories are approximate: the same head may behave differently for different prompts.

## Notable Research Groups

These organizations have published influential work in mechanistic interpretability:

- **Anthropic**: Published foundational work on transformer circuits, superposition, and dictionary learning for interpreting neural networks
- **EleutherAI**: Open-source AI research group that has contributed tools and analysis for model interpretability
- **Redwood Research**: Focuses on alignment-relevant interpretability, including causal interventions on model behavior
- **DeepMind (Google)**: Research on understanding internal representations and how models store knowledge

## Further Reading

If you want to explore the research behind this dashboard's techniques:

- "A Mathematical Framework for Transformer Circuits" (Elhage et al., Anthropic) -- foundational paper on how attention heads compose into circuits
- "In-context Learning and Induction Heads" (Olsson et al., Anthropic) -- how models learn to copy patterns from context
- "Locating and Editing Factual Associations in GPT" (Meng et al.) -- how facts are stored in MLP layers
- "Attention Is All You Need" (Vaswani et al., 2017) -- the original Transformer paper

These papers are referenced here for context. The dashboard provides a hands-on way to explore many of the concepts they describe.
