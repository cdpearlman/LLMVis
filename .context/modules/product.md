# Product Definition

## Vision

Demystify the inner workings of Transformer-based LLMs for students and curious individuals. Combine interactive visualizations with hands-on experimentation to transform abstract architectural concepts into tangible, observable phenomena.

## Core Value Proposition

- **Visual Learning**: Translate complex matrix operations and data flows into clear, interactive representations (attention maps, logit lens)
- **Interactive Experimentation**: Go beyond observation — let users manipulate the model (ablation, activation patching) and immediately see consequences
- **Educational Scaffolding**: Support varying expertise levels with layered content, from tooltips to deep-dive glossaries to AI-guided chat

## Key Features

- **Sequential Data Flow Visualization**: Step-by-step data transformation through model layers
- **Component Breakdown**: Detailed inspection views for self-attention (heads, weights) and MLPs
- **Interactive Experiments**:
  - Ablation studies: selectively disable heads/layers to observe output impact
  - Activation steering: modify activation values in real-time
  - Prompt comparison: compare internal activations from different inputs side-by-side
- **Integrated Education**:
  - Contextual tooltips for immediate clarity
  - Glossary panel with in-depth definitions and video links
  - AI chatbot with RAG-powered knowledge base (30 docs covering transformer concepts, usage, experiments, troubleshooting, interpretability)
  - Step-by-step guided experiments for beginners

## Brand & Voice

- **Tone**: Enthusiastic and accessible yet concise. Encouraging to learners while remaining direct and functional.
- **Framing**: Speak to curiosity — "How does this work?" and "What happens if...?"
- Avoid excessive jargon or long analogies. Prioritize clarity.

## Visual Identity

- **Aesthetic**: Clean & modern. High whitespace, legible typography, clear visual hierarchy.
- **Modes**: Light and dark, both with high contrast for data visualizations.
- **Color Palette**: Consistent color language for different model components (e.g., specific colors for attention vs MLP) to aid mental mapping.

## UI Patterns

- **Progressive Disclosure**: Tooltips for brief context, in-situ descriptions paired with interactive examples, glossary/chatbot for depth.
- **Sandbox Explorer**: Comprehensive control panel for free-form exploration (toggles, sliders, ablation switches).
- **Comparison View**: Integrated into the sandbox so users see modification impact relative to original state.
- **On-Demand Depth**: Keep the primary interface simple with clear paths to dive deeper.

## User Experience

The interface centers on exploration and clarity. Users start by selecting a model and inputting text. The dashboard unfolds the model's processing pipeline, letting users zoom into specific components. Experimentation modes are clearly distinguished: hypothesize ("What if I turn off this head?") and test. Educational resources are omnipresent but non-intrusive — available on-demand.
