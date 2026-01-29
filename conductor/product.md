# Product Definition

## Initial Concept
A tool for capturing activations from transformer models and visualizing attention patterns using bertviz and an interactive Dash web application.

## Vision
To demystify the inner workings of Transformer-based Large Language Models (LLMs) for students and curious individuals. By combining interactive visualizations with hands-on experimentation capabilities, the tool transforms abstract architectural concepts into tangible, observable phenomena, fostering a deep, intuitive understanding of how these powerful models process information.

## Target Audience
- **Primary:** Machine Learning Students and AI enthusiasts.
- **Secondary:** Any individual seeking a practical, interactive way to learn about Transformer architectures and mechanical interpretability.

## Core Value Proposition
- **Visual Learning:** Translates complex matrix operations and data flows into clear, interactive visual representations (Attention Maps, Logit Lens).
- **Interactive Experimentation:** Goes beyond static observation by allowing users to manipulate the model (Ablation, Activation Patching) and immediately see the consequences.
- **Educational Scaffolding:** Supports users of varying expertise levels with layered educational content, from simple tooltips to deep-dive glossaries and future AI-guided tutorials.

## Key Features
- **Sequential Data Flow Visualization:** Illustrates how data transforms step-by-step through the model's layers.
- **Component Breakdown:** Detailed inspection views for key components like Self-Attention (heads, weights) and MLPs.
- **Interactive Experiments:**
    - **Ablation Studies:** selectively disable heads or layers to observe impact on output.
    - **Activation Steering:** modify activation values in real-time.
    - **Prompt Comparison:** compare internal activations resulting from two different input prompts side-by-side.
- **Integrated Education:**
    - Contextual tooltips for immediate clarity.
    - Dedicated "Glossary" panel for in-depth definitions.
    - Foundation for AI-guided tutorials.

## User Experience
The interface centers on exploration and clarity. Users start by selecting a model and inputting text. The dashboard then unfolds the model's processing pipeline, allowing users to "zoom in" on specific components. Experimentation modes are clearly distinguished, enabling users to hypothesize ("What if I turn off this head?") and test. Educational resources are omnipresent but non-intrusive, available on-demand to explain the *what* and *why* of what is being visualized.
