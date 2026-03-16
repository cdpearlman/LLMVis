# RESEARCH-jargon-reduction

## Objective
Analyze the LLMVis project for technical jargon (e.g., "head", "ablation", "tokenizing") that may confuse younger or less technical audiences, and outline plain-English alternatives and explanations. The goal is to make the educational dashboard more accessible without completely losing technical accuracy.

## Findings: Jargon in User-Facing Components

### 1. "Ablation" / "Ablate"
**Locations found:**
- `app.py`: "Ablation" tab in the Investigation Panel
- `components/ablation_panel.py`: "What is Ablation?", "Add Head to Ablation List", "Run Ablation Experiment", "Ablated Output"

**Issue:** "Ablation" is a highly specialized term originally from neuroscience/medicine, adapted for machine learning to mean removing a component to observe its effect. It is completely opaque to a layperson.

**Proposed Replacements:**
- Replace primary "Ablation" headers with "Component Removal" or "Test by Removing".
- Replace "Run Ablation Experiment" with "Test What Happens When Removed".
- Keep "Ablation" only in an advanced glossary, a tooltip or as a secondary subtitle: "Test by Removing (Ablation)".

### 2. "Token" / "Tokenization" / "Tokenizing"
**Locations found:**
- `app.py`: "Number of New Tokens" 
- `components/pipeline.py`: "Tokenization" pipeline stage, "Your text is split into tokens"
- `components/investigation_panel.py`: "Token Attribution", "Target Token"

**Issue:** While "token" is explained somewhat in the app ("pieces that the model can understand"), phrases like "Token Attribution" or "Tokenization" sound overly academic as top-level headings.

**Proposed Replacements:**
- Use "Text Chunking" or "Word Pieces" as the primary heading, introducing "Tokens" as the technical equivalent inside the explanation.
- Change "Token Attribution" to "Which words influenced the answer?" or "Word Influence Score".

### 3. "Attention Head" / "Head"
**Locations found:**
- `components/pipeline.py`: "Attention has multiple heads"
- `components/ablation_panel.py`: "Select heads to ablate", "Attention Head Roles"

**Issue:** "Head" in "Attention Head" is an architectural metaphor that doesn't organically mean anything to someone unfamiliar with the transformer architecture (multi-head attention).

**Proposed Replacements:**
- Reframe as "Attention Pattern", "Context Detector" or "Feature Spotter".
- Explain "Heads" as "Specialized Detectors" or "Information Filters" (e.g., "The model uses multiple 'Detectors' (called heads) to find relationships...").

### 4. "Embedding" / "Vector"
**Locations found:**
- `components/pipeline.py`: "Embedding" stage, "high-dimensional vector", "lookup table"

**Issue:** "Vector" and "high-dimensional" sound like advanced mathematical phrases which may intimidate a younger user.

**Proposed Replacements:**
- Instead of "Embedding", use "Meaning Translation" or "Number Dictionary".
- Instead of "high-dimensional vector", use "list of numbers representing meaning".

### 5. "MLP" / "Feed-Forward"
**Locations found:**
- `components/pipeline.py`: "MLP (Feed-Forward)", "MLP layers"

**Issue:** MLP (Multi-Layer Perceptron) is legacy neural network jargon that is not intuitive in the context of what it actually does (storing factual knowledge).

**Proposed Replacements:**
- Rename the pipeline stage to "Factual Knowledge" or "Memory Layers".
- Include "(MLP/Feed-Forward)" only as a small subtitle for those who want the exact technical term.

### 6. "Logit" / "Probability" / "Activation Score"
**Locations found:**
- `components/ablation_panel.py`: "Average prob shift", "activation: 85%"

**Issue:** Jargon related to raw model outputs. "Activation score" sounds like a futuristic/sci-fi term.

**Proposed Replacements:**
- "Activation score" -> "Confidence level" or "Strength of focus".
- "Average prob shift" -> "Average change in confidence".

## What Was Ruled Out
- **Completely eliminating all technical terms:** We ruled out removing words like "Token" or "Model" entirely because learning some valid terminology is part of the educational value of the tool. Instead, we will prioritize plain English for headings and actions, and introduce the technical terms in parentheses or explanatory sentences.
- **Rewriting the RAG Docs right now:** The `rag_docs/` folder contains extensive markdown documentation. For this primary pass, we ruled out modifying those long-form files, focusing entirely on the primary UI headers, buttons, and short stage descriptions in `app.py` and the `components/` directory to maximize immediate impact.

## Next Steps
Review this document. If these jargon replacements aligned with the intended goal for younger audiences, we can draft `SPEC-jargon-reduction.md` with explicit strings to be updated across the application.