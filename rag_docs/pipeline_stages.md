# Pipeline Stages

## Overview

The pipeline visualization shows the 5 stages your text passes through inside the transformer model. A flow indicator at the top shows the path: **Input → Tokens → Embed → Attention → MLP → Output**. Click any stage to expand it and see details.

## Stage 1: Tokenization

**Icon**: Puzzle piece | **Summary shows**: "X tokens"

This stage displays how your input text was split into tokens. Each row shows:
- The **token** (the text piece, displayed in a blue box)
- An arrow pointing to its **ID** (the number the model uses internally, in a purple box)

Notice that spaces are often attached to the beginning of the following word (e.g., " cat" with a leading space). This is normal for models like GPT-2 that use BPE tokenization.

## Stage 2: Embedding

**Icon**: Cube | **Summary shows**: "X-dim vectors"

This stage shows how token IDs are converted into numerical vectors using a pre-learned embedding table. You'll see:
- A visual flow: Token ID → Lookup Table → Vector
- The embedding dimension (e.g., 768 for GPT-2)
- An explanation of how the lookup table was learned during training

## Stage 3: Attention

**Icon**: Eye | **Summary shows**: "X heads × Y layers"

This is the most detailed stage. It includes:
- **Head Categories**: Attention heads are automatically categorized by their behavior pattern (Previous-Token, First/Positional, Bag-of-Words, Syntactic, Other). Click each category to see which specific heads belong to it.
- **BertViz Visualization**: An interactive attention map showing which tokens attend to which. Lines connect tokens on the left to tokens on the right. Thicker lines mean stronger attention.

**Navigating BertViz**: Single-click a head square to select/deselect it. Double-click to show only that head. Hover over tokens or lines to see exact attention weights.

## Stage 4: MLP (Feed-Forward)

**Icon**: Network | **Summary shows**: "X layers"

This stage shows the expand-then-compress pattern of the feed-forward network:
- Input dimension (e.g., 768) → Expanded dimension (e.g., 3072, which is 4x larger) → Back to input dimension
- An explanation of why this expansion matters for storing factual knowledge
- The total number of layers in the model

## Stage 5: Output Selection

**Icon**: Bullseye | **Summary shows**: "→ [predicted token]"

This stage reveals the model's prediction:
- Your full prompt with the **predicted next token** highlighted
- A **confidence percentage** for the top prediction
- A **top-5 bar chart** showing the five most likely next tokens and their probabilities
- A note about how beam search and other techniques can influence the final output
