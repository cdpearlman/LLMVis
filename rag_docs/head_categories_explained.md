# Attention Head Categories Explained

## What Are Head Categories?

The dashboard automatically analyzes all attention heads in the model and categorizes them based on their behavior patterns. This helps you understand what each head is doing without having to inspect every attention map manually.

Head categories appear in **Stage 3 (Attention)** of the pipeline. Click any category to expand it and see which specific heads (like L0-H3, L2-H11) belong to it.

## The Five Categories

### Previous-Token Heads

**What they do**: These heads strongly attend to the **immediately preceding token**. For every token at position *i*, the head focuses most of its attention on position *i-1*.

**Why they matter**: Previous-token heads help the model track local context -- the word that just came before. They're important for bigram patterns (common two-word combinations like "of the" or "in a").

**Detection**: A head is classified as Previous-Token if, on average, more than 40% of each token's attention goes to the token directly before it.

**In the dashboard**: These heads are labeled with a purple color. Ablating them often causes noticeable changes in predictions.

### First/Positional Heads

**What they do**: These heads focus heavily on the **first token** in the sequence or show strong **positional patterns** (always attending to a specific position regardless of content).

**Why they matter**: The first token often serves as a "default" attention target. Positional heads help the model keep track of where it is in the sequence.

**Detection**: Classified when average attention to the first token exceeds 25%.

### Bag-of-Words (BoW) Heads

**What they do**: These heads spread their attention **broadly and evenly** across many tokens, without focusing strongly on any particular one.

**Why they matter**: BoW heads capture a general summary of the entire input. They help the model maintain an overall sense of what the text is about.

**Detection**: Classified when the attention distribution has high entropy (≥ 0.65 normalized) and no single token receives more than 35% attention.

### Syntactic Heads

**What they do**: These heads attend to tokens at **consistent distances**, suggesting they track grammatical or structural relationships (like subject-verb pairs).

**Why they matter**: Syntactic heads help the model understand grammar and sentence structure. They might connect a verb to its subject or a pronoun to what it refers to.

**Detection**: Classified when tokens consistently attend to other tokens at similar distances, with low variance in attention distances.

### Other

**What they do**: Heads that don't clearly fit any of the above patterns. They may have mixed or context-dependent behavior.

**Why they matter**: "Other" doesn't mean unimportant. These heads may serve specialized roles that only activate for certain inputs. They're worth investigating through ablation experiments.

## Using Categories for Experiments

Head categories are especially useful for guiding ablation experiments:
- Ablate a **Previous-Token** head to see if local context patterns break
- Ablate a **BoW** head to see if the model loses global context
- Compare the effect of ablating heads from different categories on the same prompt
