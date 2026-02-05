# Embeddings Explained

## What Are Embeddings?

After tokenization breaks your text into token IDs, the model needs to convert those IDs into something it can actually compute with. This is where **embeddings** come in. An embedding is a list of numbers (a **vector**) that represents a token's meaning.

## The Lookup Table

Think of embeddings as a giant dictionary. Each token ID maps to a specific vector of numbers. For GPT-2, each token maps to a vector of **768 numbers**. For larger models, this vector might have 2048, 4096, or more numbers.

This dictionary (called an **embedding table**) was learned during training. The model figured out which numbers best represent each token by seeing how tokens are used across billions of text examples. Once training is done, the table is fixed -- the same token always maps to the same vector.

## Why Vectors?

Why use lists of numbers instead of just the token ID? Because vectors let the model capture **meaning** and **relationships**:

- Words with similar meanings (like "happy" and "joyful") end up with similar vectors
- Related concepts are grouped nearby in the vector space
- Directions in the space can capture relationships (e.g., "king" - "man" + "woman" ≈ "queen")

This is what allows the model to generalize -- even if it has never seen a specific sentence before, it can work with the underlying meanings of the tokens.

## Positional Information

There's one more detail: the model also needs to know the **order** of tokens (since "dog bites man" is different from "man bites dog"). This is handled by adding **positional encodings** to the embeddings -- extra numbers that tell the model where each token sits in the sequence.

## What You See in the Dashboard

In **Stage 2 (Embedding)** of the pipeline, you can see:

- The dimension of the embedding vectors (e.g., "768-dimensional" for GPT-2)
- A visual showing the flow: Token ID → Lookup Table → Vector
- An explanation of how the lookup table was created during training

This is the point where raw token IDs become rich numerical representations that the rest of the model can process.
