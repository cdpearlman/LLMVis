# MLP (Feed-Forward) Layers Explained

## What Are MLP Layers?

After attention gathers context from other tokens, each token's representation passes through a **Multi-Layer Perceptron (MLP)**, also called a **Feed-Forward Network (FFN)**. While attention handles relationships between tokens, the MLP processes each token independently -- and this is where much of the model's **factual knowledge** is stored.

## What They Do

Think of the MLP as the model's **memory bank**. During training, the MLP weights learned to encode facts, patterns, and associations from the training data. When the model processes "The capital of France is," the MLP layers help recall that "Paris" is the answer.

Researchers have found that specific facts are often stored in specific MLP neurons. This is one of the key findings in mechanistic interpretability research.

## The Expand-Then-Compress Pattern

Each MLP layer follows a distinctive pattern:

1. **Expand**: The token's representation is projected into a much larger space (typically 4x the hidden dimension). For GPT-2, this means going from 768 dimensions to 3072 dimensions.
2. **Activate**: A non-linear activation function is applied (like GELU or SiLU), which allows the network to represent complex patterns.
3. **Compress**: The expanded representation is projected back down to the original size (768 for GPT-2).

**Why expand then compress?** The expansion creates space for many individual neurons to each "vote" on whether a specific concept or fact is relevant. The compression then combines these votes into a refined representation. Each neuron in the expanded layer can activate for specific concepts.

## Attention + MLP = One Layer

In each Transformer layer, attention and MLP work together:

1. **Attention** gathers relevant context from other tokens
2. **MLP** retrieves stored knowledge and transforms the representation
3. The result is added back to the **residual stream** (the running representation)

This happens in every layer. GPT-2 has 12 such layers; each one further refines the model's understanding.

## What You See in the Dashboard

In **Stage 4 (MLP/Feed-Forward)** of the pipeline, you can see:

- The expand-compress flow: Input dimension → Expanded dimension → Output dimension
- The number of layers in the model
- An explanation of why the expansion matters for knowledge storage
