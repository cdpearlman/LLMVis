# The Attention Mechanism

## What Is Attention?

Attention is the core innovation that makes Transformers powerful. It allows the model to look at **all tokens in the input simultaneously** and figure out which ones are relevant to each other. When processing any token, the model can "attend to" (focus on) other tokens that provide useful context.

## An Intuitive Analogy

Imagine you're reading a sentence and you come across the word "it." To understand what "it" refers to, you look back at earlier words -- maybe "the cat" or "the ball." Your brain is doing something like attention: scanning the context and focusing on the most relevant parts.

The model does this mathematically by computing **attention scores** between every pair of tokens.

## How It Works: Queries, Keys, and Values

Attention uses three concepts (don't worry about the math -- the intuition is what matters):

- **Query (Q)**: "What am I looking for?" -- each token asks a question
- **Key (K)**: "What do I contain?" -- each token advertises what information it has
- **Value (V)**: "Here's my actual information" -- the content that gets passed along

The model computes how well each Query matches each Key. High matches mean strong attention. The final output for each token is a weighted combination of all the Values, weighted by these attention scores.

## Multi-Head Attention

Instead of computing attention once, Transformers compute it multiple times in parallel using different **attention heads**. Each head learns to look for different kinds of relationships:

- One head might track which word a pronoun refers to
- Another might focus on the immediately preceding token
- Another might spread attention broadly across the sentence

GPT-2 has **12 attention heads per layer** (144 heads total across 12 layers). The dashboard categorizes these heads to help you understand their roles.

## What You See in the Dashboard

In **Stage 3 (Attention)** of the pipeline, you can see:

- **Head Categories**: Heads are grouped by pattern type -- Previous-Token, First/Positional, Bag-of-Words, Syntactic, and Other. Click each category to see which specific heads (like L0-H3) belong to it.
- **BertViz Visualization**: An interactive attention map. Lines connect tokens on the left to the tokens they attend to on the right. Thicker, darker lines mean stronger attention. You can click on specific heads to focus on them.

**Want more technical detail?** Ask the chatbot about Q/K/V dot products, softmax normalization, or how attention scores are computed mathematically.
