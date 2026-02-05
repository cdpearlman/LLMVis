# The Transformer Architecture

## What Is a Transformer?

A **Transformer** is the specific type of neural network architecture used by modern LLMs like GPT-2, LLaMA, and others. It was introduced in the 2017 paper "Attention Is All You Need" and quickly became the dominant approach for language tasks.

## The Key Innovation: Attention

Before Transformers, language models processed words one at a time, left to right. Transformers changed this by introducing the **attention mechanism**, which lets the model look at all words in the input simultaneously and figure out which ones are relevant to each other.

For example, in "The cat sat on the mat because it was tired," attention helps the model connect "it" back to "the cat."

## How Layers Stack

A Transformer is built from identical **layers** stacked on top of each other. Each layer has two main parts:

1. **Attention**: Looks at relationships between all tokens
2. **MLP (Feed-Forward Network)**: Processes each token's information individually, retrieving stored knowledge

A small model like GPT-2 has 12 layers. Larger models may have 32, 64, or more.

Information flows through these layers sequentially. After each layer, the model's understanding of the text becomes more refined. Early layers tend to capture basic patterns (like grammar), while later layers capture more complex meanings.

## Encoder vs. Decoder

The original Transformer had two halves:

- **Encoder**: Reads and understands the full input (used in models like BERT)
- **Decoder**: Generates text one token at a time (used in GPT-style models)

Most LLMs you'll encounter in this dashboard are **decoder-only** models. This means they generate text left-to-right, predicting one token at a time based on everything that came before it. Each token can only "see" the tokens to its left -- it cannot look ahead.

## The Residual Stream

There is an important concept called the **residual stream** (or "residual connection"). Think of it as a conveyor belt running through all the layers. Each layer reads from this stream, does some processing, and adds its result back. This means information from early layers is preserved and can be used by later layers.

## How This Connects to the Dashboard

The dashboard's 5-stage pipeline follows the exact path data takes through a Transformer: Tokenization, Embedding, Attention, MLP, and Output. When you expand each stage, you're seeing what happens at that point in the architecture.
