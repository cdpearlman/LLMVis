# Beam Search and Generation

## What Is Text Generation?

When you click "Analyze" in the dashboard, the model generates a continuation of your prompt. The simplest approach is **greedy decoding**: at each step, pick the single most likely next token. This is fast but can lead to repetitive or suboptimal text.

## What Is Beam Search?

Beam search is a smarter generation strategy that explores **multiple possible paths simultaneously** instead of committing to the single best token at each step.

### How It Works

Imagine you're at a fork in the road. Greedy decoding always takes the path that looks best right now. Beam search keeps multiple paths open and evaluates them over several steps before deciding which one is actually best overall.

More specifically:
1. At each step, the model considers the top candidates for the next token
2. It keeps the **N best partial sequences** (where N is the number of beams)
3. It extends each of those sequences by one more token
4. It ranks all the possibilities and keeps the top N again
5. This continues until the desired number of tokens have been generated

The result is that beam search can find better overall sequences, even if the individual tokens at each step aren't always the highest-probability choice.

### Example

With the prompt "The cat sat on the" and 3 beams:
- Beam 1 might generate: "The cat sat on the mat and purred"
- Beam 2 might generate: "The cat sat on the floor and looked"
- Beam 3 might generate: "The cat sat on the couch and slept"

Each beam represents a different possible continuation.

## Generation Controls in the Dashboard

In the generator section, you have two settings:

- **Number of Generation Choices (Beams)**: How many parallel paths to explore (1-5). With 1 beam, you get greedy decoding. More beams = more diverse results but slower.
- **Number of New Tokens**: How many tokens to generate beyond your prompt (1-20). Longer generations take more time.

## How Generated Sequences Are Used

After generation, the dashboard displays the resulting sequences. You can:
1. **Read them**: See how the model would continue your prompt
2. **Select one for comparison**: Click "Select for Comparison" to use that sequence as a baseline in ablation experiments. This lets you see how removing attention heads changes the full generated sequence, not just the immediate next token.

## Greedy vs. Beam Search

| Feature | Greedy (1 beam) | Beam Search (2+ beams) |
|---------|-----------------|------------------------|
| Speed | Fast | Slower |
| Quality | May miss better paths | Finds better overall sequences |
| Diversity | Only one output | Multiple alternatives |
| Best for | Quick exploration | Thorough analysis |
