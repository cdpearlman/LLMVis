# Interpreting Attention Maps

## Quick Reference

The BertViz attention visualization in Stage 3 shows how tokens attend to each other. Here's how to read the patterns.

## Reading the Visualization

The BertViz display shows:
- **Left column**: Each token in your input (the "query" -- the token doing the looking)
- **Right column**: The same tokens (the "keys" -- what's being looked at)
- **Lines between them**: Attention connections. Each line shows how much one token attends to another.

### Line Properties
- **Thicker, more opaque lines** = stronger attention (the model focuses more on this connection)
- **Thin, faint lines** = weak attention (some attention, but not much)
- **No line** = very little or no attention between those tokens

### Interacting with BertViz
- **Single-click** a head square at the top to select/deselect it
- **Double-click** a head square to view only that head (deselects all others)
- **Hover** over a token or line to see exact attention weights

## Common Attention Patterns

### Diagonal Pattern (Previous-Token)

You see each token strongly attending to the token directly before it, creating a diagonal line of strong connections.

**What it means**: This head tracks local word order. It's useful for bigram patterns -- sequences of two words that commonly appear together.

**Looks like**: A staircase pattern of thick lines, each shifted one position to the left.

### Vertical Stripe (First-Token / Positional)

You see many or all tokens attending to the same position (usually the first token), creating a vertical column of lines.

**What it means**: The first token often serves as a "default sink" for excess attention. This is an artifact of the softmax function -- attention weights must sum to 1.0, so when a head has nothing specific to attend to, it sends attention to the first token.

**Looks like**: Many thick lines all pointing to the same token on the right side.

### Uniform / Diffuse (Bag-of-Words)

You see many thin lines spreading from each token to many other tokens, with no strong focus on any particular one.

**What it means**: This head is gathering a broad summary of the entire input, rather than focusing on specific relationships. It helps the model maintain an overall sense of context.

**Looks like**: A dense web of thin, similarly-weighted lines.

### Structured Connections (Syntactic)

You see specific, purposeful-looking connections that skip across tokens -- like a token attending to a word several positions away in a consistent pattern.

**What it means**: This head may be tracking grammatical relationships. For example, a verb attending to its subject, or a pronoun attending to its antecedent.

**Looks like**: A few thick lines making specific connections, often spanning several token positions.

### Mixed / Context-Dependent

Some heads show patterns that change based on the input. They might show one pattern for factual prompts and another for creative prompts.

**What it means**: These heads are flexible and context-sensitive. They don't have a single fixed pattern but adapt to the input.

## Tips for Reading Attention Maps

- **Start with one head at a time**: Double-click individual heads to isolate their patterns. Looking at all heads simultaneously is confusing.
- **Compare heads across layers**: The same type of pattern (like Previous-Token) may appear in early layers but not late layers, or vice versa.
- **Match patterns to categories**: Use the head categories as a guide. If a head is categorized as "Previous-Token," look for the diagonal pattern to confirm.
- **Hover for details**: The exact attention weight tells you more than just the visual thickness. Two lines might look similar but have meaningfully different weights.
- **Context matters**: The same head can show different patterns for different prompts. Try multiple inputs to understand a head's full behavior.
