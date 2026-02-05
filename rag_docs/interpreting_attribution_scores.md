# Interpreting Attribution Scores

## Quick Reference

Token attribution scores tell you how much each input token influenced a specific prediction. Here's how to read and interpret them.

## Understanding the Scores

Attribution scores are **normalized** so that the most influential token gets a score of 1.0, and all other scores are relative to it.

### High Score (0.7 - 1.0)

This token was **highly influential**. The model relied heavily on this token when making its prediction. For factual predictions, these are usually the content words that carry the key information.

**Example**: In "The capital of France is" → "Paris", the token "France" typically gets the highest score because it directly determines which capital the model predicts.

### Medium Score (0.3 - 0.7)

This token had **moderate influence**. It contributed context that helped the prediction but wasn't the primary driver.

**Example**: In the same prompt, "capital" might get a medium score -- it tells the model to predict a city name, but "France" specifies which one.

### Low Score (0.0 - 0.3)

This token had **minimal influence** on this specific prediction. It may be a function word (the, of, is) or a word that doesn't directly relate to what's being predicted.

## Comparing Attribution Methods

### Integrated Gradients vs. Simple Gradient

- **Integrated Gradients** averages gradients over many intermediate steps between a "blank" baseline and the actual input. This produces more reliable, less noisy scores. Use it when you want trustworthy results.
- **Simple Gradient** takes a single gradient measurement. It's faster but can be noisy -- scores may overemphasize some tokens or miss subtle contributions. Good for quick exploration.

**When they disagree**: If the two methods give very different rankings, the attribution is likely noisy. Trust Integrated Gradients for the more accurate picture.

## Why Results Vary by Target Token

Attribution is computed **with respect to a specific target token**. Different targets can be driven by entirely different input tokens.

**Example** with prompt "Alice gave Bob a gift because she":
- Target "liked": High attribution for "Alice" (she liked something) and "Bob" (she liked Bob)
- Target "was": High attribution for "she" and "Alice" (describing Alice's state)
- Target "wanted": High attribution for "gift" (she wanted to give a gift)

This is one of the most powerful uses of attribution -- it reveals which input tokens support different possible continuations.

## Common Patterns

- **Content words dominate**: Nouns, verbs, and adjectives typically have higher attribution than function words
- **Recent tokens often matter more**: Tokens closer to the prediction point tend to have higher attribution, especially for local patterns
- **Distant tokens can matter too**: For long-range dependencies (like pronoun resolution), distant tokens can have surprisingly high attribution
- **Punctuation varies**: Commas and periods sometimes have notable attribution because they signal sentence structure

## Tips

- Try the same prompt with multiple target tokens to see how attribution shifts
- Short prompts (5-10 tokens) give the clearest attribution results
- If all scores are roughly equal, the model may be uncertain or the prediction may not depend on any single token
- Use attribution alongside ablation for a fuller picture: attribution tells you which input tokens matter; ablation tells you which internal components matter
