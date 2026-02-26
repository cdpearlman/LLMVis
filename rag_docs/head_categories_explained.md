# Attention Head Categories

This document explains the different types of attention heads found in transformer models. These categories are determined through **offline analysis** using TransformerLens and **verified at runtime** against your actual input.

## Categories

### Previous Token
**Symbol:** ● (active on most inputs)

Attends to the immediately preceding token — like reading left to right. This head helps the model track local word-by-word patterns. It's one of the most common and reliable head types.

**What to look for in the visualization:** Strong diagonal line one position below the main diagonal.

### Induction
**Symbol:** ● when repeated tokens exist, ○ otherwise

Completes repeated patterns: if the model saw [A][B] before and now sees [A], it predicts [B] will follow. This is one of the most important mechanisms in transformer language models.

**Requires:** Repeated tokens in your input. If no tokens repeat, this category appears grayed out.

**Try this prompt:** "The cat sat on the mat. The cat" — the repeated "The cat" activates induction heads.

### Duplicate Token
**Symbol:** ● when duplicate tokens exist, ○ otherwise

Notices when the same word appears more than once, acting like a highlighter for repeated words. Helps the model track which words have already been said.

**Requires:** Repeated tokens in your input.

**Try this prompt:** "The cat sat. The cat slept." — the repeated words activate duplicate-token heads.

### Positional / First-Token
**Symbol:** ● (active on most inputs)

Always pays attention to the very first word, using it as a fixed anchor point. The first token often serves as a "default" position when no specific token is relevant.

**What to look for:** Strong vertical line at column 0 (all tokens attending to position 0).

### Diffuse / Spread
**Symbol:** ● (active on most inputs)

Spreads attention evenly across many words, gathering general context rather than focusing on one spot. Provides a "big picture" summary of the input.

**What to look for:** No strong patterns — attention is spread roughly evenly across all tokens.

### Other / Unclassified

Heads whose dominant pattern doesn't fit the categories above. These may perform more complex or context-dependent operations.

## How It Works

1. **Offline Analysis:** A TransformerLens script analyzes each head across many test inputs and assigns categories based on dominant behavior patterns.
2. **Runtime Verification:** When you enter a prompt, the app checks whether each head's known role is actually active on your specific input.
3. **Active vs Inactive:** A filled circle (●) means the head's role is triggered. An open circle (○) means the role exists but isn't triggered on your current input (e.g., no repeated tokens for induction).

## Important Note

These categories are simplified labels based on each head's dominant behavior pattern. In reality, attention heads can serve multiple roles and may behave differently depending on the input.
