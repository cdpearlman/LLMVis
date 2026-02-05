# Experiment: Exploring Attention Patterns

## Goal

Learn to read attention visualizations and understand what different attention head categories reveal about how the model processes text.

## Prerequisites

Complete "Your First Analysis" first so you're familiar with the basic workflow.

## Steps

### Step 1: Run an Analysis

1. Select **GPT-2 (124M)** and enter the prompt: `The cat sat on the mat because it was`
2. Click **Analyze**.
3. This prompt is ideal because it contains a pronoun ("it") that needs to be resolved -- the model must figure out that "it" refers to "the cat."

### Step 2: Open the Attention Stage

1. Expand **Stage 3 (Attention)** in the pipeline.
2. Look at the **head categories** section at the top.

### Step 3: Explore Previous-Token Heads

1. Click on **"Previous-Token"** to expand the category.
2. Note which heads are listed (e.g., L0-H3, L1-H7).
3. In the **BertViz visualization** below, double-click on one of those head squares. This will show only that head's attention pattern.
4. **What to look for**: You should see a strong diagonal pattern -- each token attends heavily to the token directly before it. Lines should mostly connect each word to the previous word.

### Step 4: Explore First/Positional Heads

1. Click on **"First/Positional"** to see which heads focus on the first token.
2. Double-click one of those heads in BertViz.
3. **What to look for**: You should see many tokens sending attention lines to "The" (the first token). This is a common pattern -- the first token acts as a "sink" for attention when there's no better target.

### Step 5: Explore Bag-of-Words Heads

1. Find a **"Bag-of-Words"** head in the categories.
2. View it in BertViz.
3. **What to look for**: Attention should be spread broadly and evenly across many tokens. Lines will be thin and numerous rather than thick and focused. This head is gathering a general summary of the whole input.

### Step 6: Look for Interesting Patterns

1. Now single-click to select multiple heads from different categories.
2. Look for heads where the token "it" attends strongly to "cat" -- this would suggest the head is helping resolve the pronoun reference.
3. Try hovering over the word "it" on the left side of BertViz to see which words it attends to most strongly.

### Step 7: Try Different Prompts

Run the analysis again with different prompts and compare:
- `Alice gave the book to Bob because she` (pronoun resolution: does "she" attend to "Alice"?)
- `The dogs in the park were` (subject-verb agreement: does the model connect "dogs" to "were"?)
- `1 2 3 4 5` (number sequence: what patterns emerge with non-natural-language input?)

## What You Should Learn

- Different attention heads serve different purposes
- Previous-Token heads are the easiest to identify visually (strong diagonal pattern)
- The same prompt can reveal different patterns in different heads
- BertViz is a powerful tool for understanding attention, but it takes practice to read fluently
- Not all heads have obvious patterns -- and that's okay. The "Other" category captures complex, context-dependent behavior.
