# Experiment: Your First Analysis

## Goal

Learn how to run your first analysis and walk through each pipeline stage to understand how a transformer model processes text.

## Prerequisites

None -- this is the starting experiment.

## Steps

### Step 1: Select a Model

1. In the **generator section** at the top, find the "Select Model" dropdown.
2. Choose **"GPT-2 (124M)"** from the list.
3. Wait for the model to load. You'll see a status message indicating the model is ready.

### Step 2: Enter a Prompt

1. In the **"Enter Prompt"** textarea, type: `The cat sat on the`
2. Leave the generation settings at their defaults (1 beam, a few tokens).

### Step 3: Run the Analysis

1. Click the **"Analyze"** button.
2. Wait for the analysis to complete. The pipeline stages and generation results will appear.

### Step 4: Explore the Generated Sequences

Look at the **generated sequence(s)** below the generator. You should see how GPT-2 continues your prompt. Common completions might include "mat," "floor," "bed," or similar words.

### Step 5: Walk Through the Pipeline

Now expand each of the **5 pipeline stages** by clicking on them:

**Stage 1 - Tokenization**: Click to expand. You'll see your prompt split into tokens. Notice how each word (and its leading space) becomes a separate token. Count the tokens -- "The cat sat on the" should produce about 5 tokens.

**Stage 2 - Embedding**: Click to expand. You'll see that each token was converted into a 768-dimensional vector. This is GPT-2's hidden dimension.

**Stage 3 - Attention**: Click to expand. This is the richest stage:
- Look at the **head categories**. You should see heads grouped into Previous-Token, First/Positional, Bag-of-Words, Syntactic, and Other.
- Click on a category (like "Previous-Token") to see which specific heads belong to it.
- Below the categories, you'll see the **BertViz visualization**. Try clicking on individual head squares to see their attention patterns.

**Stage 4 - MLP**: Click to expand. You'll see the expand-compress pattern: 768 → 3072 → 768. This shows GPT-2's feed-forward network dimensions.

**Stage 5 - Output**: Click to expand. You'll see:
- Your prompt with the predicted next token highlighted
- The confidence percentage
- A top-5 bar chart showing the model's top predictions

### Step 6: Reflect

Think about what you observed:
- How many tokens did your prompt become?
- What was the model's top prediction? How confident was it?
- Were there any surprising alternative predictions in the top 5?

## What's Next?

Try changing the prompt and running the analysis again. Compare results with different inputs:
- A factual prompt: "The capital of France is"
- A creative prompt: "Once upon a time, there was a"
- A technical prompt: "The function takes an input and"

Then move on to **Experiment: Exploring Attention Patterns** to dive deeper into what the attention heads are doing.
