# Experiment: Understanding Token Attribution

## Goal

Learn how to use token attribution to identify which parts of your input most influenced the model's prediction. Compare two attribution methods and see how results change with different target tokens.

## Prerequisites

- Complete "Your First Analysis"

## Steps

### Step 1: Run an Analysis with a Meaningful Prompt

1. Select **GPT-2 (124M)** and enter the prompt: `The capital of France is`
2. Click **Analyze**.
3. Check Stage 5 -- the model should predict something like "Paris" or "the" with high confidence. Note the top prediction.

### Step 2: Open the Attribution Panel

1. Scroll down to the **Investigation Panel**.
2. Click the **"Token Attribution"** tab.

### Step 3: Run Simple Gradient Attribution

1. Select **"Simple Gradient (faster, less accurate)"** as the attribution method.
2. Leave the **Target Token** dropdown empty (this defaults to the top prediction).
3. Click **"Compute Attribution"**.

### Step 4: Read the Results

Look at the two visualizations:

**Color-coded tokens**: Your input tokens are displayed as colored boxes. Darker blue means higher influence.
- You should see that **"France"** has a very dark color -- it's the most relevant token for predicting "Paris."
- **"capital"** likely also has a notable color -- it sets up the context for a city name.
- Function words like **"The"**, **"of"**, and **"is"** should be lighter -- they contribute less to this specific prediction.

**Bar chart**: Shows the same information as horizontal bars with scores. Longer bars = more influence.

**Hover over any token chip** to see the exact attribution score.

### Step 5: Compare with Integrated Gradients

1. Now switch to **"Integrated Gradients (more accurate, slower)"**.
2. Click **"Compute Attribution"** again.
3. Compare the results. Integrated Gradients should give a more refined picture:
   - The relative ordering of token importance may shift slightly
   - Integrated Gradients tends to produce more reliable scores, especially for distinguishing tokens of moderate importance

### Step 6: Change the Target Token

1. In the **Target Token** dropdown, select a different token from the top-5 predictions (e.g., if the model also considered "a" or "the" as alternatives to "Paris").
2. Run attribution again.
3. **What to look for**: Different target tokens are driven by different input tokens. For example:
   - "Paris" might be strongly driven by "France" and "capital"
   - A generic token like "the" might be driven more by "is" (as a common grammatical continuation)

### Step 7: Try a Different Prompt

Run attribution on: `Alice gave the book to Bob because she`
- Which tokens drive the prediction of the next word?
- Does "Alice" have high attribution (suggesting the model connects "she" to "Alice")?
- Does "Bob" have lower attribution than "Alice" for this prediction?

## What You Should Learn

- Token attribution reveals which input tokens "caused" a particular prediction
- Content words (nouns, verbs) typically have higher attribution than function words (the, of, is)
- Different target tokens can be driven by completely different input tokens
- Integrated Gradients is more accurate but slower; Simple Gradient gives a quick approximation
- Attribution helps you understand the "why" behind a model's prediction

## What's Next?

Move on to **Experiment: Comparing Heads** to combine ablation with your understanding of head categories, or try **Experiment: Beam Search** to explore how the model generates longer sequences.
