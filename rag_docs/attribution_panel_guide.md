# Attribution Panel Guide

## What Is Token Attribution?

Token attribution measures **which input tokens had the most influence on the model's prediction**. It answers the question: "Which parts of my input mattered most for this output?"

For example, if your prompt is "The capital of France is" and the model predicts "Paris," attribution will show that "France" and "capital" had the highest influence on that prediction.

## How to Use the Attribution Panel

The attribution panel is in the **Investigation Panel** at the bottom of the dashboard, under the "Token Attribution" tab.

### Step-by-Step

1. **Run an analysis first**: Click "Analyze" on a prompt so the model has made a prediction.
2. **Choose an attribution method**:
   - **Integrated Gradients**: More accurate but slower. Computes attribution by gradually "building up" the input and measuring how each token contributes along the way.
   - **Simple Gradient**: Faster but less precise. Takes a single gradient measurement to estimate token importance.
3. **Choose a target token (optional)**: By default, attribution is computed for the model's top prediction. You can select a different token from the top-5 predictions dropdown to see which input tokens would drive *that* alternative prediction.
4. **Click "Compute Attribution"**.

### Reading the Results

The results have two visualizations:

- **Color-coded token chips**: Your input tokens are displayed as colored boxes. Darker blue = higher influence. Hover over any chip to see the exact attribution score.
- **Horizontal bar chart**: Shows the same information as bars, with attribution scores labeled. Longer bars = more influential tokens.

### What Attribution Scores Mean

- **High score (near 1.0)**: This token was highly influential for the prediction. It strongly pushed the model toward the target token.
- **Low score (near 0.0)**: This token had little influence on this particular prediction.
- **Scores are normalized**: The highest-influence token gets a score of 1.0, and others are scaled relative to it.

### Tips

- Compare attribution for different target tokens. You might find that different input tokens drive different predictions.
- Try Integrated Gradients for the most reliable results, especially when you want to draw conclusions.
- Short prompts give cleaner attribution results since there are fewer tokens to compare.
- Function words (like "the" or "is") often have low attribution; content words (nouns, verbs) tend to have higher attribution.
