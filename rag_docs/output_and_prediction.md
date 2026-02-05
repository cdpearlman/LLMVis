# Output and Prediction

## How Does the Model Choose the Next Token?

After your text has passed through all the Transformer layers (attention + MLP in each), the model needs to make a prediction: what token comes next? This final step converts the model's internal representation into a probability distribution over its entire vocabulary.

## Logits: Raw Scores

The model's final hidden state for the last token is multiplied by the embedding table (in reverse) to produce a score for **every token in the vocabulary**. These raw scores are called **logits**. A higher logit means the model thinks that token is more likely.

For GPT-2, this means producing about 50,257 scores -- one for each token in its vocabulary.

## Softmax: Turning Scores into Probabilities

Raw logits can be any number (positive or negative). To get actual probabilities, the model applies a function called **softmax**, which:

1. Converts all scores to positive numbers
2. Makes them all add up to 1.0 (100%)
3. Preserves the ranking (higher logits → higher probabilities)

After softmax, we can say things like "the model predicts 'mat' with 45% probability."

## Temperature

**Temperature** is a setting that controls how "confident" or "creative" the model's predictions are:

- **Low temperature (e.g., 0.1)**: Makes the model very confident -- the top prediction gets almost all the probability. Good for factual, predictable text.
- **High temperature (e.g., 1.5)**: Spreads probability more evenly, making less likely tokens more probable. Good for creative, varied text.
- **Temperature = 1.0**: The default, unmodified distribution.

## Greedy Decoding vs. Sampling

Once we have probabilities, how do we pick the actual next token?

- **Greedy decoding**: Always pick the token with the highest probability. Simple but can be repetitive.
- **Sampling**: Randomly pick a token weighted by the probabilities. More varied but less predictable.
- **Beam search**: Explore multiple possible sequences simultaneously and pick the best overall path. This is available in the dashboard's generation controls.

## What You See in the Dashboard

In **Stage 5 (Output Selection)** of the pipeline:

- The **predicted token** is highlighted after your prompt text, along with its confidence percentage
- A **top-5 bar chart** shows the five most likely next tokens and their probabilities
- A note explains how beam search and other techniques can influence the final selection beyond just the top-1 token
