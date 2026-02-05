# Key Terminology

An extended glossary of terms you may encounter while using the Transformer Explanation Dashboard.

## Core Concepts

**Token**: A small piece of text that the model processes. Can be a word, part of a word, or a punctuation mark. The model's fundamental unit of input and output.

**Embedding**: A vector (list of numbers) that represents a token's meaning. Similar tokens have similar embeddings.

**Attention**: The mechanism that lets each token look at all other tokens to gather relevant context. Uses Queries, Keys, and Values.

**Attention Head**: One instance of the attention mechanism. Each layer has multiple heads that look for different patterns simultaneously.

**Layer**: One complete processing step in the Transformer, containing both attention and MLP components. GPT-2 has 12 layers; larger models have more.

**MLP / Feed-Forward Network (FFN)**: The component in each layer that processes tokens individually, storing and retrieving factual knowledge. Uses an expand-then-compress pattern.

## Architecture Terms

**Residual Stream**: The "conveyor belt" of information running through all layers. Each layer reads from it and adds back its contribution. This preserves information from earlier layers.

**Layer Normalization (LayerNorm)**: A technique applied before or after each sublayer that stabilizes the numbers, keeping them in a reasonable range. This helps training and makes the model more robust.

**Parameters / Weights**: The learnable numbers in the model. These are adjusted during training to improve predictions. GPT-2 has ~124 million parameters.

**Hidden Dimension**: The size of the internal vector representations. For GPT-2, this is 768 -- meaning each token is represented by 768 numbers at each layer.

**Vocabulary**: The complete set of tokens the model knows. GPT-2 has a vocabulary of about 50,257 tokens.

## Training and Inference

**Training**: The process of adjusting the model's parameters by showing it billions of text examples. The model learns to predict the next token and its parameters are updated to reduce prediction errors.

**Inference**: Using the trained model to make predictions on new text. This is what happens when you click "Analyze" in the dashboard -- no learning occurs, the model just processes your input.

**Forward Pass**: One complete trip of data through the model, from input tokens to output predictions. The dashboard visualizes this forward pass.

**Gradient**: A measure of how much each parameter contributed to the model's prediction error. Used during training to update parameters, and in attribution experiments to measure token importance.

**Loss**: A number measuring how wrong the model's predictions are. During training, the goal is to minimize this. Lower loss means better predictions.

**Fine-tuning**: Taking a pre-trained model and training it further on a specific dataset to specialize its behavior.

## Prediction Terms

**Logits**: The raw, unnormalized scores the model assigns to every possible next token before converting to probabilities.

**Softmax**: The function that converts logits into probabilities (positive numbers that sum to 1.0).

**Probability Distribution**: The complete set of probabilities over all possible next tokens. The dashboard shows the top 5.

**Temperature**: A setting that controls prediction confidence. Low temperature = more focused; high temperature = more spread out.

**Beam Search**: A generation strategy that explores multiple possible sequences simultaneously instead of just picking the single best token at each step.
