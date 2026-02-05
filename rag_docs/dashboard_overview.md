# Dashboard Overview

## What Is This Dashboard?

The Transformer Explanation Dashboard is an interactive web application that lets you look inside transformer-based language models as they process text. Instead of treating the model as a "black box," you can see exactly what happens at each step when the model makes a prediction.

## How to Navigate

The dashboard has several main sections, from top to bottom:

### 1. Header and Sidebar
At the top is the dashboard title. On the left edge, there's a collapsible **sidebar** (click the hamburger menu icon). The sidebar contains advanced configuration options for selecting which internal model components to hook into. Most users won't need to change these settings.

### 2. Generator Section
This is where you start. It contains:
- **Model dropdown**: Choose which transformer model to load (e.g., GPT-2, Qwen2.5)
- **Prompt input**: Type the text you want the model to analyze
- **Generation settings**: Control beam search parameters (number of beams and tokens to generate)
- **Analyze button**: Click to run the model and see results

### 3. Generation Results
After clicking Analyze, you'll see one or more **generated sequences** -- these are the model's continuations of your prompt. If you used beam search with multiple beams, you'll see multiple possible continuations. Click "Select for Comparison" on any sequence to use it in ablation experiments.

### 4. Pipeline Visualization
This is the core educational section. It shows **5 expandable stages** that your text passes through:
1. **Tokenization**: How your text is split into tokens
2. **Embedding**: How tokens become number vectors
3. **Attention**: How the model finds relationships between tokens
4. **MLP**: How stored knowledge is retrieved
5. **Output**: What the model predicts and how confident it is

Click any stage to expand it and see detailed explanations and visualizations.

### 5. Investigation Panel
At the bottom, two experiment tabs let you investigate *why* the model made its prediction:
- **Ablation**: Remove specific attention heads and see what changes
- **Token Attribution**: Measure which input tokens influenced the prediction most

### 6. AI Assistant (Chatbot)
The floating robot icon in the bottom-right corner opens the AI chatbot. It can answer questions about transformers, explain what you're seeing in the dashboard, and guide you through experiments.

## Typical Workflow

1. Select a model (start with GPT-2 if you're new)
2. Enter a prompt (e.g., "The cat sat on the")
3. Click "Analyze" to run the model
4. Explore the 5 pipeline stages to understand how the model processed your input
5. Use the Investigation Panel to run ablation or attribution experiments
6. Ask the chatbot if anything is unclear
