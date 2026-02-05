# Troubleshooting and FAQ

## Common Issues

### Model takes a long time to load

**Why**: The first time you load a model, it must be downloaded from HuggingFace. GPT-2 (124M) is about 500MB; larger models are much bigger.

**Fix**: Be patient on the first load. Subsequent loads should be faster because the model is cached locally. If loading is consistently slow, try a smaller model.

### Ablating a head has no effect

**Why**: Not every head is important for every input. Many attention heads are redundant -- the model has learned to distribute work across multiple heads, so removing one doesn't always change the output.

**Fix**: This is actually an interesting finding! Try:
- Ablating a head from a different category (Previous-Token heads often show more effect)
- Using a different prompt (some prompts depend more on specific heads)
- Ablating multiple heads simultaneously to see if their combined removal has an effect

### Attribution takes too long

**Why**: Integrated Gradients is computationally expensive because it runs the model multiple times (typically 50 steps) to build up the attribution scores.

**Fix**: Switch to "Simple Gradient" for faster (though less accurate) results. Or use a shorter prompt -- fewer tokens means faster computation.

### The model's prediction seems wrong or nonsensical

**Why**: Small models like GPT-2 (124M) have limited knowledge and can produce incorrect facts, repetitive text, or non-sequiturs. The model was trained on data from before 2019 and has a limited understanding of the world.

**Fix**: This is expected behavior for small models. The dashboard is designed for exploring *how* the model works, not for getting useful outputs. Try different prompts or a different model.

### BertViz visualization is hard to read

**Why**: With 12+ heads selected simultaneously, the attention lines overlap and become a dense mess.

**Fix**: Double-click on a single head in the BertViz visualization to isolate it. Then explore heads one at a time. Use the head categories to guide which heads to investigate.

### The dashboard becomes slow or unresponsive

**Why**: Larger models require more memory and computation. Running multiple experiments without refreshing can also accumulate memory usage.

**Fix**: Try a smaller model. Refresh the browser page if things get sluggish. Close other memory-intensive applications.

## Frequently Asked Questions

### Which model should I start with?

**GPT-2 (124M)** is the best starting model. It's small, fast, well-studied, and has clean attention patterns that are easy to understand. Move to Qwen2.5-0.5B once you're comfortable for a comparison.

### What prompts work best for learning?

Start with short, simple prompts (5-10 words) that have clear, predictable continuations:
- "The cat sat on the" (predict a location)
- "The capital of France is" (predict a fact)
- "Once upon a time there was a" (predict a story element)

These give clear, interpretable results in the pipeline and experiments.

### Can I use my own model?

Yes! Type any HuggingFace model ID into the model dropdown. The dashboard supports GPT-2, LLaMA, OPT, GPT-NeoX, BLOOM, Falcon, and MPT architectures. Unknown architectures may need manual configuration in the sidebar.

### What's the difference between the pipeline and the investigation panel?

The **pipeline** (5 stages) shows what happens during the model's forward pass -- how your input is processed step by step. The **investigation panel** (ablation + attribution) lets you run experiments to understand *why* the model made a specific prediction.

### How do head categories get determined?

The dashboard automatically analyzes each attention head's pattern using heuristic rules (based on thresholds for attention distributions). For example, a head is classified as "Previous-Token" if more than 40% of each token's attention goes to the immediately preceding token. These categories are computed fresh for each analysis.

### Can I save my results?

Currently, results are displayed in the browser and aren't saved between sessions. You can take screenshots or copy text from the chatbot (using the copy button on messages) to record your findings.
