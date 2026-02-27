# Recommended Starting Points

## Best First Model

**GPT-2 (124M)** is the ideal starting model because:
- It loads quickly and runs fast
- It has a manageable size (12 layers, 12 heads = 144 heads total)
- It's the most studied model in mechanistic interpretability research
- Most educational examples and tutorials reference GPT-2

## Good Starter Prompts

### For Exploring Basic Predictions

| Prompt | What It Tests |
|--------|--------------|
| `The cat sat on the` | Simple object prediction (mat, floor, bed) |
| `The capital of France is` | Factual recall (Paris) |
| `1 + 1 =` | Basic arithmetic |
| `Once upon a time` | Creative story continuation |

### For Exploring Attention Patterns

| Prompt | What It Shows |
|--------|--------------|
| `The cat sat on the mat because it was` | Pronoun resolution: does "it" attend to "cat" or "mat"? |
| `Alice gave the book to Bob because she` | Gendered pronoun resolution |
| `The dogs in the park were` | Subject-verb agreement across a prepositional phrase |
| `I went to the store and bought` | Sequential event prediction |

### For Ablation Experiments

| Prompt | Why It's Good for Ablation |
|--------|---------------------------|
| `The cat sat on the` | Simple enough that ablating one head can change the prediction |
| `The president of the` | Factual prompts show clear ablation effects on knowledge retrieval |
| `She picked up the phone and` | Action continuation is sensitive to Previous-Token head ablation |

### For Attribution Experiments

| Prompt | What Attribution Reveals |
|--------|------------------------|
| `The capital of France is` | "France" should have highest attribution for "Paris" |
| `The doctor told the nurse that she` | Which noun drives the pronoun prediction? |
| `The large red ball rolled down the` | Do adjectives or nouns matter more? |

## Suggested Experiment Order

If you're new to the dashboard, follow this path:

1. **Experiment: Your First Analysis** -- Learn the basics with GPT-2 and a simple prompt
2. **Experiment: Exploring Attention Patterns** -- Understand what attention heads do
3. **Experiment: Your First Ablation** -- Remove a head and see what happens
4. **Experiment: Token Attribution** -- See which input tokens drive predictions
5. **Experiment: Comparing Heads** -- Systematically compare head categories
6. **Experiment: Beam Search** -- Explore alternative generation paths

## After the Basics: Cross-Model Comparisons

Once you've completed the guided experiments, try comparing models to see how architecture affects behavior:

- **GPT-2 vs GPT-Neo 125M**: Same size and PE type, but GPT-Neo alternates local/global attention — see how attention scope matters
- **GPT-2 vs Pythia-160M**: Same size but different positional encoding (absolute vs rotary) — see how RoPE changes attention patterns
- **GPT-2 vs OPT-125M**: Same size but OPT uses ReLU instead of GELU — compare MLP behavior
- **GPT-2 vs GPT-2 Medium**: Same architecture at different scales — see how head specialization changes with more layers
- **Pythia-160M vs Qwen2.5-0.5B**: Both use rotary PE but different normalization (LayerNorm vs RMSNorm) and activation (GELU vs SiLU)
- **Try longer prompts**: See how attention patterns change with more context
- **Combine techniques**: Use attribution to find important tokens, then ablate heads to find the components that process those tokens
