# Experiment: Exploring Alternative Predictions with Beam Search

## Goal

Learn how beam search reveals multiple possible continuations of a prompt, and see how ablating attention heads can redirect the model's generation from one path to another.

## Prerequisites

- Complete "Your First Analysis"
- Complete "Your First Ablation"

## Steps

### Step 1: Generate Multiple Beams

1. Select **GPT-2 (124M)** and enter the prompt: `Once upon a time there was a`
2. Set **Number of Generation Choices (Beams)** to **3**.
3. Set **Number of New Tokens** to **8**.
4. Click **Analyze**.

### Step 2: Compare the Beams

You should see 3 different generated sequences. Look at how they differ:
- **Beam 1**: The model's top-ranked overall sequence
- **Beam 2**: The second-best sequence
- **Beam 3**: The third-best sequence

Notice how they might start the same but diverge at some point. For example:
- Beam 1: "Once upon a time there was a young man who lived"
- Beam 2: "Once upon a time there was a little girl who loved"
- Beam 3: "Once upon a time there was a king who ruled"

The beams share a common prefix because the early tokens were confident, but as generation continues, different paths emerge.

### Step 3: Select a Beam for Comparison

1. Click **"Select for Comparison"** on Beam 1 (the top-ranked sequence).
2. This stores it as the baseline for ablation comparison.

### Step 4: Investigate What Drives the Divergence

1. Look at **Stage 5 (Output)** to see the top-5 predictions for the immediate next token.
2. Note the top prediction and its probability. Are the alternatives close in probability? If so, the model was uncertain, which explains why beams diverge early.

### Step 5: Ablate a Head and Re-Generate

1. Go to the **Ablation** tab in the Investigation Panel.
2. From the head categories in Stage 3, pick a **Previous-Token** head (e.g., L0-H3).
3. Add it and click **"Run Ablation Experiment."**
4. Look at the **Full Generation Comparison**:
   - Did the ablated generation diverge from the original?
   - Did the model take a completely different path, or just change a word or two?
   - Did the ablated generation match one of the other beams you saw earlier?

### Step 6: Try a Stronger Ablation

1. **Clear** the selected heads.
2. Add **two or three** Previous-Token heads from different layers.
3. Run the ablation again.
4. Compare: does ablating multiple heads cause a bigger divergence than ablating one?

### Step 7: Experiment with Different Beam Settings

1. Change the prompt to: `The scientist discovered that the`
2. Try with **1 beam** (greedy decoding): note the single output.
3. Try with **3 beams**: see the alternatives.
4. Try with **5 beams**: do you get even more diverse options?

Notice how increasing beams reveals the model's uncertainty -- places where multiple continuations are roughly equally likely.

## What You Should Learn

- **Beam search reveals model uncertainty**: When the model isn't sure, multiple beams show the different paths it's considering.
- **Ablation can redirect generation**: Removing important heads can push the model from one beam to another, showing that different attention heads support different generation paths.
- **More beams = more alternatives**: But beyond 3-5 beams, the additional paths are often low-probability and less interesting.
- **Generation is a chain**: Each token depends on the previous ones, so a small change early (from ablation or beam selection) can cascade into a very different output.

## What's Next?

You've now completed the core experiments. Try combining techniques:
- Run attribution to find which input tokens matter, then ablate the heads that seem to process those tokens
- Compare how GPT-2 and Qwen2.5-0.5B handle the same prompt with the same beam settings
