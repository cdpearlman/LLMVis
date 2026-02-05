# Ablation Panel Guide

## What Is Ablation?

Ablation is an experiment where you **remove (disable) specific attention heads** from the model and observe how the output changes. If removing a head causes the prediction to change significantly, that head was important for this particular input. If the prediction barely changes, the head was either redundant or not relevant to this context.

The term comes from neuroscience, where "ablation" means removing part of the brain to study its function.

## How to Use the Ablation Panel

The ablation panel is found in the **Investigation Panel** at the bottom of the dashboard, under the "Ablation" tab.

### Step-by-Step

1. **Run an analysis first**: You need to have clicked "Analyze" on a prompt before ablation is available.
2. **Select a generation for comparison**: Click "Select for Comparison" on one of the generated sequences. This gives the ablation experiment a full generation to compare against.
3. **Choose a head to ablate**: Use the **Layer** and **Head** dropdowns to pick a specific attention head (e.g., Layer 0, Head 3). Click the **+** button to add it.
4. **Add more heads (optional)**: You can add multiple heads from different layers. Each appears as a chip (e.g., "L0-H3") with an × button to remove it.
5. **Run the experiment**: Click "Run Ablation Experiment."
6. **View results**: The panel shows a side-by-side comparison of the original vs. ablated generation, plus the probability change for the immediate next token.

### Understanding the Results

- **Full Generation Comparison**: Shows the original generated text alongside what the model generates with those heads removed. If the text changed, the ablated heads were important for that generation.
- **Probability Change**: Shows the immediate next-token probability before and after ablation (e.g., "72.3% → 45.1% (-27.2%)"). A large drop means the head was important.
- **No change?** If ablating a head has no effect, it may be redundant, or it may serve a function that isn't relevant to this specific prompt. Try a different prompt or a different head.

### Tips

- Start by ablating heads from the **Previous-Token** category -- these often have noticeable effects.
- Try ablating heads from the **Other** category for comparison -- these often have less impact.
- Use the **head categories** in the Attention stage (Stage 3) to pick interesting heads to ablate.
- You can ablate heads from multiple layers simultaneously to see compound effects.
