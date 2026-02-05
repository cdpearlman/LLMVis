# Experiment: Your First Ablation

## Goal

Learn how ablation works by removing an attention head and observing how it changes the model's prediction. Discover which heads matter and which are redundant.

## Prerequisites

- Complete "Your First Analysis"
- Complete "Exploring Attention Patterns" (so you know about head categories)

## Steps

### Step 1: Set Up the Analysis

1. Select **GPT-2 (124M)** and enter the prompt: `The cat sat on the`
2. Set **Number of Generation Choices** to 1 and **Number of New Tokens** to 5.
3. Click **Analyze**.
4. Note the model's prediction in Stage 5 and the generated sequence.

### Step 2: Select a Sequence for Comparison

1. In the generated sequences section, click **"Select for Comparison"** on the generated sequence.
2. This stores the original generation so the ablation experiment can compare against it.

### Step 3: Find a Head to Ablate

1. Expand **Stage 3 (Attention)** and look at the head categories.
2. Find a head from the **Previous-Token** category. Let's say it's **L0-H3** (yours may differ).
3. Note this head -- Previous-Token heads often have noticeable effects when removed.

### Step 4: Set Up the Ablation

1. Scroll down to the **Investigation Panel** and make sure the **"Ablation"** tab is selected.
2. In the **Layer** dropdown, select the layer of your chosen head (e.g., 0).
3. In the **Head** dropdown, select the head number (e.g., 3).
4. Click the **+** button to add it. You should see a chip appear: "L0-H3".

### Step 5: Run the Ablation

1. Click **"Run Ablation Experiment"**.
2. Wait for results to appear.

### Step 6: Analyze the Results

Look at the ablation results:

- **Full Generation Comparison**: Compare the original text to the ablated text. Did the generated sequence change?
- **Probability Change**: Look at the immediate next-token probability change. For example, "72.3% → 45.1% (-27.2%)" would mean removing this head significantly reduced the model's confidence.

### Step 7: Try Ablating a Different Head

1. Click **"Clear Selected Heads"** to reset.
2. Now pick a head from the **"Other"** category (these often have less obvious roles).
3. Add it and run the ablation again.
4. Compare: was the effect larger or smaller than the Previous-Token head?

### Step 8: Compare Your Results

| Head | Category | Probability Change | Generation Changed? |
|------|----------|-------------------|-------------------|
| L0-H3 | Previous-Token | (fill in) | (yes/no) |
| L?-H? | Other | (fill in) | (yes/no) |

**Typical findings**:
- Previous-Token heads in early layers often cause noticeable probability drops when ablated
- Many "Other" heads have minimal impact for simple prompts
- The same head may matter more or less depending on the specific prompt

## What You Should Learn

- Ablation is a tool for measuring the importance of individual model components
- Not all heads are equally important -- some are redundant
- The effect of ablation depends on the specific input prompt
- This technique is used by researchers to understand how models work internally

## What's Next?

Move on to **Experiment: Token Attribution** to learn a different approach -- instead of removing components, measure which input tokens drive the prediction.
