# Experiment: Comparing Heads Across Categories

## Goal

Systematically ablate heads from each category to discover which types of attention heads matter most for different prompts. Build intuition for how attention head roles vary.

## Prerequisites

- Complete "Your First Ablation" (know how to use the ablation panel)
- Complete "Exploring Attention Patterns" (understand head categories)

## Steps

### Step 1: Set Up a Simple Prompt

1. Select **GPT-2 (124M)** and enter: `The cat sat on the`
2. Set beams to 1 and tokens to 5.
3. Click **Analyze**.
4. **Select the generated sequence for comparison** by clicking "Select for Comparison."
5. Note the original prediction and probability in Stage 5 (e.g., "mat" at 45%).

### Step 2: Record the Head Categories

1. Expand **Stage 3 (Attention)** and note one head from each category:
   - **Previous-Token**: _______ (e.g., L0-H3)
   - **First/Positional**: _______ (e.g., L0-H1)
   - **Bag-of-Words**: _______ (e.g., L2-H5)
   - **Syntactic**: _______ (e.g., L4-H2)
   - **Other**: _______ (e.g., L1-H8)

### Step 3: Ablate One Head at a Time

For each head you noted, do the following:
1. Go to the **Ablation** tab in the Investigation Panel.
2. **Clear** any previously selected heads.
3. **Add** just the one head from the current category.
4. Click **"Run Ablation Experiment."**
5. Record the results:

| Category | Head | Probability Change | Generation Changed? |
|----------|------|-------------------|-------------------|
| Previous-Token | | | |
| First/Positional | | | |
| Bag-of-Words | | | |
| Syntactic | | | |
| Other | | | |

### Step 4: Analyze Your Results

Look at the table you've filled in:
- **Which category caused the biggest probability drop?** Previous-Token heads often have the largest impact on simple prompts because local context matters a lot.
- **Which category had the least effect?** BoW and Other heads often show smaller effects for short prompts.
- **Did any ablation change the generated text?** A generation change is a stronger signal than just a probability change.

### Step 5: Try a More Complex Prompt

Now repeat the process with a prompt that requires more sophisticated processing:

1. Enter: `The doctors told the patient that they would need`
2. Analyze and select the generation for comparison.
3. Ablate one head from each category again and record results.

**What to expect**: For this more complex prompt:
- **Syntactic heads** may matter more (there are grammatical dependencies like "doctors...they")
- **First/Positional heads** may show more impact because the sentence structure is more complex
- The pattern of which categories matter may shift compared to the simple prompt

### Step 6: Compare Results Between Prompts

| Category | Simple Prompt Impact | Complex Prompt Impact |
|----------|---------------------|----------------------|
| Previous-Token | | |
| First/Positional | | |
| Bag-of-Words | | |
| Syntactic | | |
| Other | | |

## What You Should Learn

- No single head category is always the "most important" -- it depends on the prompt
- Simple prompts tend to rely more on Previous-Token heads (local patterns)
- Complex prompts with grammatical dependencies may rely more on Syntactic heads
- Some heads are redundant for certain inputs but critical for others
- Ablation is most informative when you compare across conditions (categories, prompts, or both)

## Advanced Challenge

Try ablating **two heads simultaneously** from the same category. Does removing two Previous-Token heads have a bigger effect than removing one? Or does the model have enough redundancy to compensate?
