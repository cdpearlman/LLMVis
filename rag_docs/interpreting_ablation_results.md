# Interpreting Ablation Results

## Quick Reference

When you ablate an attention head and see the results, here's how to interpret what happened.

## Probability Changes

The dashboard shows the immediate next-token probability before and after ablation (e.g., "72.3% → 45.1% (-27.2%)").

### Large Probability Drop (>10%)

The ablated head was **important** for this prediction. It was actively contributing to the model's confidence in the top token. This head likely plays a significant role in processing this specific input.

**Example**: Ablating a Previous-Token head when the model is predicting a word that commonly follows the previous word (like predicting "the" after "on").

### Small Probability Drop (1-10%)

The head has **some contribution** but isn't critical. Other heads or MLP layers may provide overlapping information. The model has some redundancy that compensates for the missing head.

### Negligible Change (<1%)

The head was likely **redundant for this input**. It may serve a function that isn't relevant to this particular prompt, or other heads provide the same information.

**Important**: This doesn't mean the head is useless -- it might be critical for other prompts. Try the same head with different inputs.

### Probability Increase

Occasionally, ablating a head can **increase** the probability of the top prediction. This means the head was actually pulling the model away from this prediction -- it was a "competing signal." This is an interesting finding that suggests the head was promoting a different output.

## Generation Changes

The full generation comparison shows whether the ablated model produces different text.

### Generation Changed

The head was important enough that removing it altered the model's entire output sequence. This is a strong signal of importance. Look at where the texts diverge -- the point of divergence tells you where the head's contribution was most critical.

### Generation Stayed the Same

Even if the probability shifted, the model still chose the same tokens. This means the head's contribution wasn't large enough to cross the decision boundary. The model is robust to losing this head for this particular input.

## Multi-Head Ablation

When you ablate multiple heads simultaneously:

- **Additive effects**: If ablating heads A and B together has a bigger effect than either alone, the heads contributed independently to the prediction.
- **Redundant heads**: If ablating both has about the same effect as ablating just one, the heads may have been providing the same information.
- **Synergistic effects**: Rarely, ablating two heads together can have a much larger effect than the sum of their individual effects. This suggests the heads work together as a circuit.

## Tips for Interpretation

- Always compare ablation effects across different head categories on the same prompt
- Try the same head on multiple prompts to see if its importance is consistent or input-dependent
- A head's category (Previous-Token, Syntactic, etc.) gives you a hypothesis about why it matters -- ablation lets you test that hypothesis
- Remember that ablation is a blunt tool: removing a head removes all of its functions, not just the one you're interested in
