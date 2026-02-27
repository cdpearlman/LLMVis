# LLMVis UX & Explanation Review
**Date:** 2026-02-26  
**Reviewer:** JARVIS  
**Method:** Playwright automated walkthrough of https://cdpearlman-llmvis.hf.space (GPT-2 124M, prompt: "The cat sat on the mat. The cat")  
**Reference:** `attention_handoff.md` (attention head categorization spec)

---

## Executive Summary

The app is in solid working shape. The pipeline storytelling is clean, the BertViz integration works, and attribution renders well. The two biggest gaps against the handoff spec are: (1) the attention head categorization is broken — 132/144 heads are mislabeled as "First/Positional," swamping all meaningful signal; and (2) the induction, duplicate, and diffuse head categories from the spec are entirely absent. Beyond that, the attention visualization is the weakest explanation panel — it shows the heatmap but doesn't teach the student what to look for. Ablation UX also has friction and never surfaced results in testing.

---

## 1. Overall Layout & First Impression

**What's good:**
- Clean gradient header, uncluttered layout
- The pipeline section ("How the Model Processes Your Input") is a strong pedagogical frame — the numbered steps with the flow chip bar (Input → Tokens → Embed → Attention → MLP → Output) is excellent
- Glossary modal auto-opens on first visit, which is a good onboarding move
- The sidebar module selection (showing `transformer.h.{N}.attn` etc.) is a nice power-user layer

**Issues:**
- **Glossary modal close button is off-screen** at default viewport widths. The `×` renders at x≈1858 on a 1400px window. Students on laptops will be stuck staring at a modal they can't close without scrolling right. Fix: position the close button inside the modal boundary, not at the document edge.
- **45-second cold start with no feedback.** After clicking Analyze, the pipeline stages show "Awaiting analysis..." with no progress indicator, spinner, or ETA. For a student, this looks broken. Fix: add a loading spinner or "Model is warming up (~30s)..." message on first run.
- **Generation Settings sliders are confusing.** "Number of Generation Choices" with values 1/3/5 is jargon. Students don't know what beam search is. The label should be "Explore How Many Different Continuations?" or similar, with a tooltip. The current glossary entry on Beam Search is good but isn't linked from the slider.

---

## 2. Tokenization Stage

**What's good:**
- Clean token→ID table. Exactly the right content.
- "Your text is split into 10 tokens" summary in the header is great.

**Issues:**
- **No visual "aha" moment.** The table shows Token→ID correctly, but doesn't show *why* "The" becomes 464 vs "the" becoming 262. The capitalization distinction (same word, different token) is sitting right there in this example and the app doesn't call it out. This is a perfect teachable moment — highlight it.
- **No subword tokenization example.** The prompt was simple English so all tokens were whole words. When a student types something with subwords (e.g., "transformers"), they won't know that's unusual. Consider adding a note: "Notice: some words may split into multiple pieces — try typing 'unhappiness' to see subword tokenization."
- **The token ID numbers mean nothing to students.** Worth a one-liner: "These IDs are just addresses in a vocabulary table of tens of thousands of words and word-pieces."

---

## 3. Embedding Stage

**What's good:**
- The `Token ID → Lookup Table → [768-dimensional vector]` flow diagram is clean and conceptually correct.
- The callout box ("How the lookup table was created: During training on billions of text examples...") is excellent — this is exactly the kind of "where did this come from?" context students need.

**Issues:**
- **No actual data shown.** The stage says "768-dimensional vector" but never shows a student what even 5 dimensions of that vector look like. Even a truncated display like `[0.23, -1.41, 0.07, ...]` would make it real.
- **No similarity demo.** The explanation says "words with similar meanings (like 'happy' and 'joyful') have similar vectors" — but doesn't show it. A small cosine similarity callout using tokens actually in the input ("'cat' and 'mat' are somewhat similar; 'cat' and 'The' are not") would land this point.
- **Missing: positional embeddings.** This is a significant omission. The embedding stage in a transformer is `token_embedding + positional_embedding`. The current explanation only covers token embeddings. Students who read further literature will be confused. Add: "Each token also gets a positional embedding added — a second vector encoding *where* in the sequence it appears."

---

## 4. Attention Stage

This is the most important and most underbuilt section. The handoff doc has a detailed vision that is only partially implemented.

### 4a. Head Category Panel

**Critical bug: First/Positional is consuming 132/144 heads.**

The categorization output:
- Previous-Token: 6 heads ✓ (reasonable)
- First/Positional: **132 heads** ✗ (this is ~92% of all heads — clearly wrong)
- Syntactic: 5 heads (plausible)
- Other: 1 head

This makes the category panel meaningless. A student sees a wall of 132 head IDs under "First/Positional" and learns nothing. The classification threshold for positional heads is almost certainly too loose, OR the `all_scores` from the offline script are being compared against an incorrect threshold. The handoff spec calls for a cap of ~8 heads per category with layer diversity enforcement — that logic is either not implemented or the thresholds need significant tuning.

**Missing categories from the spec:**
The handoff doc specifies 6 categories:
1. ✅ Previous Token (implemented)
2. ❌ **Induction** (missing entirely)
3. ❌ **Duplicate Token** (missing entirely)
4. ✅ First/Positional (implemented but broken threshold)
5. ❌ **Diffuse / Bag-of-Words** (missing entirely)
6. ✅ Other/Unclassified (implemented)

"Syntactic" appears as a category but isn't in the handoff spec — unclear where it came from or how it's detected.

**Missing: runtime activation scoring.** The spec calls for each head to show an activation score on the *current input* (e.g., whether induction heads are firing given the repeated "The cat" in the prompt). Nothing like this exists yet — heads are just listed as belonging to categories with no indication of whether they're active or dormant on this specific input.

**Missing: greyed-out heads with "suggested prompts."** The spec's pedagogically most powerful idea — "Try adding a repeated sentence to see induction heads light up" — doesn't exist at all. This is the thing that turns passive observation into active discovery.

### 4b. Attention Visualization (BertViz)

**What's good:**
- BertViz integration works and renders the attention heatmap
- The navigation instructions (single click, double click, hover) are clear

**Issues:**
- **No guided interpretation.** The visualization shows lines but doesn't tell the student what they're looking at. For a student who just read that "some heads track pronouns," they need a nudge: "Try Layer 4, Head 11 — this head often looks at the previous word." Right now the student opens a heatmap of spaghetti lines and has no idea what to conclude.
- **The attention viz and head category panel are disconnected.** Clicking a head in the category list should highlight/select it in the BertViz below. The handoff spec mentions this: "Clicking a head navigates to its attention heatmap." That linkage doesn't exist.
- **No explanation of what "good" attention looks like.** The viz shows all heads at once by default. For a 12×12 model that's 144 attention patterns — overwhelming. The default view should be a single interesting head (e.g., the strongest previous-token head), not all heads.
- **Layer selector is bare.** The "Layer: [dropdown]" control has no context. Why would a student change the layer? Add: "Earlier layers tend to capture syntax; later layers capture meaning."

---

## 5. MLP (Feed-Forward) Stage

**What's good:**
- The `768d → 3072d → 768d` expand/compress diagram is clean
- The "Why expand then compress?" callout box is excellent — the neuron activation framing is correct
- "This happens in each of the model's 12 layers, with attention and MLP working together" is a good summary

**Issues:**
- **No connection to the current input.** The Paris/France example is generic and not connected to the actual prompt being analyzed. Consider: "For your prompt, the MLP layers are likely retrieving knowledge about common English sentence structures."
- **No visualization.** MLP is the only stage with purely static text and a diagram. Even a simple bar chart of "top activated neurons at layer X" would make this real. The handoff doc doesn't spec this out, but it's a gap.
- **Missing: the residual stream framing.** The glossary defines "Residual Stream" but the MLP stage doesn't mention that the MLP *adds* to the residual stream rather than replacing it. This is fundamental to why the model can accumulate knowledge across layers.

---

## 6. Output Selection Stage

**What's good:**
- Top-5 next-token predictions with probability bars is exactly right
- The full-sentence context display with highlighted predicted token is excellent UX
- The "Note on Token Selection" callout about Beam Search and MoE is appropriately nuanced

**Issues:**
- **"13.5% confidence" framing is misleading.** "Confidence" implies certainty; this is a softmax probability, which is better described as "the model assigned a 13.5% probability to 'was' as the next word." Students may misread this as "the model is 13.5% confident it's right."
- **No contrast with wrong predictions.** The chart shows top-5 but doesn't explain *why* the model predicted "was" over "sat." A connection back to attribution ("The token 'cat' had the highest influence on predicting 'was'") would close the loop.
- **The token slider is unclear.** "Step through generated tokens" with a slider defaulting to 0 and showing "was" is confusing — it looks like there's nothing to step through. Label it: "Generated token 1 of 1: was" and grey out or hide the slider when only 1 token was generated.

---

## 7. Token Attribution Panel

**What's good:**
- The visualization works well — darker tokens = more important is intuitive
- The bar chart with normalized attribution scores is clean
- Results matched expectations: "was" (the second "cat" token, position 9) scored 1.0, "The" scored 0.87 — sensible given the prompt structure

**Issues:**
- **"Simple Gradient" is selected by default, not "Integrated Gradients."** The UI labels Simple Gradient as "faster, less accurate" and Integrated Gradients as "more accurate, slower" — but defaults to the less accurate one. For an educational tool where accuracy matters more than speed, this should be reversed. Or at minimum, note: "For learning purposes, Integrated Gradients gives more reliable results."
- **No explanation of what attribution scores mean in plain English.** The callout says "Tokens with higher attribution scores contributed more to the model's prediction" — but students need: "The second 'cat' scored highest because the model is pattern-matching 'The cat...' to predict what typically follows 'The cat' in English text."
- **No visual connection to the actual attention visualization.** If "was" had high attribution from "cat," students should be able to click through to see which attention heads facilitated that. Right now attribution and attention are completely siloed.
- **Target Token dropdown is confusing.** "Use top predicted token (default)" is fine, but the empty text box below it with "Leave empty to compute attribution for the top predicted token" is redundant and confusing — why show a text box that you immediately tell them not to fill?

---

## 8. Ablation Panel

**Issues (mostly UX):**
- **Ablation didn't show results in automated testing** — the head selection reset when switching tabs, suggesting state management issues between the Ablation and Attribution tabs.
- **No presets or suggestions.** The student faces a blank "Layer / Head" picker and has no idea which heads are interesting to ablate. The category panel above already identified previous-token heads (L4-H11, etc.) — there should be a "Try ablating this head" link from the category panel directly into the ablation form.
- **"Run Ablation Experiment" is permanently greyed out** until a head is added. The disabled state has no tooltip explaining why. Add: "Add at least one head above to run the experiment."
- **No explanation of what to expect.** Before running, tell students: "If this head is important, the top prediction may change. If it doesn't change, the head wasn't critical for this input."
- **No result interpretation.** After running (when it works), the diff between original and ablated predictions needs plain-English interpretation: "Removing L4-H11 changed 'was' (13.5%) → 'sat' (18.2%). This suggests that head was suppressing 'sat' as a prediction."

---

## 9. Sidebar

**What's good:**
- The "Model loaded successfully! Detected family: GPT-2 architecture" green badge is good UX
- Module selection dropdowns (Attention Modules, Layer Blocks, Normalization Parameters) make sense for power users

**Issues:**
- **Sidebar purpose is unclear to students.** There's no explanation of what changing "Attention Modules" does or why a student would want to. This entire panel reads like a developer debug tool that was left exposed.
- **"Clear Selections" does what, exactly?** No tooltip.
- Consider: either hide the sidebar behind an "Advanced" toggle for student mode, or add inline documentation for each control.

---

## 10. Chatbot (Robot Icon)

The robot icon is visible at bottom-right but the chat panel contents weren't captured in automated testing (JS error prevented inspection). Recommend manual review of the chatbot's response quality and whether it contextualizes responses to the current model/prompt state.

---

## Priority Recommendations for Cursor

### 🔴 Critical (do these first)

1. **Fix attention head categorization thresholds.** First/Positional capturing 132/144 heads makes the entire category panel meaningless. Tighten the threshold, enforce the ~8-head cap per category from the spec, and add layer diversity. This is the highest-impact fix.

2. **Add the missing head categories.** Induction, Duplicate Token, and Diffuse are all specced in `attention_handoff.md` with detection logic. They need to be implemented. Induction is especially important for this exact prompt (repeated "The cat").

3. **Fix the modal close button off-screen bug.** Students can't close the glossary modal on standard laptop viewports. Easy CSS fix: `position: absolute; right: 16px` inside the modal container, not the document.

4. **Add a loading state after clicking Analyze.** 45 seconds of static "Awaiting analysis..." with no spinner is a UX failure. Add a pulsing animation or "Loading model..." progress message.

### 🟡 High Priority

5. **Connect head categories to the BertViz visualization.** Clicking a head ID (e.g., L4-H11) in the category panel should auto-select that head in the attention viz below.

6. **Add runtime activation scoring to head categories.** Per the spec: show whether each head type is active on the current input. Gray out induction heads if there's no repetition in the input, with a "Try: 'The cat sat. The cat'" suggested prompt.

7. **Add positional embeddings to the Embedding stage explanation.** Currently missing an entire half of what embeddings are.

8. **Fix ablation state management.** Head selections shouldn't reset when switching between Ablation and Attribution tabs.

9. **Change attribution default to Integrated Gradients.** It's the more accurate method; this is an educational tool, not a speed benchmark.

10. **Capitalize on the tokenization "aha" moment.** "The" (464) vs "the" (262) is sitting right there in the example. Call it out explicitly.

### 🟢 Enhancements

11. **Add guided "what to look for" text to the attention visualization.** Pick one interesting head per model (pre-annotated) and surface it as a recommendation: "Try Layer 4, Head 11 to see a previous-token head in action."

12. **Add suggested prompts for exploring each head category.** "To see induction heads activate, try: 'The cat sat on the mat. The cat...'"

13. **Reframe "confidence" in Output stage.** Replace with "probability" throughout.

14. **Link attribution results to attention heads.** "The token 'cat' was most influential — see which heads connected it to the prediction in the Attention stage."

15. **Fix the Output stage token slider** — hide or disable it when only 1 token was generated.

16. **Add a brief "what would you like to explore?" prompt to the ablation UI** with pre-suggested heads from the category panel.

17. **Sidebar: add explanatory text** for what Module Selection controls, or hide it in an "Advanced" section.

---

## What's Already Strong (Don't Break)

- The 5-stage pipeline structure and the flow chip bar — keep it exactly as is
- The BertViz integration — it works and the navigation instructions are clear
- The callout boxes in Embedding and MLP — these are the best explanation text in the app
- The token attribution visualization (darker = more important) — intuitive and correct
- The top-5 output prediction chart — exactly the right content
- The glossary modal content — all 8 entries are well-written

---

## Comparison to Handoff Spec

| Spec Feature | Status |
|---|---|
| 6 head categories (Previous Token, Induction, Duplicate, Positional, Diffuse, Other) | ⚠️ Partial — 3/6 missing, Positional broken |
| Per-head activation scores on current input | ❌ Not implemented |
| Active/inactive state display (filled vs open circle) | ❌ Not implemented |
| Greyed-out heads with suggested prompts | ❌ Not implemented |
| Click head → navigate to attention heatmap | ❌ Not implemented |
| Runtime verification module | ❌ Not implemented |
| One-time offline analysis script | ✅ Appears to have run (JSON exists) |
| Educational tooltips per category | ⚠️ Partial — descriptions exist but brief |
