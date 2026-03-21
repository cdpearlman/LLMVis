# Showcase Demo Plan

## Context
Presenting the Transformer Explanation Dashboard at a science-fair-style showcase. Free-roam format, K-12 students (mixed grades, elementary through high school), groups of 3-5 at a time, ~1-3 minute interactions. Large external monitor with mouse for students. Engagement is the primary goal; feedback (QR code form on phone) is secondary.

## Strategy: "Guided Magic Trick" (Hybrid of Options 1 + 3)

The chatbot is the anchor. You run a short, repeatable demo loop that feels like showing a magic trick, with a volunteer moment built in. The loop is designed to be restarted at any point when a new group walks up.

---

### The Demo Loop (~2 minutes)

**1. Hook (15 sec)**
- Chatbot is open on screen with a friendly pre-loaded exchange visible (something like "How does AI understand words?")
- As students approach: *"Have you ever used ChatGPT? Want to see what's happening inside its brain?"*
- For younger kids: *"Want to see how a robot reads?"*

**2. The Question (15 sec)**
- Ask a student to suggest a short sentence, or use a pre-loaded one if they're shy
- Type it into the app (or have them type it — handing the mouse over is a good engagement moment for one student)
- Run inference so the dashboard populates

**3. The Reveal (30 sec)**
- Switch to the visualization view. *"This is what the AI sees when it reads your sentence."*
- Show the attention map — point out how some words "look at" other words more. Use simple language:
  - High school: *"Each word figures out which other words matter most for understanding it"*
  - Middle school: *"It's like the AI highlighting the important connections"*
  - Elementary: *"See how these words are friends? The bright colors mean they're paying attention to each other"*
- This is the visual wow moment — attention heatmaps are inherently striking

**4. The Experiment (30 sec) — Volunteer Moment**
- *"What do you think happens if we break part of the AI's brain?"*
- Hand the mouse to a student. Guide them to disable an attention head (ablation)
- Let them click. Output changes. Visible reaction.
- *"You just did what AI researchers do — you ran an experiment on a neural network."*

**5. The Exit (15 sec)**
- *"If you thought that was cool, I'd love your feedback — scan this QR code and tell me what you'd want to explore next"*
- Point to QR code
- If they linger, offer the chatbot: *"You can also ask the AI chatbot any question about how it works"* — this gives curious students a self-directed path while you reset for the next group

---

### Adapting to the Audience On-the-Fly

| Signal | Adapt |
|--------|-------|
| Young kids (K-3) | Skip the experiment step, focus on the visual ("look at the colors!"). Keep it to 60 seconds. |
| Shy group | Don't force the mouse hand-off. You drive, narrate enthusiastically. |
| Engaged teen | Let them drive. Show them the chatbot. Point them to activation steering if they want more. Let them stay. |
| Large crowd (5+) | You drive the whole thing. Speak louder. Keep it short and punchy. Don't hand off the mouse. |
| Someone asks "how does it really work?" | This is your dream scenario — point them to the chatbot and the data flow visualization. Let them explore. |
| Empty moment (no one at table) | Reset to chatbot with an eye-catching exchange on screen. Good time for a break. |

---

### Physical Setup

- **Monitor**: Dashboard visible, chatbot tab open as default state
- **Laptop**: Facing you, used for quick resets or switching views
- **Mouse**: On the student side of the table, positioned for easy hand-off
- **QR code**: Printed, propped up visibly near the monitor (not flat on table — eye level if possible)
- **Optional**: A one-line sign/card: *"See inside an AI's brain"* or *"How does ChatGPT think?"* — a hook for passersby

---

### Concrete "Wow Moment" Experiments (GPT-2 — 12 layers, 12 heads)

Use **5-8 generated tokens** — fast enough to keep the demo snappy, long enough to see divergence when heads are ablated.

#### Experiment 1: "Breaking Pattern Memory" (Induction Heads) — PRIMARY DEMO

**Prompt:** `The cat sat on the mat. The cat sat on the`

**Why it's the best pick:**
- Every student can see the pattern and predict "mat"
- The model should also predict "mat" — normal output feels satisfying
- Ablating induction heads should break pattern completion — the model "forgets" it already saw this
- Universal across all ages

**Attention view talking points:**
- Before ablation: attention lines from the second "cat" reach back to the first "cat" and "mat"
- *"See how the AI is looking back at the first sentence? It's copying the pattern."*

**Heads to ablate (ranked by impact):**
1. **Layer 5, Head 1** (score 0.44) — strongest induction head, try first
2. **Layer 5, Head 5** (score 0.41)
3. **Layer 5, Head 0** (score 0.34)

GPT-2's induction heads cluster heavily in Layer 5. This is actually a bonus for the demo — you can say *"all the pattern-matching lives in one layer"* and ablate multiple heads in the same layer with a few clicks.

**Testing protocol:**
- [ ] Run normal inference → confirm "mat" is top prediction
- [ ] Note the attention pattern in BertViz (lines from second occurrence back to first)
- [ ] Ablate L5-H1 alone → does the prediction change?
- [ ] Ablate L5-H1 + L5-H5 together → stronger effect?
- [ ] Ablate L5-H0 + L5-H1 + L5-H5 (triple knockout in one layer) → maximum disruption
- [ ] Also try L6-H9 (score 0.30) and L7-H10 (score 0.28) if Layer 5 ablation alone isn't enough
- [ ] Find the **minimal ablation** that visibly changes the output (ideal = one click, big change)
- [ ] Record before/after outputs

**Narration by age:**
- Elementary: *"We broke its memory! It forgot what it already read!"*
- Middle school: *"We turned off the pattern-matching part."*
- High school: *"These are called induction heads — they detect and complete repeated sequences. Notice how they're all in Layer 5."*

---

#### Experiment 2: "Breaking Grammar" (Previous Token Heads) — BACKUP DEMO

**Prompt:** `The chef gave the waiter a generous tip because`

**Why it works:**
- Natural sentence anyone understands
- Previous-token heads maintain word-by-word coherence (grammar)
- Ablating them can produce broken or nonsensical continuations

**Heads to ablate:**
1. **Layer 4, Head 11** (score 0.97) — near-perfect previous-token head, GPT-2's strongest specialized head overall
2. **Layer 2, Head 2** (score 0.57)
3. **Layer 3, Head 7** (score 0.44)

L4-H11 at 0.97 is remarkable — it's almost entirely dedicated to "look at the word right before me." This makes it a great single-click ablation target.

**Testing protocol:**
- [ ] Run normal inference → note coherent continuation
- [ ] Ablate L4-H11 alone → does grammar degrade? (high likelihood given 0.97 score)
- [ ] Ablate L4-H11 + L2-H2 → more degradation?
- [ ] Generate ~8 tokens — does output become nonsensical?
- [ ] **If effect is subtle, deprioritize** — move Experiment 3 to backup

**Narration:** *"This part of the AI's brain reads one word at a time, left to right — like you do. This one head spends 97% of its effort just looking at the previous word. What happens if we turn it off?"*

---

#### Experiment 3: "Double Knockout" (Induction + Duplicate Token) — GRAND FINALE

**Same prompt as Experiment 1:** `The cat sat on the mat. The cat sat on the`

This layers on top of Experiment 1 for engaged groups who want more.

**How it works:** Duplicate-token heads notice "I've seen 'the' and 'cat' before" while induction heads complete the pattern. They're a team — ablating both should be devastating.

**Duplicate-token heads to ablate:**
1. **Layer 0, Head 1** (score 0.42) — strongest
2. **Layer 0, Head 5** (score 0.42)
3. **Layer 1, Head 11** (score 0.33)

Note: GPT-2's duplicate-token heads are in early layers (0-1) while induction heads are in Layer 5. This makes a nice story: *"The early layers spot the repetition, the middle layers use it to predict what comes next."*

**Testing protocol:**
- [ ] With Experiment 1's prompt loaded, ablate duplicate heads alone → compare impact to induction ablation
- [ ] Ablate BOTH induction (L5-H1) + duplicate (L0-H1) heads together
- [ ] Try full combo: all induction + all duplicate heads → maximum "brain damage"
- [ ] Record the most dramatic combo

**Narration:** *"First we turned off pattern matching. Now let's ALSO turn off the part that notices repeated words... watch what happens."*

---

#### Experiment 4: "The Bat Problem" (Ambiguity — Attention Viz Only)

**Prompt:** `The bat flew over the`

**Why it works:**
- "Bat" is ambiguous (animal vs. baseball)
- Attention should show the model connecting "bat" to "flew" to disambiguate
- Great for the "Reveal" step (step 3 in demo loop) — pure attention visualization wow
- Less about ablation, more about "look what the AI notices"

**What to look for:**
- [ ] Run inference → prediction should be flying-context (field, trees, fence, etc.)
- [ ] In BertViz, find which heads strongly connect "bat" → "flew"
- [ ] *"The AI figured out this is a flying bat, not a baseball bat — look at how it connects 'bat' to 'flew'."*

**Optional ablation:** If you find a head that strongly links "bat" → "flew", try ablating it — does the prediction shift to baseball context? Would be extraordinary if it works, but may not. Test it.

---

#### Pre-Testing Workflow

**Round 1: Find your champion** (30 min)
1. Load GPT-2
2. Run Experiment 1 — test all ablation combos, record outputs
3. Run Experiment 2 — test ablation combos, record outputs (L4-H11 at 0.97 is especially promising)
4. Run Experiment 4 — note attention patterns
5. Pick primary demo based on most dramatic, reliable difference

**Round 2: Lock in details** (15 min)
6. For champion demo: find the **single-head ablation** that produces the biggest change
7. Note exact before/after outputs
8. Practice narration 3x
9. If single-head isn't dramatic enough, prepare a 2-head combo

**Round 3: Backup ready** (10 min)
10. Prepare second demo (different prompt, different head category)
11. Memorize which heads to click for both demos

#### Quick Reference Card (fill in after testing, bring to showcase)

| Demo | Prompt | Normal Output | Ablate | Ablated Output |
|------|--------|---------------|--------|----------------|
| Primary | | | L?-H? | |
| Backup | | | L?-H? | |

---

### Pre-Showcase Prep Checklist

- [ ] Identify 2-3 pre-loaded sentences that produce visually striking attention patterns (you said you have these)
- [ ] Practice the 2-minute loop 3-5 times until it's smooth and natural
- [ ] Test the monitor setup — make sure text is readable from 4-5 feet away (increase font/zoom if needed)
- [ ] Verify the ablation demo works reliably with your chosen sentences (don't want a dud in front of students)
- [ ] Print QR code large enough to scan from arm's length
- [ ] Have a backup sentence ready in case a student's suggestion produces boring attention patterns
- [ ] Check that dark mode / light mode looks good on the external monitor under showcase lighting

---

### Key Principles

1. **You are the narrator, not the app.** The app is a prop for your story. The magic is in your framing, not the UI.
2. **Repeatable > perfect.** You'll run this loop 20+ times. It needs to be tight and restartable, not elaborate.
3. **One wow moment is enough.** Don't try to show everything. The attention map + ablation is your one-two punch.
4. **Hand off the mouse, not the narrative.** Students click where you point. You keep talking.
5. **Read the room fast.** You have ~5 seconds to gauge the group's age/interest level and calibrate your language.
