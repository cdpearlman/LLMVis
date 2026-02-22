# Handoff: Attention Head Categorization System

## Architecture Overview

```
ONE-TIME (offline)                    RUNTIME (per user interaction)
┌─────────────────────┐              ┌────────────────────────────┐
│ TransformerLens      │              │ PyVene forward pass        │
│ analysis script      │──► JSON ──►│ (already exists)           │
│ (runs once per model)│   file      │         │                  │
└─────────────────────┘              │         ▼                  │
                                     │ Activation checker         │
                                     │ (lightweight, no TL dep)   │
                                     │         │                  │
                                     │         ▼                  │
                                     │ UI: heads shown with       │
                                     │ active/inactive state      │
                                     └────────────────────────────┘
```

## Categories to Implement

**6 categories**, each with its TL detection method and runtime verification:

| # | Category | TL Detection Method | Runtime Verification | Educational Explanation |
|---|----------|---------------------|----------------------|-------------------------|
| 1 | **Previous Token** | Built-in `"previous_token_head"`. Pattern: diagonal offset -1. Run on any text. | Check diagonal-1 attention mass > threshold. Always applicable. | "This head looks at the word right before the current one. Like reading left to right." |
| 2 | **Induction** | Built-in `"induction_head"`. Pattern: token after prior occurrence of current token. Run on 50+ random repeated sequences, average scores. | Find repeated tokens in user input. Check if attention follows the [A][B]...[A]→B pattern. Gray out if no repetition in input. | "This head finds patterns that happened before and predicts they'll happen again. If it saw 'the cat' earlier, it expects the same words to follow." |
| 3 | **Duplicate Token** | Built-in `"duplicate_token_head"`. Pattern: attention to positions with same token. Run on same repeated sequences as induction. | Check if attention concentrates on positions with identical token IDs. Gray out if no duplicates in input. | "This head notices when the same word appears more than once, like a highlighter for repeated words." |
| 4 | **Positional / First-Token** | Custom pattern: column 0 = 1, rest = 0. Run on varied text. | Check column-0 attention mass > threshold. Always applicable. | "This head always pays attention to the very first word, using it as an anchor point." |
| 5 | **Diffuse / Bag-of-Words** | Custom metric (not pattern-based): compute normalized entropy of each head's attention distribution across many inputs. High entropy = diffuse. | Check if attention entropy is high and max attention is low. Always applicable. | "This head spreads its attention evenly across many words, gathering general context rather than focusing on one spot." |
| 6 | **Other / Unclassified** | Heads that score below threshold on all 5 categories above. | No runtime check needed. Show as neutral. | "This head's pattern doesn't fit our simple categories -- it may be doing something more complex." |

## Component 1: One-Time Analysis Script

**File:** `scripts/analyze_heads.py` (new, standalone, not part of the Dash app)

**Dependencies:** `transformer-lens`, `torch`, `json` (only needed for this script, not at runtime)

**Workflow:**

1. For each model in the target list:
   - Load as `HookedTransformer`
   - Generate test inputs:
     - 50 random repeated-token sequences (for induction + duplicate detection)
     - 20 varied natural-language sentences (for previous-token, positional, diffuse)
   - Run `detect_head()` for each built-in category, averaging scores across inputs
   - Run custom detection for positional (column-0 pattern) and diffuse (entropy computation)
   - Collect all scores into a `[n_layers, n_heads]` tensor per category
2. For each category, identify "top heads":
   - Threshold-based: all heads with score > 0.4 (tune per category)
   - Enforce layer diversity: if top heads cluster in one layer, also include the best head from other layers that exceeds a lower threshold (e.g., 0.25)
   - Cap at ~8 heads per category to keep UI manageable
3. Write JSON output

**Target models to analyze (verify TL support first):**

- `gpt2` (definitely supported)
- `Qwen/Qwen2.5-0.5B` (verify -- TL has Qwen weight converters)
- `EleutherAI/pythia-70m` through `pythia-410m` (if you re-enable Pythia in the UI)
- `facebook/opt-125m` (if you re-enable OPT)
- research more target models and re-configure @utils/model_config.py

## Component 2: JSON Data File

**File:** `utils/head_categories.json`

**Structure:**

```json
{
  "gpt2": {
    "model_name": "gpt2",
    "num_layers": 12,
    "num_heads": 12,
    "analysis_date": "2026-02-16",
    "categories": {
      "previous_token": {
        "display_name": "Previous Token",
        "description": "Attends to the immediately preceding token",
        "icon": "arrow-left",
        "top_heads": [
          {"layer": 4, "head": 11, "score": 0.87},
          {"layer": 2, "head": 3, "score": 0.72}
        ]
      },
      "induction": {
        "display_name": "Induction",
        "description": "Completes repeated patterns: [A][B]...[A] → [B]",
        "icon": "repeat",
        "requires_repetition": true,
        "top_heads": [
          {"layer": 5, "head": 5, "score": 0.95},
          {"layer": 5, "head": 1, "score": 0.91},
          {"layer": 6, "head": 9, "score": 0.88}
        ]
      }
    },
    "all_scores": {
      "previous_token": [[0.12, 0.05, ...], ...],
      "induction": [[0.01, 0.02, ...], ...]
    }
  }
}
```

The `all_scores` matrix (full `[n_layers][n_heads]` scores) is included for potential future use (heatmap of head roles, etc.) but the `top_heads` lists are what the UI consumes.

## Component 3: Runtime Verification Module

**File:** Extend existing `utils/head_detection.py`

**What changes:**

- Evaluate existing functionality and remove all functions and excess code that will be replaced or is unnecessary
- Add a `load_head_categories(model_name)` function that reads from the JSON
- Add a `verify_head_activation(attention_weights, tokens, head_info, category)` function that:
  - Takes the attention matrix `[seq_len, seq_len]` for a specific head
  - Takes the input token IDs
  - Takes the category name
  - Returns an activation score (0.0 to 1.0)
- Each category has its own verification logic:

| Category | Verification Logic |
|----------|-------------------|
| `previous_token` | Mean of diagonal-1 values |
| `induction` | If repeated tokens exist: measure attention from position i to j+1 where token[i]==token[j]. If no repeats: return 0.0 (gray) |
| `duplicate_token` | If repeated tokens exist: measure attention from later occurrence to earlier occurrence. If no repeats: return 0.0 |
| `positional` | Mean of column-0 attention values |
| `diffuse` | Normalized entropy of attention distribution |

- Add a `get_active_head_summary(activation_data, model_name)` function that:
  - Loads categories from JSON
  - For each top head in each category, runs verification on the current attention weights
  - Returns a structure the UI can consume: `{category: [{layer, head, score, activation_score, is_active}, ...]}`

**Key design point:** This module does NOT import TransformerLens. It uses only `torch` and the attention weight tensors already captured by PyVene. The pattern-comparison math from TL's source is ~15 lines that you reimplement directly.

## Component 4: UI Changes

**File:** Extend `components/investigation_panel.py` or create a new section in the pipeline view

**Display concept:**

```
┌─────────────────────────────────────────┐
│ Attention Head Roles                    │
│                                         │
│ ● Previous Token          ○ Induction   │
│   L4-H11 ████████░░ 0.82    L5-H5 (no  │
│   L2-H3  ██████░░░░ 0.65    repetition  │
│                              in input)   │
│ ● Positional              ○ Duplicate   │
│   L0-H1  █████████░ 0.91    L3-H0 (no  │
│   L1-H4  ██████░░░░ 0.58    duplicates) │
│                                         │
│ ● Diffuse/Spread                        │
│   L7-H8  ████████░░ 0.78               │
│                                         │
│  ● = active on your input               │
│  ○ = role exists but not triggered      │
│                                         │
│  ⓘ Why are some grayed out?             │
│  "Some heads only activate when your    │
│   input has specific patterns, like     │
│   repeated words. Try: 'The cat sat     │
│   on the mat. The cat slept.'"          │
└─────────────────────────────────────────┘
```

**Key UI elements:**

- Filled circle (active) vs open circle (inactive/grayed) for each category
- Per-head activation bars showing runtime strength
- A tooltip/info box explaining *why* heads are grayed (with suggested prompts that would activate them)
- Clicking a head navigates to its attention heatmap in the existing BertViz visualization

**The "suggested prompt" is pedagogically powerful:** it invites the student to experiment. "Try adding a repeated sentence to see induction heads light up." This turns passive observation into active discovery.

## Implementation Order

1. **Verify TL model support** for each target model (quick test: can you `HookedTransformer.from_pretrained("gpt2")` and `"Qwen/Qwen2.5-0.5B"`?)
2. **Write the one-time script** (`scripts/analyze_heads.py`) -- start with GPT-2 only
3. **Generate the JSON** for GPT-2
4. **Build the runtime verification** in `utils/head_detection.py` (extend, don't replace existing code)
5. **Build the UI component**
6. **Run the script** on remaining models, expanding the JSON
7. **Add the educational tooltips and suggested prompts**

## What This Does NOT Cover (Future Work)

- **Successor heads, name movers, copy suppression:** These require output/logit analysis, not attention-pattern analysis. They could be added later via MAPS or manual annotation.
- **Polysemanticity:** A head can belong to multiple categories. The JSON supports this (a head can appear in multiple `top_heads` lists). The UI should communicate this -- "This head is primarily an induction head but also shows previous-token behavior."
- **Per-input category discovery:** This system identifies *known* heads. It doesn't discover new categories or identify heads doing unexpected things on a specific input. Your existing heuristic code could remain as a secondary "what's happening right now" view.
