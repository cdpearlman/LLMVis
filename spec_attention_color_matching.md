# SPEC: Head Color Swatches in LLMVis
**Status:** Ready for implementation  
**Date:** 2026-03-05  

---

## Goal

Add colored square swatches next to attention head labels in:
1. The attention head category panels (pipeline.py)
2. The ablation panel's selected head chips (ablation_panel.py)

The colors must match what BertViz's head_view renders for each head index, so students can visually correlate "L4-H11 [magenta square]" in the panel with the magenta checkbox in the BertViz visualization.

---

## Core Insight

BertViz's `head_view(..., html_action='return')` returns the full JS inlined as a string. We can patch `d3.schemeCategory10` → a custom 24-color palette at generation time. The swatches in our UI then reference the same palette — perfect correlation, zero color collisions.

**Why not schemeCategory10?** GPT-2 (default model) has 12 heads/layer. schemeCategory10 cycles every 10, so H10 = H0 (both blue) and H11 = H1 (both orange). That defeats the purpose.

---

## The Palette

Dark24 (Plotly) with two near-invisible entries replaced:

```python
BERTVIZ_HEAD_COLORS = [
    '#2E91E5',  # H0  - blue
    '#E15F99',  # H1  - pink-red
    '#1CA71C',  # H2  - green
    '#FB0D0D',  # H3  - red
    '#DA16FF',  # H4  - purple
    '#00B4D8',  # H5  - cyan       (was #222A2A near-black → swapped)
    '#B68100',  # H6  - gold
    '#06D6A0',  # H7  - teal       (was #750D86 very dark purple → swapped)
    '#EB663B',  # H8  - orange
    '#511CFB',  # H9  - indigo
    '#00A08B',  # H10 - dark teal
    '#FB00D1',  # H11 - magenta
    '#FC0080',  # H12 - hot pink
    '#B2828D',  # H13 - mauve
    '#6C7C32',  # H14 - olive
    '#778AAE',  # H15 - slate blue
]
```

16 colors covers all supported models (max 16 heads/layer: gpt2-medium, pythia-410m, Qwen2.5-0.5B).

---

## Files to Change

### 1. New file: `utils/colors.py`
Define `BERTVIZ_HEAD_COLORS` here as the single source of truth. Import from here everywhere else.

```python
"""Shared color constants for LLMVis."""

# 16-color palette for BertViz head_view.
# Patched into BertViz at generation time (model_patterns.py) and mirrored
# in UI swatches (pipeline.py, ablation_panel.py) for visual correlation.
# Based on Plotly Dark24, with H5 and H7 replaced for legibility.
BERTVIZ_HEAD_COLORS = [
    '#2E91E5',  # H0  - blue
    '#E15F99',  # H1  - pink-red
    '#1CA71C',  # H2  - green
    '#FB0D0D',  # H3  - red
    '#DA16FF',  # H4  - purple
    '#00B4D8',  # H5  - cyan
    '#B68100',  # H6  - gold
    '#06D6A0',  # H7  - teal
    '#EB663B',  # H8  - orange
    '#511CFB',  # H9  - indigo
    '#00A08B',  # H10 - dark teal
    '#FB00D1',  # H11 - magenta
    '#FC0080',  # H12 - hot pink
    '#B2828D',  # H13 - mauve
    '#6C7C32',  # H14 - olive
    '#778AAE',  # H15 - slate blue
]

def head_color(head_index: int) -> str:
    """Return the swatch color for a given head index."""
    return BERTVIZ_HEAD_COLORS[head_index % len(BERTVIZ_HEAD_COLORS)]
```

---

### 2. `utils/model_patterns.py` — `generate_bertviz_html()`

After the `head_view(...)` call, patch the returned HTML string to replace BertViz's default color scheme with ours. Both the primary and fallback d3 lines must be replaced.

**Locate:** the `else:` block in `generate_bertviz_html()` (around line 1414):
```python
html_result = head_view(attentions, tokens, html_action='return')
return html_result.data if hasattr(html_result, 'data') else str(html_result)
```

**Replace with:**
```python
from utils.colors import BERTVIZ_HEAD_COLORS
html_result = head_view(attentions, tokens, html_action='return')
html_str = html_result.data if hasattr(html_result, 'data') else str(html_result)

# Patch BertViz color scheme to match our swatch palette (no collisions for ≤16 heads)
_colors_js = repr(BERTVIZ_HEAD_COLORS).replace("'", '"')  # JSON-safe array literal
_patch = f"headColors = d3.scaleOrdinal({_colors_js});"
html_str = html_str.replace(
    'headColors = d3.scaleOrdinal(d3.schemeCategory10);',
    _patch
)
html_str = html_str.replace(
    'headColors = d3.scale.category10();',
    _patch
)
return html_str
```

---

### 3. `components/pipeline.py` — `create_attention_content()`

Import `head_color` from `utils.colors` at the top of the file.

In the `head_items` loop (around line 536), the head label is currently:
```python
html.Span(label, style={
    'fontFamily': 'monospace', 'fontSize': '12px', 'fontWeight': '500',
    'minWidth': '60px', 'color': '#495057' if is_active else '#aaa',
}, title=f"See Layer {head_info['layer']}, Head {head_info['head']} in the visualization below"),
```

**Replace with:**
```python
html.Span([
    html.Span("■ ", style={
        'color': head_color(head_info['head']),
        'fontSize': '14px',
    }),
    label
], style={
    'fontFamily': 'monospace', 'fontSize': '12px', 'fontWeight': '500',
    'minWidth': '60px', 'color': '#495057' if is_active else '#aaa',
    'display': 'inline-flex', 'alignItems': 'center',
}, title=f"See Layer {head_info['layer']}, Head {head_info['head']} in the visualization below"),
```

Note: The swatch always renders at full color even for inactive heads — the parent `html.Div` already applies `opacity: 0.5` for inactive heads, which will naturally dim the swatch too. No special handling needed.

---

### 4. `components/ablation_panel.py` — `create_selected_heads_display()`

Import `head_color` from `utils.colors` at the top of the file.

In the chips loop, the label is currently:
```python
label = f"L{layer}-H{head}"

chips.append(
    html.Span([
        html.Span(label, style={'marginRight': '6px'}),
        ...
    ], ...)
)
```

**Replace** the `html.Span(label, ...)` with:
```python
html.Span([
    html.Span("■ ", style={
        'color': head_color(head),
        'fontSize': '14px',
        'marginRight': '2px',
    }),
    html.Span(label, style={'marginRight': '6px'}),
], style={'display': 'inline-flex', 'alignItems': 'center'}),
```

---

## Exit Criteria

- [ ] BertViz head_view renders with the new palette — H0 is blue, H10 is dark teal, H11 is magenta (visually distinct from H1)
- [ ] Every head label in the attention category panels shows a colored square matching BertViz
- [ ] Every ablation chip shows a matching colored square
- [ ] Spot checks: `L0-H0` = `#2E91E5`, `L4-H10` = `#00A08B`, `L4-H11` = `#FB00D1`
- [ ] `BERTVIZ_HEAD_COLORS` / `head_color()` defined in exactly one place (`utils/colors.py`), imported everywhere else
- [ ] No functional changes to callbacks, stores, or any other logic
- [ ] Existing tests still pass

## Non-Goals
- No changes to model_view (not used in this app)
- No hover/interactive effects on swatches
- No changes to per-category color scheme (the purple/blue/green category accent colors are separate)
- No changes beyond the 4 files listed above
