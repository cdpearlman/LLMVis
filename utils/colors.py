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
