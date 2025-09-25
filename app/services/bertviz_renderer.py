"""
BertViz rendering service for attention visualizations.
Handles generation of thumbnails and full interactive views.
"""

import logging
import base64
import io
from typing import List, Dict, Any, Optional
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from transformers import AutoTokenizer

from ..constants import TILE_WIDTH, TILE_HEIGHT
from ..utils.data_contract import extract_attention_weights, sort_modules_by_layer

logger = logging.getLogger(__name__)

def extract_layer_number_from_name(module_name: str) -> int:
    """Extract layer number from module name."""
    import re
    numbers = re.findall(r'\d+', module_name)
    if not numbers:
        return 0
    return int(numbers[0])

def extract_single_layer_attention(attention_outputs: Dict[str, Any], module_name: str) -> torch.Tensor:
    """
    Extract attention weights for a single layer module.
    
    Args:
        attention_outputs: Dictionary of attention outputs
        module_name: Name of the attention module
        
    Returns:
        Attention weights tensor for the layer
    """
    if module_name not in attention_outputs:
        raise ValueError(f"Module {module_name} not found in attention outputs")
    
    output_data = attention_outputs[module_name]["output"]
    
    # Attention modules return (output, attention_weights) tuple
    if isinstance(output_data, (list, tuple)) and len(output_data) >= 2:
        weights_data = output_data[1]
    else:
        raise ValueError(f"Expected attention output to be tuple with 2+ elements for {module_name}")
    
    if isinstance(weights_data, torch.Tensor):
        weights = weights_data
    else:
        weights = torch.tensor(weights_data)
    
    logger.info(f"Extracted attention weights for {module_name}: shape {weights.shape}")
    return weights

def get_tokens_from_input_ids(tokenizer: AutoTokenizer, input_ids: List[int]) -> List[str]:
    """
    Convert input IDs to token strings for visualization.
    
    Args:
        tokenizer: Tokenizer instance
        input_ids: List of token IDs
        
    Returns:
        List of printable token strings
    """
    raw_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # Clean tokens for visualization
    tokens = []
    for token in raw_tokens:
        if token.startswith('Ġ'):
            # GPT-2 style: Ġ prefix indicates space
            clean_token = token.replace('Ġ', ' ')
        elif token.startswith('▁'):
            # SentencePiece style: ▁ prefix indicates space
            clean_token = token.replace('▁', ' ')
        else:
            clean_token = token
        tokens.append(clean_token)
    
    return tokens

def create_attention_heatmap(attention_weights: torch.Tensor, tokens: List[str]) -> go.Figure:
    """
    Create a simple attention heatmap using Plotly.
    
    Args:
        attention_weights: Attention tensor [batch, heads, seq_len, seq_len]
        tokens: List of token strings
        
    Returns:
        Plotly figure
    """
    # Take first batch, average over heads
    if len(attention_weights.shape) == 4:
        attn_matrix = attention_weights[0].mean(dim=0).numpy()  # [seq_len, seq_len]
    elif len(attention_weights.shape) == 3:
        attn_matrix = attention_weights.mean(dim=0).numpy()
    else:
        raise ValueError(f"Unexpected attention shape: {attention_weights.shape}")
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=attn_matrix,
        x=tokens,
        y=tokens,
        colorscale='Blues',
        showscale=False,
        hovertemplate='From: %{y}<br>To: %{x}<br>Attention: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        width=TILE_WIDTH,
        height=TILE_HEIGHT,
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        showlegend=False
    )
    
    return fig

def render_layer_thumbnail(
    attention_weights: torch.Tensor, 
    tokens: List[str], 
    size: tuple = (TILE_WIDTH, TILE_HEIGHT)
) -> str:
    """
    Create a lightweight thumbnail visualization for a single layer.
    
    Args:
        attention_weights: Attention weights for the layer
        tokens: List of token strings
        size: Size tuple (width, height)
        
    Returns:
        Base64-encoded image string or HTML string
    """
    try:
        # Create simple heatmap
        fig = create_attention_heatmap(attention_weights, tokens)
        
        # Convert to image
        img_bytes = pio.to_image(fig, format='png', width=size[0], height=size[1])
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Return as data URL
        thumbnail_html = f'<img src="data:image/png;base64,{img_base64}" width="{size[0]}" height="{size[1]}" style="border: 1px solid #ddd; border-radius: 4px;">'
        
        logger.info(f"Generated thumbnail for layer with {len(tokens)} tokens")
        return thumbnail_html
        
    except Exception as e:
        logger.error(f"Failed to create thumbnail: {e}")
        # Return placeholder
        return f'<div style="width:{size[0]}px; height:{size[1]}px; border: 1px solid #ddd; display:flex; align-items:center; justify-content:center; color:#666;">Attention Preview<br/>({len(tokens)} tokens)</div>'

def render_layer_head_view(attention_weights: torch.Tensor, tokens: List[str]) -> str:
    """
    Generate full interactive BertViz head view for a single layer.
    
    Args:
        attention_weights: Attention weights for the layer
        tokens: List of token strings
        
    Returns:
        HTML string with interactive BertViz head view
    """
    try:
        from bertviz import head_view
        
        # BertViz expects attention as tuple of tensors (one per layer)
        # We have a single layer, so wrap it
        attentions = (attention_weights.unsqueeze(0),)  # Add batch dim if needed
        
        # Generate head view HTML
        html_output = head_view(attentions, tokens, html_action='return')
        
        # Extract HTML content
        if hasattr(html_output, 'data'):
            html_content = html_output.data
        else:
            html_content = str(html_output)
        
        logger.info(f"Generated head view for layer with {attention_weights.shape}")
        return html_content
        
    except Exception as e:
        logger.error(f"Failed to generate head view: {e}")
        
        # Fallback: create a detailed heatmap
        try:
            fig = create_attention_heatmap(attention_weights, tokens)
            fig.update_layout(
                width=800,
                height=600,
                title="Attention Patterns (Head View Fallback)",
                xaxis=dict(showticklabels=True, tickangle=45),
                yaxis=dict(showticklabels=True)
            )
            
            html_content = pio.to_html(fig, include_plotlyjs='cdn')
            return html_content
            
        except Exception as e2:
            logger.error(f"Fallback also failed: {e2}")
            return f"<div style='padding:20px; color:red;'>Failed to generate head view: {e}</div>"

def render_full_model_view(all_attention_weights: List[torch.Tensor], tokens: List[str]) -> str:
    """
    Generate full interactive BertViz model view across all layers.
    
    Args:
        all_attention_weights: List of attention weights, one per layer
        tokens: List of token strings
        
    Returns:
        HTML string with interactive BertViz model view
    """
    try:
        from bertviz import model_view
        
        # BertViz expects tuple of tensors
        attentions = tuple(all_attention_weights)
        
        # Generate model view HTML
        html_output = model_view(attentions, tokens, html_action='return')
        
        # Extract HTML content
        if hasattr(html_output, 'data'):
            html_content = html_output.data
        else:
            html_content = str(html_output)
        
        logger.info(f"Generated model view for {len(all_attention_weights)} layers")
        return html_content
        
    except Exception as e:
        logger.error(f"Failed to generate model view: {e}")
        
        # Fallback: create layered heatmaps
        try:
            from plotly.subplots import make_subplots
            
            rows = min(4, len(all_attention_weights))
            cols = max(1, len(all_attention_weights) // rows)
            
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=[f"Layer {i}" for i in range(len(all_attention_weights))],
                specs=[[{"type": "xy"}] * cols for _ in range(rows)]
            )
            
            for i, weights in enumerate(all_attention_weights[:rows*cols]):
                row = (i // cols) + 1
                col = (i % cols) + 1
                
                if len(weights.shape) == 4:
                    attn_matrix = weights[0].mean(dim=0).numpy()
                elif len(weights.shape) == 3:
                    attn_matrix = weights.mean(dim=0).numpy()
                else:
                    continue
                
                fig.add_trace(
                    go.Heatmap(
                        z=attn_matrix,
                        colorscale='Blues',
                        showscale=(i == 0)
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(
                height=200 * rows,
                title="Multi-Layer Attention Patterns (Model View Fallback)"
            )
            
            html_content = pio.to_html(fig, include_plotlyjs='cdn')
            return html_content
            
        except Exception as e2:
            logger.error(f"Fallback also failed: {e2}")
            return f"<div style='padding:20px; color:red;'>Failed to generate model view: {e}</div>"

def extract_all_attention_weights(activation_data: Dict[str, Any]) -> List[torch.Tensor]:
    """
    Extract attention weights for all layers from activation data.
    
    Args:
        activation_data: Complete activation data
        
    Returns:
        List of attention weight tensors, sorted by layer
    """
    attention_weights_dict = extract_attention_weights(activation_data)
    
    # Sort by layer number
    sorted_modules = sort_modules_by_layer(list(attention_weights_dict.keys()))
    
    # Extract weights in order
    all_weights = []
    for module_name in sorted_modules:
        weights = attention_weights_dict[module_name]
        all_weights.append(weights)
    
    logger.info(f"Extracted attention weights for {len(all_weights)} layers")
    return all_weights

def generate_bertviz_outputs(
    activation_data: Dict[str, Any],
    tokens: List[str],
    layer_idx: Optional[int] = None
) -> Dict[str, str]:
    """
    Generate all BertViz outputs for the given data.
    
    Args:
        activation_data: Complete activation data
        tokens: List of token strings
        layer_idx: If provided, generate layer-specific views for this layer
        
    Returns:
        Dictionary with generated HTML content
    """
    results = {}
    
    try:
        # Extract all attention weights
        all_attention_weights = extract_all_attention_weights(activation_data)
        
        if not all_attention_weights:
            raise ValueError("No attention weights found")
        
        # Generate thumbnails for all layers
        results['thumbnails'] = {}
        for i, weights in enumerate(all_attention_weights):
            thumbnail_html = render_layer_thumbnail(weights, tokens)
            results['thumbnails'][i] = thumbnail_html
        
        # Generate layer-specific head view if requested
        if layer_idx is not None and 0 <= layer_idx < len(all_attention_weights):
            layer_weights = all_attention_weights[layer_idx]
            results['head_view'] = render_layer_head_view(layer_weights, tokens)
        
        # Generate full model view
        results['model_view'] = render_full_model_view(all_attention_weights, tokens)
        
        logger.info(f"Generated BertViz outputs: {list(results.keys())}")
        
    except Exception as e:
        logger.error(f"Failed to generate BertViz outputs: {e}")
        results['error'] = str(e)
    
    return results
