"""
Layout mathematics for grid positioning and edge path computation.
Handles the distribution of layer tiles across up to 4 rows and curved edge paths.
"""

import math
import logging
from typing import List, Tuple, Dict, Any

from ..constants import MAX_ROWS, TILE_WIDTH, TILE_HEIGHT, CURVE_FACTOR

logger = logging.getLogger(__name__)

def compute_grid_positions(num_layers: int, max_rows: int = MAX_ROWS) -> List[Tuple[int, int]]:
    """
    Compute grid positions for layer tiles, distributing evenly across up to max_rows.
    
    Args:
        num_layers: Number of layers to position
        max_rows: Maximum number of rows to use
        
    Returns:
        List of (row, col) positions for each layer
    """
    if num_layers <= 0:
        return []
    
    # Calculate optimal distribution
    if num_layers <= max_rows:
        # Few layers: use single row
        positions = [(0, i) for i in range(num_layers)]
    else:
        # Many layers: distribute evenly across rows
        cols_per_row = math.ceil(num_layers / max_rows)
        
        positions = []
        for layer_idx in range(num_layers):
            row = layer_idx // cols_per_row
            col = layer_idx % cols_per_row
            positions.append((row, col))
    
    logger.info(f"Positioned {num_layers} layers in grid: max {max(p[0] for p in positions) + 1} rows")
    return positions

def calculate_tile_coordinates(positions: List[Tuple[int, int]], spacing: int = 20) -> List[Dict[str, float]]:
    """
    Calculate pixel coordinates for tiles based on grid positions.
    
    Args:
        positions: List of (row, col) grid positions
        spacing: Spacing between tiles in pixels
        
    Returns:
        List of coordinate dictionaries with 'x', 'y', 'width', 'height'
    """
    coordinates = []
    
    for row, col in positions:
        x = col * (TILE_WIDTH + spacing) + spacing
        y = row * (TILE_HEIGHT + spacing) + spacing
        
        coordinates.append({
            'x': x,
            'y': y,
            'width': TILE_WIDTH,
            'height': TILE_HEIGHT,
            'center_x': x + TILE_WIDTH / 2,
            'center_y': y + TILE_HEIGHT / 2
        })
    
    logger.info(f"Calculated coordinates for {len(coordinates)} tiles")
    return coordinates

def compute_edge_paths(
    coordinates: List[Dict[str, float]], 
    layer_results: List[List[Tuple[str, float]]],
    curve_factor: float = CURVE_FACTOR
) -> List[Dict[str, Any]]:
    """
    Compute curved edge paths between adjacent layers for top-k tokens.
    
    Args:
        coordinates: List of tile coordinates
        layer_results: Top-k results for each layer [(token, prob), ...]
        curve_factor: Bezier curve strength (0.0 = straight, 0.5 = moderate curve)
        
    Returns:
        List of edge dictionaries with path data and styling
    """
    if len(coordinates) < 2:
        return []
    
    edges = []
    
    for layer_idx in range(len(coordinates) - 1):
        current_coord = coordinates[layer_idx]
        next_coord = coordinates[layer_idx + 1]
        
        if layer_idx >= len(layer_results):
            continue
            
        current_results = layer_results[layer_idx]
        
        # Create edges for top-k tokens (up to 3)
        for rank, (token, probability) in enumerate(current_results[:3]):
            edge = create_curved_edge(
                current_coord, next_coord, 
                token, probability, 
                layer_idx, rank, 
                curve_factor
            )
            edges.append(edge)
    
    logger.info(f"Generated {len(edges)} edge paths")
    return edges

def create_curved_edge(
    start_coord: Dict[str, float],
    end_coord: Dict[str, float],
    token: str,
    probability: float,
    layer_idx: int,
    rank: int,
    curve_factor: float
) -> Dict[str, Any]:
    """
    Create a single curved edge between two tiles.
    
    Args:
        start_coord: Starting tile coordinates
        end_coord: Ending tile coordinates
        token: Token string for this edge
        probability: Token probability
        layer_idx: Source layer index
        rank: Rank of this token (0=top, 1=second, 2=third)
        curve_factor: Curve strength
        
    Returns:
        Edge dictionary with path and styling information
    """
    # Calculate connection points (right side of start tile to left side of end tile)
    start_x = start_coord['x'] + start_coord['width']
    start_y = start_coord['center_y'] + (rank - 1) * 10  # Offset vertically by rank
    
    end_x = end_coord['x']
    end_y = end_coord['center_y'] + (rank - 1) * 10
    
    # Calculate control points for Bezier curve
    dx = end_x - start_x
    control_offset = dx * curve_factor
    
    control1_x = start_x + control_offset
    control1_y = start_y
    
    control2_x = end_x - control_offset
    control2_y = end_y
    
    # Create SVG path string for cubic Bezier curve
    path = f"M {start_x},{start_y} C {control1_x},{control1_y} {control2_x},{control2_y} {end_x},{end_y}"
    
    # Calculate opacity based on probability
    from ..utils.token_formatting import compute_edge_opacity
    opacity = compute_edge_opacity(probability, 1.0)  # Will be normalized later
    
    edge = {
        'path': path,
        'token': token,
        'probability': probability,
        'layer_idx': layer_idx,
        'rank': rank + 1,  # 1-based for display
        'opacity': opacity,
        'start_x': start_x,
        'start_y': start_y,
        'end_x': end_x,
        'end_y': end_y,
        'hover_text': f"Layer {layer_idx} → {layer_idx + 1}<br>Rank {rank + 1}: {token} ({probability:.3f})"
    }
    
    return edge

def normalize_edge_opacities(edges: List[Dict[str, Any]], min_opacity: float = 0.15) -> List[Dict[str, Any]]:
    """
    Normalize edge opacities within each layer group.
    
    Args:
        edges: List of edge dictionaries
        min_opacity: Minimum opacity for visibility
        
    Returns:
        Updated edges with normalized opacities
    """
    # Group edges by layer
    layer_groups = {}
    for edge in edges:
        layer_idx = edge['layer_idx']
        if layer_idx not in layer_groups:
            layer_groups[layer_idx] = []
        layer_groups[layer_idx].append(edge)
    
    # Normalize within each group
    for layer_idx, group_edges in layer_groups.items():
        if not group_edges:
            continue
            
        # Find max probability in this layer
        max_prob = max(edge['probability'] for edge in group_edges)
        
        # Normalize opacities
        for edge in group_edges:
            from ..utils.token_formatting import compute_edge_opacity
            edge['opacity'] = compute_edge_opacity(edge['probability'], max_prob, min_opacity)
    
    logger.info(f"Normalized opacities for {len(edges)} edges across {len(layer_groups)} layers")
    return edges

def calculate_dashboard_dimensions(coordinates: List[Dict[str, float]], padding: int = 50) -> Dict[str, int]:
    """
    Calculate total dimensions needed for the dashboard.
    
    Args:
        coordinates: List of tile coordinates
        padding: Padding around the entire layout
        
    Returns:
        Dictionary with 'width' and 'height' dimensions
    """
    if not coordinates:
        return {'width': 400, 'height': 300}
    
    max_x = max(coord['x'] + coord['width'] for coord in coordinates)
    max_y = max(coord['y'] + coord['height'] for coord in coordinates)
    
    dimensions = {
        'width': int(max_x + padding),
        'height': int(max_y + padding)
    }
    
    logger.info(f"Dashboard dimensions: {dimensions['width']} x {dimensions['height']}")
    return dimensions

def get_scroll_config(dimensions: Dict[str, int], max_width: int = 1200, max_height: int = 800) -> Dict[str, Any]:
    """
    Determine scroll configuration based on content dimensions.
    
    Args:
        dimensions: Dashboard dimensions
        max_width: Maximum viewport width
        max_height: Maximum viewport height
        
    Returns:
        Scroll configuration dictionary
    """
    config = {
        'horizontal_scroll': dimensions['width'] > max_width,
        'vertical_scroll': dimensions['height'] > max_height,
        'viewport_width': min(dimensions['width'], max_width),
        'viewport_height': min(dimensions['height'], max_height),
        'content_width': dimensions['width'],
        'content_height': dimensions['height']
    }
    
    logger.info(f"Scroll config: h={config['horizontal_scroll']}, v={config['vertical_scroll']}")
    return config
