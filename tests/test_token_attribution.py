"""
Tests for utils/token_attribution.py

Tests the visualization data formatting function (pure logic).
The gradient computation functions require models and are not tested here.
"""

import pytest
from utils.token_attribution import create_attribution_visualization_data


class TestCreateAttributionVisualizationData:
    """Tests for create_attribution_visualization_data function."""
    
    def test_returns_correct_structure(self, mock_attribution_result):
        """Should return list of dicts with required keys."""
        result = create_attribution_visualization_data(mock_attribution_result)
        
        assert isinstance(result, list)
        assert len(result) == 3  # 3 tokens in mock data
        
        required_keys = ['token', 'index', 'attribution', 'normalized', 'color', 'text_color']
        for item in result:
            for key in required_keys:
                assert key in item, f"Missing key: {key}"
    
    def test_preserves_token_order(self, mock_attribution_result):
        """Tokens should be in same order as input."""
        result = create_attribution_visualization_data(mock_attribution_result)
        
        assert result[0]['token'] == 'Hello'
        assert result[1]['token'] == ' world'
        assert result[2]['token'] == '!'
        
        assert result[0]['index'] == 0
        assert result[1]['index'] == 1
        assert result[2]['index'] == 2
    
    def test_preserves_attribution_values(self, mock_attribution_result):
        """Raw attribution values should be preserved."""
        result = create_attribution_visualization_data(mock_attribution_result)
        
        assert result[0]['attribution'] == 0.5
        assert result[1]['attribution'] == 1.0
        assert result[2]['attribution'] == 0.2
    
    def test_color_format(self, mock_attribution_result):
        """Colors should be valid RGB format."""
        result = create_attribution_visualization_data(mock_attribution_result)
        
        for item in result:
            color = item['color']
            assert color.startswith('rgb(')
            assert color.endswith(')')
            # Extract RGB values
            rgb_str = color[4:-1]
            r, g, b = [int(x) for x in rgb_str.split(',')]
            assert 0 <= r <= 255
            assert 0 <= g <= 255
            assert 0 <= b <= 255
    
    def test_text_color_contrast(self, mock_attribution_result):
        """Text color should be black or white for contrast."""
        result = create_attribution_visualization_data(mock_attribution_result)
        
        for item in result:
            assert item['text_color'] in ['#000000', '#ffffff']
    
    def test_high_attribution_gets_color(self):
        """High attribution should result in colored background."""
        data = {
            'tokens': ['high'],
            'token_ids': [1],
            'attributions': [1.0],  # Maximum positive attribution
            'normalized_attributions': [1.0],
            'target_token': 'x',
            'target_token_id': 100
        }
        result = create_attribution_visualization_data(data)
        
        # High positive attribution should have red-ish color (r=255)
        color = result[0]['color']
        rgb_str = color[4:-1]
        r, g, b = [int(x) for x in rgb_str.split(',')]
        
        # Red should be at max, green/blue should be reduced
        assert r == 255
        assert g < 255  # Reduced for visibility
        assert b < 255
    
    def test_handles_zero_attributions(self):
        """Zero attributions should produce neutral colors."""
        data = {
            'tokens': ['zero'],
            'token_ids': [1],
            'attributions': [0.0],
            'normalized_attributions': [0.0],
            'target_token': 'x',
            'target_token_id': 100
        }
        result = create_attribution_visualization_data(data)
        
        # Zero normalized attribution should give white-ish color
        color = result[0]['color']
        rgb_str = color[4:-1]
        r, g, b = [int(x) for x in rgb_str.split(',')]
        
        # All components should be high (near white)
        assert r == 255
        assert g == 255
        assert b == 255
    
    def test_handles_negative_attributions(self):
        """Negative attributions should get blue-ish color."""
        data = {
            'tokens': ['negative'],
            'token_ids': [1],
            'attributions': [-1.0],  # Negative attribution
            'normalized_attributions': [1.0],  # Abs normalized
            'target_token': 'x',
            'target_token_id': 100
        }
        result = create_attribution_visualization_data(data)
        
        # Negative attribution should have blue-ish color
        color = result[0]['color']
        rgb_str = color[4:-1]
        r, g, b = [int(x) for x in rgb_str.split(',')]
        
        # Blue should be at max, red/green should be reduced
        assert b == 255
        assert r < 255
        assert g < 255


class TestAttributionVisualizationEdgeCases:
    """Edge case tests for create_attribution_visualization_data."""
    
    def test_handles_single_token(self):
        """Should handle single token input."""
        data = {
            'tokens': ['only'],
            'token_ids': [1],
            'attributions': [0.5],
            'normalized_attributions': [1.0],  # Normalized to max
            'target_token': 'x',
            'target_token_id': 100
        }
        result = create_attribution_visualization_data(data)
        
        assert len(result) == 1
        assert result[0]['token'] == 'only'
    
    def test_handles_empty_input(self):
        """Should handle empty token list."""
        data = {
            'tokens': [],
            'token_ids': [],
            'attributions': [],
            'normalized_attributions': [],
            'target_token': 'x',
            'target_token_id': 100
        }
        result = create_attribution_visualization_data(data)
        
        assert result == []
    
    def test_handles_special_characters_in_tokens(self):
        """Should handle tokens with special characters."""
        data = {
            'tokens': ['<s>', '</s>', '\n', '  '],
            'token_ids': [1, 2, 3, 4],
            'attributions': [0.1, 0.2, 0.3, 0.4],
            'normalized_attributions': [0.25, 0.5, 0.75, 1.0],
            'target_token': 'x',
            'target_token_id': 100
        }
        result = create_attribution_visualization_data(data)
        
        assert len(result) == 4
        assert result[0]['token'] == '<s>'
        assert result[2]['token'] == '\n'
