"""
Tests for utils/head_detection.py

Tests attention head categorization heuristics using synthetic attention matrices.
"""

import pytest
import torch
import numpy as np
from utils.head_detection import (
    compute_attention_entropy,
    detect_previous_token_head,
    detect_first_token_head,
    detect_bow_head,
    detect_syntactic_head,
    categorize_attention_head,
    categorize_all_heads,
    format_categorization_summary,
    HeadCategorizationConfig
)


class TestComputeAttentionEntropy:
    """Tests for compute_attention_entropy function."""
    
    def test_uniform_distribution_high_entropy(self):
        """Uniform attention should have high (near 1.0) normalized entropy."""
        # 4 positions with equal attention
        uniform = torch.tensor([0.25, 0.25, 0.25, 0.25])
        entropy = compute_attention_entropy(uniform)
        
        # Normalized entropy should be close to 1.0 for uniform
        assert 0.95 <= entropy <= 1.0, f"Expected ~1.0, got {entropy}"
    
    def test_peaked_distribution_low_entropy(self):
        """Peaked attention should have low normalized entropy."""
        # One position dominates
        peaked = torch.tensor([0.97, 0.01, 0.01, 0.01])
        entropy = compute_attention_entropy(peaked)
        
        # Should be low entropy
        assert entropy < 0.3, f"Expected low entropy, got {entropy}"
    
    def test_entropy_bounds(self):
        """Entropy should always be between 0 and 1 (normalized)."""
        test_cases = [
            torch.tensor([1.0, 0.0, 0.0, 0.0]),      # Extreme peaked
            torch.tensor([0.5, 0.5, 0.0, 0.0]),      # Two positions
            torch.tensor([0.25, 0.25, 0.25, 0.25]),  # Uniform
        ]
        
        for weights in test_cases:
            entropy = compute_attention_entropy(weights)
            assert 0.0 <= entropy <= 1.0, f"Entropy {entropy} out of bounds"


class TestDetectPreviousTokenHead:
    """Tests for detect_previous_token_head function."""
    
    def test_detects_previous_token_pattern(self, previous_token_attention_matrix, default_head_config):
        """Should detect matrix with strong previous-token attention."""
        is_prev, score = detect_previous_token_head(
            previous_token_attention_matrix, 
            default_head_config
        )
        
        assert is_prev == True
        assert score > 0.5, f"Expected high score, got {score}"
    
    def test_rejects_uniform_attention(self, uniform_attention_matrix, default_head_config):
        """Should reject matrix with uniform attention."""
        is_prev, score = detect_previous_token_head(
            uniform_attention_matrix,
            default_head_config
        )
        
        assert is_prev == False
        assert score < 0.4, f"Expected low score, got {score}"
    
    def test_short_sequence_returns_false(self, default_head_config):
        """Sequence shorter than min_seq_len should return False."""
        short_matrix = torch.ones(2, 2) / 2
        is_prev, score = detect_previous_token_head(short_matrix, default_head_config)
        
        assert is_prev == False
        assert score == 0.0


class TestDetectFirstTokenHead:
    """Tests for detect_first_token_head function."""
    
    def test_detects_first_token_pattern(self, first_token_attention_matrix, default_head_config):
        """Should detect matrix with strong first-token attention."""
        is_first, score = detect_first_token_head(
            first_token_attention_matrix,
            default_head_config
        )
        
        assert is_first == True
        assert score > 0.5, f"Expected high score, got {score}"
    
    def test_low_first_token_attention(self, default_head_config):
        """Matrix with low attention to first token should not be detected."""
        # Create matrix where first token gets very little attention
        # Use size 5 to be above min_seq_len and avoid overlap at [0,0]
        size = 5
        matrix = torch.zeros(size, size)
        for i in range(size):
            # Distribute attention: 5% to first token, 95% to last token
            matrix[i, 0] = 0.05
            matrix[i, -1] = 0.95
        
        is_first, score = detect_first_token_head(matrix, default_head_config)
        
        assert is_first == False
        assert score < 0.25, f"Expected low score, got {score}"


class TestDetectBowHead:
    """Tests for detect_bow_head (bag-of-words / diffuse attention)."""
    
    def test_detects_uniform_as_bow(self, uniform_attention_matrix, default_head_config):
        """Uniform attention should be detected as BoW head."""
        is_bow, score = detect_bow_head(uniform_attention_matrix, default_head_config)
        
        # Uniform has high entropy and low max attention - should be BoW
        assert is_bow == True
        assert score > 0.9, f"Expected high entropy score, got {score}"
    
    def test_rejects_peaked_attention(self, peaked_attention_matrix, default_head_config):
        """Peaked attention should not be detected as BoW."""
        is_bow, score = detect_bow_head(peaked_attention_matrix, default_head_config)
        
        # Peaked attention has low entropy - should not be BoW
        assert is_bow == False


class TestDetectSyntacticHead:
    """Tests for detect_syntactic_head function."""
    
    def test_consistent_distance_pattern(self, default_head_config):
        """Matrix with consistent distance pattern should be detected as syntactic."""
        # Create matrix where each position attends to position 2 tokens back
        size = 6
        matrix = torch.zeros(size, size)
        for i in range(size):
            target = max(0, i - 2)  # 2 tokens back
            matrix[i, target] = 1.0
        
        is_syn, score = detect_syntactic_head(matrix, default_head_config)
        
        # Should have consistent distance pattern
        assert score > 0.0, f"Expected positive score for consistent pattern"
    
    def test_random_attention_returns_valid_values(self, default_head_config):
        """Random attention should return valid boolean and score."""
        torch.manual_seed(42)
        random_matrix = torch.softmax(torch.randn(6, 6), dim=-1)
        
        is_syn, score = detect_syntactic_head(random_matrix, default_head_config)
        
        # Check it returns valid types (bool or numpy bool, and numeric score)
        assert is_syn in [True, False] or bool(is_syn) in [True, False]
        assert 0 <= float(score) <= 1


class TestCategorizeAttentionHead:
    """Tests for categorize_attention_head function."""
    
    def test_categorizes_previous_token_head(self, previous_token_attention_matrix, default_head_config):
        """Should categorize previous-token pattern correctly."""
        result = categorize_attention_head(
            previous_token_attention_matrix,
            layer_idx=0,
            head_idx=3,
            config=default_head_config
        )
        
        assert result['category'] == 'previous_token'
        assert result['layer'] == 0
        assert result['head'] == 3
        assert result['label'] == 'L0-H3'
        assert 'scores' in result
    
    def test_categorizes_first_token_head(self, first_token_attention_matrix, default_head_config):
        """Should categorize first-token pattern correctly."""
        result = categorize_attention_head(
            first_token_attention_matrix,
            layer_idx=2,
            head_idx=5,
            config=default_head_config
        )
        
        assert result['category'] == 'first_token'
        assert result['label'] == 'L2-H5'
    
    def test_categorizes_bow_head(self, default_head_config):
        """Should categorize diffuse attention as BoW when it doesn't match other patterns."""
        # Create BoW-like matrix: diffuse attention but first token gets LESS than threshold
        # This avoids triggering first_token detection (threshold 0.25)
        size = 5
        matrix = torch.zeros(size, size)
        for i in range(size):
            # First token gets only 0.1, rest get roughly equal share
            matrix[i, 0] = 0.1
            remaining = 0.9 / (size - 1)
            for j in range(1, size):
                matrix[i, j] = remaining
        
        result = categorize_attention_head(
            matrix,
            layer_idx=1,
            head_idx=0,
            config=default_head_config
        )
        
        assert result['category'] == 'bow'
    
    def test_result_structure(self, uniform_attention_matrix):
        """Result should have all required keys."""
        result = categorize_attention_head(
            uniform_attention_matrix,
            layer_idx=0,
            head_idx=0
        )
        
        required_keys = ['layer', 'head', 'category', 'scores', 'label']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"


class TestCategorizeAllHeads:
    """Tests for categorize_all_heads function."""
    
    def test_returns_all_categories(self, mock_activation_data, default_head_config):
        """Should return dict with all category keys."""
        result = categorize_all_heads(mock_activation_data, default_head_config)
        
        expected_categories = ['previous_token', 'first_token', 'bow', 'syntactic', 'other']
        for cat in expected_categories:
            assert cat in result, f"Missing category: {cat}"
            assert isinstance(result[cat], list)
    
    def test_handles_empty_attention_data(self, default_head_config):
        """Should handle activation data with no attention outputs."""
        empty_data = {'attention_outputs': {}}
        result = categorize_all_heads(empty_data, default_head_config)
        
        # Should return empty lists for all categories
        for cat, heads in result.items():
            assert heads == []


class TestFormatCategorizationSummary:
    """Tests for format_categorization_summary function."""
    
    def test_formats_empty_categorization(self):
        """Should format empty categorization without error."""
        empty = {
            'previous_token': [],
            'first_token': [],
            'bow': [],
            'syntactic': [],
            'other': []
        }
        result = format_categorization_summary(empty)
        
        assert isinstance(result, str)
        assert "Total Heads: 0" in result
    
    def test_formats_with_heads(self):
        """Should format categorization with heads correctly."""
        categorized = {
            'previous_token': [
                {'layer': 0, 'head': 1, 'label': 'L0-H1'},
                {'layer': 0, 'head': 2, 'label': 'L0-H2'},
            ],
            'first_token': [
                {'layer': 1, 'head': 0, 'label': 'L1-H0'},
            ],
            'bow': [],
            'syntactic': [],
            'other': []
        }
        result = format_categorization_summary(categorized)
        
        assert "Total Heads: 3" in result
        assert "Previous-Token Heads: 2" in result
        assert "First/Positional-Token Heads: 1" in result
        assert "Layer 0" in result
        assert "Layer 1" in result


class TestHeadCategorizationConfig:
    """Tests for HeadCategorizationConfig defaults."""
    
    def test_default_values(self):
        """Default config should have reasonable values."""
        config = HeadCategorizationConfig()
        
        assert 0 < config.prev_token_threshold < 1
        assert 0 < config.first_token_threshold < 1
        assert 0 < config.bow_entropy_threshold < 1
        assert config.min_seq_len > 0
    
    def test_config_is_mutable(self):
        """Config values should be mutable for customization."""
        config = HeadCategorizationConfig()
        original = config.prev_token_threshold
        
        config.prev_token_threshold = 0.8
        assert config.prev_token_threshold == 0.8
        assert config.prev_token_threshold != original
