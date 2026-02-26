"""
Tests for utils/head_detection.py

Tests the offline JSON + runtime verification head categorization system.
"""

import pytest
import torch
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch, mock_open
from utils.head_detection import (
    load_head_categories,
    verify_head_activation,
    get_active_head_summary,
    clear_category_cache,
    _compute_attention_entropy,
    _find_repeated_tokens,
)


# =============================================================================
# Sample JSON data for mocking
# =============================================================================

SAMPLE_JSON = {
    "test-model": {
        "model_name": "test-model",
        "num_layers": 2,
        "num_heads": 4,
        "analysis_date": "2026-02-26",
        "categories": {
            "previous_token": {
                "display_name": "Previous Token",
                "description": "Attends to the previous token",
                "educational_text": "Looks at the word before.",
                "icon": "arrow-left",
                "requires_repetition": False,
                "top_heads": [
                    {"layer": 0, "head": 1, "score": 0.85},
                    {"layer": 1, "head": 2, "score": 0.72}
                ]
            },
            "induction": {
                "display_name": "Induction",
                "description": "Pattern matching",
                "educational_text": "Finds repeated patterns.",
                "icon": "repeat",
                "requires_repetition": True,
                "suggested_prompt": "Try repeating words.",
                "top_heads": [
                    {"layer": 1, "head": 0, "score": 0.90}
                ]
            },
            "duplicate_token": {
                "display_name": "Duplicate Token",
                "description": "Finds duplicates",
                "educational_text": "Spots repeated words.",
                "icon": "clone",
                "requires_repetition": True,
                "suggested_prompt": "Try typing the same word twice.",
                "top_heads": [
                    {"layer": 0, "head": 3, "score": 0.78}
                ]
            },
            "positional": {
                "display_name": "Positional",
                "description": "First token focus",
                "educational_text": "Anchors to position 0.",
                "icon": "map-pin",
                "requires_repetition": False,
                "top_heads": [
                    {"layer": 0, "head": 0, "score": 0.88}
                ]
            },
            "diffuse": {
                "display_name": "Diffuse",
                "description": "Spread attention",
                "educational_text": "Even distribution.",
                "icon": "expand-arrows-alt",
                "requires_repetition": False,
                "top_heads": [
                    {"layer": 1, "head": 3, "score": 0.80}
                ]
            }
        },
        "all_scores": {}
    }
}


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the category cache before each test."""
    clear_category_cache()
    yield
    clear_category_cache()


# =============================================================================
# Tests for _compute_attention_entropy
# =============================================================================

class TestComputeAttentionEntropy:
    """Tests for _compute_attention_entropy helper."""

    def test_uniform_distribution_high_entropy(self):
        """Uniform attention should have entropy near 1.0."""
        weights = torch.ones(8) / 8
        entropy = _compute_attention_entropy(weights)
        assert entropy > 0.95

    def test_peaked_distribution_low_entropy(self):
        """Peaked attention should have low entropy."""
        weights = torch.zeros(8)
        weights[0] = 0.98
        weights[1:] = 0.02 / 7
        entropy = _compute_attention_entropy(weights)
        assert entropy < 0.3

    def test_entropy_in_range(self):
        """Entropy should always be between 0 and 1."""
        for _ in range(10):
            weights = torch.softmax(torch.randn(6), dim=0)
            entropy = _compute_attention_entropy(weights)
            assert 0.0 <= entropy <= 1.0


# =============================================================================
# Tests for _find_repeated_tokens
# =============================================================================

class TestFindRepeatedTokens:
    """Tests for _find_repeated_tokens helper."""

    def test_no_repeats(self):
        """No repetition returns empty dict."""
        assert _find_repeated_tokens([1, 2, 3, 4]) == {}

    def test_simple_repeat(self):
        """Repeated token returns positions."""
        result = _find_repeated_tokens([10, 20, 10, 30])
        assert 10 in result
        assert result[10] == [0, 2]
        assert 20 not in result

    def test_multiple_repeats(self):
        """Multiple repeated tokens tracked."""
        result = _find_repeated_tokens([5, 6, 5, 6, 7])
        assert 5 in result and 6 in result
        assert 7 not in result

    def test_empty_input(self):
        assert _find_repeated_tokens([]) == {}


# =============================================================================
# Tests for load_head_categories
# =============================================================================

class TestLoadHeadCategories:
    """Tests for load_head_categories function."""

    def test_loads_from_json(self, tmp_path):
        """Should load model data from JSON file."""
        json_file = tmp_path / "head_categories.json"
        json_file.write_text(json.dumps(SAMPLE_JSON))

        with patch('utils.head_detection._JSON_PATH', json_file):
            result = load_head_categories("test-model")
        
        assert result is not None
        assert result["model_name"] == "test-model"
        assert "previous_token" in result["categories"]

    def test_returns_none_for_unknown_model(self, tmp_path):
        """Should return None when model not in JSON."""
        json_file = tmp_path / "head_categories.json"
        json_file.write_text(json.dumps(SAMPLE_JSON))

        with patch('utils.head_detection._JSON_PATH', json_file):
            result = load_head_categories("nonexistent-model")
        
        assert result is None

    def test_returns_none_when_no_file(self, tmp_path):
        """Should return None when JSON file doesn't exist."""
        with patch('utils.head_detection._JSON_PATH', tmp_path / "missing.json"):
            result = load_head_categories("test-model")
        
        assert result is None

    def test_caches_results(self, tmp_path):
        """Should cache loaded data."""
        json_file = tmp_path / "head_categories.json"
        json_file.write_text(json.dumps(SAMPLE_JSON))

        with patch('utils.head_detection._JSON_PATH', json_file):
            result1 = load_head_categories("test-model")
            # Delete file to prove cache is used
            json_file.unlink()
            result2 = load_head_categories("test-model")
        
        assert result1 is result2

    def test_short_name_alias(self, tmp_path):
        """Should find model by short name (after /)."""
        data = {"my-model": {"model_name": "my-model", "categories": {}}}
        json_file = tmp_path / "head_categories.json"
        json_file.write_text(json.dumps(data))

        with patch('utils.head_detection._JSON_PATH', json_file):
            result = load_head_categories("org/my-model")
        
        assert result is not None


# =============================================================================
# Tests for verify_head_activation
# =============================================================================

class TestVerifyHeadActivation:
    """Tests for verify_head_activation function."""

    def test_previous_token_strong(self):
        """Strong previous-token pattern should score high."""
        size = 6
        matrix = torch.zeros(size, size)
        for i in range(1, size):
            matrix[i, i - 1] = 0.8
            matrix[i, i] = 0.2
        matrix[0, 0] = 1.0

        score = verify_head_activation(matrix, [1, 2, 3, 4, 5, 6], "previous_token")
        assert score > 0.6

    def test_previous_token_weak(self):
        """Uniform attention should have low previous-token score."""
        size = 6
        matrix = torch.ones(size, size) / size
        score = verify_head_activation(matrix, [1, 2, 3, 4, 5, 6], "previous_token")
        assert score < 0.3

    def test_induction_with_repetition(self):
        """Induction pattern should score > 0 when repeated tokens are present."""
        # Tokens: [A, B, C, A, ?] — head should attend to B (position 1) from position 3
        size = 5
        matrix = torch.ones(size, size) / size  # Baseline uniform
        matrix[3, 1] = 0.7  # Position 3 (second A) attends to position 1 (B after first A)
        
        token_ids = [10, 20, 30, 10, 40]  # Token 10 repeats
        score = verify_head_activation(matrix, token_ids, "induction")
        assert score > 0.3

    def test_induction_no_repetition(self):
        """Induction should return 0.0 when no tokens repeat."""
        matrix = torch.ones(4, 4) / 4
        score = verify_head_activation(matrix, [1, 2, 3, 4], "induction")
        assert score == 0.0

    def test_duplicate_token_with_repeats(self):
        """Duplicate-token head should score > 0 when later positions attend to earlier same token."""
        size = 5
        matrix = torch.ones(size, size) / size
        matrix[3, 0] = 0.6  # Position 3 (second occurrence of token 10) attends to position 0

        token_ids = [10, 20, 30, 10, 40]
        score = verify_head_activation(matrix, token_ids, "duplicate_token")
        assert score > 0.3

    def test_duplicate_token_no_repeats(self):
        """Should return 0.0 when no duplicates."""
        matrix = torch.ones(4, 4) / 4
        score = verify_head_activation(matrix, [1, 2, 3, 4], "duplicate_token")
        assert score == 0.0

    def test_positional_strong(self):
        """Strong first-token attention should score high."""
        size = 6
        matrix = torch.zeros(size, size)
        for i in range(size):
            matrix[i, 0] = 0.7
            matrix[i, i] = 0.3
        
        score = verify_head_activation(matrix, [1, 2, 3, 4, 5, 6], "positional")
        assert score > 0.5

    def test_diffuse_uniform(self):
        """Uniform attention should have high diffuse score."""
        size = 8
        matrix = torch.ones(size, size) / size
        score = verify_head_activation(matrix, list(range(size)), "diffuse")
        assert score > 0.8

    def test_diffuse_peaked(self):
        """Peaked attention should have low diffuse score."""
        size = 8
        matrix = torch.zeros(size, size)
        matrix[:, 0] = 1.0
        score = verify_head_activation(matrix, list(range(size)), "diffuse")
        assert score < 0.3

    def test_unknown_category(self):
        """Unknown category should return 0.0."""
        matrix = torch.ones(4, 4) / 4
        assert verify_head_activation(matrix, [1, 2, 3, 4], "nonexistent") == 0.0

    def test_short_sequence(self):
        """Very short sequence should return 0.0."""
        matrix = torch.ones(1, 1)
        assert verify_head_activation(matrix, [1], "previous_token") == 0.0


# =============================================================================
# Tests for get_active_head_summary
# =============================================================================

class TestGetActiveHeadSummary:
    """Tests for get_active_head_summary function."""

    def _make_activation_data(self, token_ids, num_layers=2, num_heads=4, seq_len=None):
        """Helper: create mock activation_data with given token_ids."""
        if seq_len is None:
            seq_len = len(token_ids)
        
        attention_outputs = {}
        for layer in range(num_layers):
            # Create uniform attention [1, num_heads, seq_len, seq_len]
            attn = torch.ones(1, num_heads, seq_len, seq_len) / seq_len
            attention_outputs[f'model.layers.{layer}.self_attn'] = {
                'output': [
                    [[0.1] * seq_len],  # hidden states (unused)
                    attn.tolist()
                ]
            }
        
        return {
            'model': 'test-model',
            'input_ids': [token_ids],
            'attention_outputs': attention_outputs,
        }

    def test_returns_none_for_unknown_model(self, tmp_path):
        """Should return None when model not in JSON."""
        json_file = tmp_path / "head_categories.json"
        json_file.write_text(json.dumps(SAMPLE_JSON))

        with patch('utils.head_detection._JSON_PATH', json_file):
            data = self._make_activation_data([1, 2, 3, 4])
            result = get_active_head_summary(data, "unknown-model")
        
        assert result is None

    def test_returns_categories_structure(self, tmp_path):
        """Should return proper structure with categories."""
        json_file = tmp_path / "head_categories.json"
        json_file.write_text(json.dumps(SAMPLE_JSON))

        with patch('utils.head_detection._JSON_PATH', json_file):
            data = self._make_activation_data([1, 2, 3, 4])
            result = get_active_head_summary(data, "test-model")
        
        assert result is not None
        assert result["model_available"] is True
        assert "categories" in result
        assert "previous_token" in result["categories"]
        assert "induction" in result["categories"]

    def test_heads_have_activation_scores(self, tmp_path):
        """Each head should have an activation_score."""
        json_file = tmp_path / "head_categories.json"
        json_file.write_text(json.dumps(SAMPLE_JSON))

        with patch('utils.head_detection._JSON_PATH', json_file):
            data = self._make_activation_data([1, 2, 3, 4])
            result = get_active_head_summary(data, "test-model")

        for cat_key, cat_data in result["categories"].items():
            for head in cat_data.get("heads", []):
                assert "activation_score" in head
                assert "is_active" in head
                assert "label" in head

    def test_induction_grayed_when_no_repeats(self, tmp_path):
        """Induction should be non-applicable when no repeated tokens."""
        json_file = tmp_path / "head_categories.json"
        json_file.write_text(json.dumps(SAMPLE_JSON))

        with patch('utils.head_detection._JSON_PATH', json_file):
            data = self._make_activation_data([1, 2, 3, 4])  # No repeats
            result = get_active_head_summary(data, "test-model")

        induction = result["categories"]["induction"]
        assert induction["is_applicable"] is False
        assert all(h["activation_score"] == 0.0 for h in induction["heads"])

    def test_induction_active_with_repeats(self, tmp_path):
        """Induction should be applicable when tokens repeat."""
        json_file = tmp_path / "head_categories.json"
        json_file.write_text(json.dumps(SAMPLE_JSON))

        with patch('utils.head_detection._JSON_PATH', json_file):
            data = self._make_activation_data([10, 20, 10, 30])  # Token 10 repeats
            result = get_active_head_summary(data, "test-model")

        induction = result["categories"]["induction"]
        assert induction["is_applicable"] is True

    def test_suggested_prompt_included(self, tmp_path):
        """Suggested prompt should appear for repetition-dependent categories."""
        json_file = tmp_path / "head_categories.json"
        json_file.write_text(json.dumps(SAMPLE_JSON))

        with patch('utils.head_detection._JSON_PATH', json_file):
            data = self._make_activation_data([1, 2, 3, 4])
            result = get_active_head_summary(data, "test-model")

        assert result["categories"]["induction"]["suggested_prompt"] is not None
        assert result["categories"]["duplicate_token"]["suggested_prompt"] is not None

    def test_other_category_always_present(self, tmp_path):
        """Other/Unclassified category should always be in the result."""
        json_file = tmp_path / "head_categories.json"
        json_file.write_text(json.dumps(SAMPLE_JSON))

        with patch('utils.head_detection._JSON_PATH', json_file):
            data = self._make_activation_data([1, 2, 3, 4])
            result = get_active_head_summary(data, "test-model")

        assert "other" in result["categories"]
