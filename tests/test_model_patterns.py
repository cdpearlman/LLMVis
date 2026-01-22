"""
Tests for utils/model_patterns.py

Tests pure logic functions that don't require model loading:
- merge_token_probabilities
- safe_to_serializable
"""

import pytest
import torch
import numpy as np
from utils.model_patterns import merge_token_probabilities, safe_to_serializable


class TestMergeTokenProbabilities:
    """Tests for merge_token_probabilities function."""
    
    def test_merges_tokens_with_leading_space(self):
        """Tokens with and without leading space should be merged."""
        token_probs = [
            (" cat", 0.15),
            ("cat", 0.05),
            (" dog", 0.10),
        ]
        result = merge_token_probabilities(token_probs)
        
        # Convert to dict for easier checking
        result_dict = dict(result)
        
        assert "cat" in result_dict
        assert abs(result_dict["cat"] - 0.20) < 1e-6  # 0.15 + 0.05
        assert "dog" in result_dict
        assert abs(result_dict["dog"] - 0.10) < 1e-6
    
    def test_sorts_by_probability_descending(self):
        """Results should be sorted by probability (highest first)."""
        token_probs = [
            ("low", 0.01),
            ("high", 0.50),
            ("medium", 0.20),
        ]
        result = merge_token_probabilities(token_probs)
        
        # Check order: high, medium, low
        assert result[0][0] == "high"
        assert result[1][0] == "medium"
        assert result[2][0] == "low"
    
    def test_handles_empty_input(self):
        """Empty input should return empty list."""
        result = merge_token_probabilities([])
        assert result == []
    
    def test_handles_single_token(self):
        """Single token should be returned as-is (stripped)."""
        result = merge_token_probabilities([(" hello", 0.5)])
        
        assert len(result) == 1
        assert result[0][0] == "hello"
        assert result[0][1] == 0.5
    
    def test_strips_multiple_spaces(self):
        """Multiple leading spaces should all be stripped."""
        token_probs = [
            ("  word", 0.3),  # Two spaces
            (" word", 0.2),   # One space
            ("word", 0.1),    # No space
        ]
        result = merge_token_probabilities(token_probs)
        
        result_dict = dict(result)
        assert "word" in result_dict
        assert abs(result_dict["word"] - 0.6) < 1e-6  # All merged


class TestSafeToSerializable:
    """Tests for safe_to_serializable function."""
    
    def test_converts_tensor_to_list(self):
        """PyTorch tensor should be converted to Python list."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = safe_to_serializable(tensor)
        
        assert isinstance(result, list)
        assert result == [1.0, 2.0, 3.0]
    
    def test_converts_nested_tensor(self):
        """2D tensor should become nested list."""
        tensor = torch.tensor([[1, 2], [3, 4]])
        result = safe_to_serializable(tensor)
        
        assert isinstance(result, list)
        assert result == [[1, 2], [3, 4]]
    
    def test_converts_list_of_tensors(self):
        """List containing tensors should have tensors converted."""
        data = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        result = safe_to_serializable(data)
        
        assert result == [[1, 2], [3, 4]]
    
    def test_converts_dict_with_tensor_values(self):
        """Dict with tensor values should have values converted."""
        data = {
            "a": torch.tensor([1.0, 2.0]),
            "b": "string_value",
            "c": 42
        }
        result = safe_to_serializable(data)
        
        assert result["a"] == [1.0, 2.0]
        assert result["b"] == "string_value"
        assert result["c"] == 42
    
    def test_handles_tuple_input(self):
        """Tuple with tensors should be converted to list."""
        data = (torch.tensor([1]), torch.tensor([2]))
        result = safe_to_serializable(data)
        
        assert isinstance(result, list)
        assert result == [[1], [2]]
    
    def test_passes_through_primitives(self):
        """Primitive types should pass through unchanged."""
        assert safe_to_serializable(42) == 42
        assert safe_to_serializable(3.14) == 3.14
        assert safe_to_serializable("hello") == "hello"
        assert safe_to_serializable(None) is None
        assert safe_to_serializable(True) is True
    
    def test_handles_deeply_nested_structure(self):
        """Should handle deeply nested structures with tensors."""
        data = {
            "level1": {
                "level2": {
                    "tensor": torch.tensor([1, 2, 3])
                }
            },
            "list": [torch.tensor([4, 5])]
        }
        result = safe_to_serializable(data)
        
        assert result["level1"]["level2"]["tensor"] == [1, 2, 3]
        assert result["list"] == [[4, 5]]
    
    def test_handles_empty_containers(self):
        """Empty lists, dicts, tuples should remain empty."""
        assert safe_to_serializable([]) == []
        assert safe_to_serializable({}) == {}
        assert safe_to_serializable(()) == []  # Tuple becomes list


class TestSafeToSerializableEdgeCases:
    """Edge case tests for safe_to_serializable."""
    
    def test_handles_scalar_tensor(self):
        """Scalar tensor should become a Python scalar."""
        scalar = torch.tensor(42.0)
        result = safe_to_serializable(scalar)
        
        # Scalar tensor.tolist() returns a Python number
        assert result == 42.0
    
    def test_handles_integer_tensor(self):
        """Integer tensor should be converted correctly."""
        tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
        result = safe_to_serializable(tensor)
        
        assert result == [1, 2, 3]
        assert all(isinstance(x, int) for x in result)
    
    def test_handles_mixed_list(self):
        """List with mixed tensor and non-tensor items should work."""
        data = [torch.tensor([1]), "string", 42, {"key": torch.tensor([2])}]
        result = safe_to_serializable(data)
        
        assert result[0] == [1]
        assert result[1] == "string"
        assert result[2] == 42
        assert result[3] == {"key": [2]}
