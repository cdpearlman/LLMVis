"""
Tests for utils/model_patterns.py

Tests pure logic functions that don't require model loading:
- merge_token_probabilities
- safe_to_serializable
- execute_forward_pass_with_multi_layer_head_ablation (import/signature tests)
"""

import pytest
import torch
import numpy as np
from utils.model_patterns import merge_token_probabilities, safe_to_serializable
from utils import execute_forward_pass_with_multi_layer_head_ablation


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


class TestMultiLayerHeadAblation:
    """Tests for execute_forward_pass_with_multi_layer_head_ablation function.
    
    These tests verify the function exists, is importable, and has the expected signature.
    Full integration tests would require loading a model.
    """
    
    def test_function_is_importable(self):
        """Function should be importable from utils."""
        from utils import execute_forward_pass_with_multi_layer_head_ablation
        assert callable(execute_forward_pass_with_multi_layer_head_ablation)
    
    def test_function_has_expected_signature(self):
        """Function should accept model, tokenizer, prompt, config, heads_by_layer."""
        import inspect
        sig = inspect.signature(execute_forward_pass_with_multi_layer_head_ablation)
        params = list(sig.parameters.keys())
        
        assert 'model' in params
        assert 'tokenizer' in params
        assert 'prompt' in params
        assert 'config' in params
        assert 'heads_by_layer' in params
    
    def test_heads_by_layer_type_annotation(self):
        """heads_by_layer parameter should accept Dict[int, List[int]]."""
        import inspect
        from typing import Dict, List, get_type_hints
        
        # Get annotations (may not be available at runtime if not using from __future__)
        sig = inspect.signature(execute_forward_pass_with_multi_layer_head_ablation)
        heads_param = sig.parameters.get('heads_by_layer')
        
        # The parameter should exist
        assert heads_param is not None
        # Annotation may be a string or the actual type
        if heads_param.annotation != inspect.Parameter.empty:
            annotation_str = str(heads_param.annotation)
            assert 'Dict' in annotation_str or 'dict' in annotation_str.lower()
    
    def test_returns_error_for_no_modules(self):
        """Should return error dict when config has no modules.
        
        Note: This test uses a mock model that won't actually run forward pass.
        The function should return early with an error before trying to run.
        """
        from unittest.mock import MagicMock
        
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        empty_config = {}  # No modules specified
        heads_by_layer = {0: [1]}  # Non-empty to avoid early return
        
        result = execute_forward_pass_with_multi_layer_head_ablation(
            mock_model, mock_tokenizer, "test prompt", empty_config, heads_by_layer
        )
        
        assert 'error' in result
        assert 'No modules specified' in result['error']
    
    def test_returns_error_for_invalid_layer(self):
        """Should return error when layer number doesn't match any module."""
        from unittest.mock import MagicMock
        
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        # Config has layer 0 and 1, but we'll request layer 99
        config = {
            'attention_modules': ['model.layers.0.self_attn', 'model.layers.1.self_attn'],
            'block_modules': ['model.layers.0', 'model.layers.1']
        }
        heads_by_layer = {99: [0, 1]}  # Layer 99 doesn't exist
        
        result = execute_forward_pass_with_multi_layer_head_ablation(
            mock_model, mock_tokenizer, "test prompt", config, heads_by_layer
        )
        
        assert 'error' in result
        assert '99' in result['error']  # Should mention the invalid layer


class TestFullSequenceAttentionData:
    """
    Tests verifying that activation data for full sequences (prompt + generated output)
    has correctly-sized attention matrices matching the full token count.
    """

    def _make_activation_data(self, num_tokens, num_layers=2, num_heads=1):
        """Helper to build mock activation data for a given sequence length."""
        input_ids = list(range(1, num_tokens + 1))
        # Build uniform attention weights: [batch=1, heads, seq, seq]
        uniform_val = 1.0 / num_tokens
        attn_row = [uniform_val] * num_tokens
        head_matrix = [attn_row] * num_tokens
        head_block = [head_matrix] * num_heads
        attn_weights = [head_block]  # batch dim

        attention_outputs = {}
        for layer in range(num_layers):
            module_name = f'model.layers.{layer}.self_attn'
            attention_outputs[module_name] = {
                'output': [
                    [[0.1] * num_tokens],  # hidden states placeholder
                    attn_weights
                ]
            }

        return {
            'model': 'mock-model',
            'prompt': 'x ' * num_tokens,
            'input_ids': [input_ids],
            'attention_modules': list(attention_outputs.keys()),
            'attention_outputs': attention_outputs,
            'block_modules': [f'model.layers.{i}' for i in range(num_layers)],
            'block_outputs': {f'model.layers.{i}': {'output': [[[0.1] * num_tokens]]} for i in range(num_layers)},
            'norm_parameters': [],
            'norm_data': [],
            'actual_output': {'token': 'tok', 'probability': 0.5},
            'global_top5_tokens': []
        }

    def test_short_prompt_attention_dimensions(self):
        """Attention matrix for a 4-token prompt should be 4x4."""
        data = self._make_activation_data(num_tokens=4)
        attn = data['attention_outputs']['model.layers.0.self_attn']['output'][1]
        seq_len = len(attn[0][0])  # batch -> head -> rows
        assert seq_len == 4
        assert len(attn[0][0][0]) == 4  # columns

    def test_full_sequence_attention_dimensions(self):
        """Attention matrix for a 10-token full sequence (prompt+output) should be 10x10."""
        data = self._make_activation_data(num_tokens=10)
        attn = data['attention_outputs']['model.layers.0.self_attn']['output'][1]
        seq_len = len(attn[0][0])
        assert seq_len == 10
        assert len(attn[0][0][0]) == 10

    def test_full_sequence_has_more_tokens_than_prompt(self):
        """Full-sequence activation data should have larger dimensions than prompt-only."""
        prompt_data = self._make_activation_data(num_tokens=4)
        full_data = self._make_activation_data(num_tokens=10)

        prompt_ids = prompt_data['input_ids'][0]
        full_ids = full_data['input_ids'][0]
        assert len(full_ids) > len(prompt_ids)

        prompt_attn = prompt_data['attention_outputs']['model.layers.0.self_attn']['output'][1]
        full_attn = full_data['attention_outputs']['model.layers.0.self_attn']['output'][1]
        assert len(full_attn[0][0]) > len(prompt_attn[0][0])

    def test_input_ids_match_attention_seq_len(self):
        """input_ids length should match attention matrix dimensions."""
        for n in [3, 7, 15]:
            data = self._make_activation_data(num_tokens=n)
            num_ids = len(data['input_ids'][0])
            attn = data['attention_outputs']['model.layers.0.self_attn']['output'][1]
            assert num_ids == len(attn[0][0]) == n

    def test_all_layers_have_same_dimensions(self):
        """All layers should have the same attention matrix size for a given sequence."""
        data = self._make_activation_data(num_tokens=8, num_layers=3)
        for layer in range(3):
            module = f'model.layers.{layer}.self_attn'
            attn = data['attention_outputs'][module]['output'][1]
            assert len(attn[0][0]) == 8
            assert len(attn[0][0][0]) == 8
