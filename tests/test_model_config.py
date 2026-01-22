"""
Tests for utils/model_config.py

Tests model family lookups, configuration retrieval, and auto-selection logic.
"""

import pytest
from utils.model_config import (
    get_model_family,
    get_family_config,
    get_auto_selections,
    _pattern_matches_template,
    MODEL_TO_FAMILY,
    MODEL_FAMILIES
)


class TestGetModelFamily:
    """Tests for get_model_family function."""
    
    def test_known_gpt2_model(self):
        """Known GPT-2 model should return 'gpt2' family."""
        assert get_model_family("gpt2") == "gpt2"
        assert get_model_family("gpt2-medium") == "gpt2"
        assert get_model_family("openai-community/gpt2") == "gpt2"
    
    def test_known_llama_model(self):
        """Known LLaMA-like models should return 'llama_like' family."""
        assert get_model_family("Qwen/Qwen2.5-0.5B") == "llama_like"
        assert get_model_family("meta-llama/Llama-2-7b-hf") == "llama_like"
        assert get_model_family("mistralai/Mistral-7B-v0.1") == "llama_like"
    
    def test_known_opt_model(self):
        """Known OPT models should return 'opt' family."""
        assert get_model_family("facebook/opt-125m") == "opt"
        assert get_model_family("facebook/opt-1.3b") == "opt"
    
    def test_unknown_model_returns_none(self):
        """Unknown models should return None."""
        assert get_model_family("unknown/model-name") is None
        assert get_model_family("random-string") is None
        assert get_model_family("") is None


class TestGetFamilyConfig:
    """Tests for get_family_config function."""
    
    def test_valid_gpt2_config(self):
        """GPT-2 family config should have correct structure."""
        config = get_family_config("gpt2")
        assert config is not None
        assert "templates" in config
        assert "attention_pattern" in config["templates"]
        assert config["templates"]["attention_pattern"] == "transformer.h.{N}.attn"
        assert config["norm_type"] == "layernorm"
    
    def test_valid_llama_config(self):
        """LLaMA-like family config should have correct structure."""
        config = get_family_config("llama_like")
        assert config is not None
        assert config["templates"]["attention_pattern"] == "model.layers.{N}.self_attn"
        assert config["norm_type"] == "rmsnorm"
        assert config["norm_parameter"] == "model.norm.weight"
    
    def test_invalid_family_returns_none(self):
        """Invalid family name should return None."""
        assert get_family_config("invalid_family") is None
        assert get_family_config("") is None
        assert get_family_config("GPT2") is None  # Case-sensitive


class TestPatternMatchesTemplate:
    """Tests for _pattern_matches_template function."""
    
    def test_exact_match(self):
        """Pattern that exactly matches template should return True."""
        assert _pattern_matches_template(
            "model.layers.{N}.self_attn",
            "model.layers.{N}.self_attn"
        ) is True
    
    def test_matching_with_n_placeholder(self):
        """Patterns with {N} placeholder should match correctly."""
        assert _pattern_matches_template(
            "transformer.h.{N}.attn",
            "transformer.h.{N}.attn"
        ) is True
    
    def test_non_matching_pattern(self):
        """Different patterns should not match."""
        assert _pattern_matches_template(
            "model.layers.{N}.self_attn",
            "transformer.h.{N}.attn"
        ) is False
    
    def test_empty_template_returns_false(self):
        """Empty template should return False."""
        assert _pattern_matches_template("model.layers.{N}.self_attn", "") is False
        assert _pattern_matches_template("", "") is False


class TestGetAutoSelections:
    """Tests for get_auto_selections function."""
    
    def test_unknown_model_returns_empty_selections(self):
        """Unknown model should return empty selections."""
        result = get_auto_selections(
            "unknown/model",
            {"model.layers.{N}.self_attn": ["model.layers.0.self_attn"]},
            {"model.norm.weight": ["model.norm.weight"]}
        )
        assert result["attention_selection"] == []
        assert result["block_selection"] == []
        assert result["norm_selection"] == []
        assert result["family_name"] is None
    
    def test_known_model_matches_patterns(self, mock_module_patterns, mock_param_patterns):
        """Known model should match appropriate patterns."""
        result = get_auto_selections(
            "Qwen/Qwen2.5-0.5B",  # llama_like family
            mock_module_patterns,
            mock_param_patterns
        )
        assert result["family_name"] == "llama_like"
        # Should find self_attn pattern
        assert "model.layers.{N}.self_attn" in result["attention_selection"]
        # Should find block pattern
        assert "model.layers.{N}" in result["block_selection"]
        # Should find norm pattern
        assert result["norm_selection"] == ["model.norm.weight"]
    
    def test_result_structure(self, mock_module_patterns, mock_param_patterns):
        """Result should have all required keys."""
        result = get_auto_selections(
            "gpt2",
            {},  # Empty patterns - no matches expected
            {}
        )
        assert "attention_selection" in result
        assert "block_selection" in result
        assert "norm_selection" in result
        assert "family_name" in result
        assert isinstance(result["attention_selection"], list)
        assert isinstance(result["norm_selection"], list)


class TestModelRegistryIntegrity:
    """Tests to verify the model registry data is consistent."""
    
    def test_all_families_have_required_fields(self):
        """All model families should have required configuration fields."""
        required_fields = ["description", "templates", "norm_type"]
        for family_name, config in MODEL_FAMILIES.items():
            for field in required_fields:
                assert field in config, f"Family {family_name} missing {field}"
    
    def test_all_mapped_families_exist(self):
        """All families referenced in MODEL_TO_FAMILY should exist in MODEL_FAMILIES."""
        for model_name, family_name in MODEL_TO_FAMILY.items():
            assert family_name in MODEL_FAMILIES, \
                f"Model {model_name} references unknown family {family_name}"
