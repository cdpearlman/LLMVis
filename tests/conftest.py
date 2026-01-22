"""
Shared pytest fixtures for the test suite.

Provides reusable mock data structures and synthetic tensors
to test utility functions without loading actual ML models.
"""

import pytest
import torch
import numpy as np


# =============================================================================
# Synthetic Attention Matrices
# =============================================================================

@pytest.fixture
def uniform_attention_matrix():
    """4x4 uniform attention matrix (each position attends equally to all)."""
    size = 4
    return torch.ones(size, size) / size


@pytest.fixture
def previous_token_attention_matrix():
    """
    4x4 attention matrix where each position attends primarily to the previous token.
    Position 0 attends to itself (no previous token).
    """
    size = 4
    matrix = torch.zeros(size, size)
    # Position 0 attends to itself
    matrix[0, 0] = 1.0
    # Other positions attend strongly to previous token
    for i in range(1, size):
        matrix[i, i-1] = 0.8
        matrix[i, i] = 0.2  # Some self-attention
    return matrix


@pytest.fixture
def first_token_attention_matrix():
    """4x4 attention matrix where all positions attend strongly to first token."""
    size = 4
    matrix = torch.zeros(size, size)
    for i in range(size):
        matrix[i, 0] = 0.7  # Strong attention to first token
        matrix[i, i] = 0.3  # Some self-attention
    return matrix


@pytest.fixture
def peaked_attention_matrix():
    """4x4 attention matrix with peaked (low entropy) attention at one position."""
    size = 4
    matrix = torch.zeros(size, size)
    # Each position attends almost entirely to position 2
    for i in range(size):
        matrix[i, 2] = 0.95
        # Distribute remaining across others
        for j in range(size):
            if j != 2:
                matrix[i, j] = 0.05 / (size - 1)
    return matrix


# =============================================================================
# Mock Activation Data Structures
# =============================================================================

@pytest.fixture
def mock_activation_data():
    """
    Mock activation data structure similar to execute_forward_pass output.
    Used for testing functions that process activation data.
    """
    return {
        'model': 'mock-model',
        'prompt': 'Hello world',
        'input_ids': [[1, 2, 3, 4]],
        'attention_modules': ['model.layers.0.self_attn', 'model.layers.1.self_attn'],
        'attention_outputs': {
            'model.layers.0.self_attn': {
                'output': [
                    [[0.1, 0.2, 0.3]],  # Hidden states (simplified)
                    [[[[0.25, 0.25, 0.25, 0.25],  # Attention weights [batch, heads, seq, seq]
                       [0.25, 0.25, 0.25, 0.25],
                       [0.25, 0.25, 0.25, 0.25],
                       [0.25, 0.25, 0.25, 0.25]]]]
                ]
            },
            'model.layers.1.self_attn': {
                'output': [
                    [[0.1, 0.2, 0.3]],
                    [[[[0.1, 0.2, 0.3, 0.4],
                       [0.1, 0.2, 0.3, 0.4],
                       [0.1, 0.2, 0.3, 0.4],
                       [0.1, 0.2, 0.3, 0.4]]]]
                ]
            }
        },
        'block_modules': ['model.layers.0', 'model.layers.1'],
        'block_outputs': {
            'model.layers.0': {'output': [[[0.1, 0.2, 0.3, 0.4]]]},
            'model.layers.1': {'output': [[[0.2, 0.3, 0.4, 0.5]]]}
        },
        'norm_parameters': ['model.norm.weight'],
        'norm_data': [[1.0, 1.0, 1.0, 1.0]],
        'actual_output': {'token': ' world', 'probability': 0.85},
        'global_top5_tokens': [
            {'token': 'world', 'probability': 0.85},
            {'token': 'there', 'probability': 0.05},
            {'token': 'friend', 'probability': 0.03},
            {'token': 'everyone', 'probability': 0.02},
            {'token': 'all', 'probability': 0.01}
        ]
    }


# =============================================================================
# Mock Module/Parameter Patterns
# =============================================================================

@pytest.fixture
def mock_module_patterns():
    """Mock module patterns as returned by extract_patterns."""
    return {
        'model.layers.{N}.self_attn': ['model.layers.0.self_attn', 'model.layers.1.self_attn'],
        'model.layers.{N}.mlp': ['model.layers.0.mlp', 'model.layers.1.mlp'],
        'model.layers.{N}': ['model.layers.0', 'model.layers.1'],
        'model.embed_tokens': ['model.embed_tokens'],
        'model.norm': ['model.norm']
    }


@pytest.fixture
def mock_param_patterns():
    """Mock parameter patterns as returned by extract_patterns."""
    return {
        'model.layers.{N}.self_attn.q_proj.weight': ['model.layers.0.self_attn.q_proj.weight'],
        'model.layers.{N}.self_attn.k_proj.weight': ['model.layers.0.self_attn.k_proj.weight'],
        'model.norm.weight': ['model.norm.weight'],
        'lm_head.weight': ['lm_head.weight']
    }


# =============================================================================
# Synthetic Logits for Ablation Metrics
# =============================================================================

@pytest.fixture
def identical_logits():
    """Two identical logit tensors for testing KL divergence = 0."""
    logits = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                            [2.0, 3.0, 4.0, 5.0]]])  # [1, 2, 4] = [batch, seq, vocab]
    return logits, logits.clone()


@pytest.fixture
def different_logits():
    """Two different logit tensors for testing KL divergence > 0."""
    logits_p = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                              [2.0, 3.0, 4.0, 5.0]]])
    logits_q = torch.tensor([[[4.0, 3.0, 2.0, 1.0],
                              [5.0, 4.0, 3.0, 2.0]]])
    return logits_p, logits_q


@pytest.fixture
def prob_delta_data():
    """Data for testing probability delta computation."""
    # Reference favors token 3, ablated favors token 0
    logits_ref = torch.tensor([[[1.0, 2.0, 3.0, 10.0],   # pos 0: predicts token 3
                                [1.0, 2.0, 10.0, 3.0]]])  # pos 1: predicts token 2
    logits_abl = torch.tensor([[[10.0, 2.0, 3.0, 1.0],   # pos 0: predicts token 0
                                [10.0, 2.0, 1.0, 3.0]]])  # pos 1: predicts token 0
    input_ids = torch.tensor([[0, 3, 2]])  # Actual tokens: start, 3, 2
    return logits_ref, logits_abl, input_ids


# =============================================================================
# Attribution Data for Visualization Tests
# =============================================================================

@pytest.fixture
def mock_attribution_result():
    """Mock output from compute_integrated_gradients or compute_simple_gradient_attribution."""
    return {
        'tokens': ['Hello', ' world', '!'],
        'token_ids': [1, 2, 3],
        'attributions': [0.5, 1.0, 0.2],  # Raw attribution scores
        'normalized_attributions': [0.5, 1.0, 0.2],  # Already normalized for simplicity
        'target_token': 'next',
        'target_token_id': 100
    }


# =============================================================================
# Head Categorization Config
# =============================================================================

@pytest.fixture
def default_head_config():
    """Default head categorization configuration for testing."""
    from utils.head_detection import HeadCategorizationConfig
    return HeadCategorizationConfig()
