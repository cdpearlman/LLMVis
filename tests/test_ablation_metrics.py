"""
Tests for utils/ablation_metrics.py

Tests KL divergence computation and probability delta calculations.
Uses synthetic tensors to avoid model loading.
"""

import pytest
import torch
import torch.nn.functional as F
from utils.ablation_metrics import compute_kl_divergence, get_token_probability_deltas


class TestComputeKLDivergence:
    """Tests for compute_kl_divergence function."""
    
    def test_identical_distributions_zero_kl(self, identical_logits):
        """KL divergence of identical distributions should be approximately 0."""
        logits_p, logits_q = identical_logits
        kl_divs = compute_kl_divergence(logits_p, logits_q)
        
        assert isinstance(kl_divs, list)
        assert len(kl_divs) == 2  # seq_len = 2
        for kl in kl_divs:
            assert abs(kl) < 1e-5, f"Expected ~0, got {kl}"
    
    def test_different_distributions_positive_kl(self, different_logits):
        """KL divergence of different distributions should be positive."""
        logits_p, logits_q = different_logits
        kl_divs = compute_kl_divergence(logits_p, logits_q)
        
        assert isinstance(kl_divs, list)
        for kl in kl_divs:
            assert kl > 0, f"Expected positive KL, got {kl}"
    
    def test_kl_divergence_asymmetry(self, different_logits):
        """KL(P||Q) should not equal KL(Q||P) for different distributions."""
        logits_p, logits_q = different_logits
        kl_pq = compute_kl_divergence(logits_p, logits_q)
        kl_qp = compute_kl_divergence(logits_q, logits_p)
        
        # They should generally be different (asymmetry of KL divergence)
        assert kl_pq != kl_qp, "KL divergence should be asymmetric"
    
    def test_handles_3d_input(self):
        """Should handle [batch, seq_len, vocab_size] input correctly."""
        logits = torch.randn(1, 5, 100)  # batch=1, seq=5, vocab=100
        kl_divs = compute_kl_divergence(logits, logits)
        
        assert len(kl_divs) == 5
        for kl in kl_divs:
            assert abs(kl) < 1e-5


class TestGetTokenProbabilityDeltas:
    """Tests for get_token_probability_deltas function."""
    
    def test_deltas_with_synthetic_data(self):
        """Test probability delta computation with known inputs."""
        # Logits shape: [1, seq_len, vocab_size] where seq_len matches input_ids
        # input_ids has 3 tokens, so logits needs 3 positions
        logits_ref = torch.tensor([[[1.0, 2.0, 3.0, 10.0],   # pos 0
                                    [1.0, 2.0, 10.0, 3.0],   # pos 1
                                    [1.0, 2.0, 3.0, 4.0]]])  # pos 2
        logits_abl = torch.tensor([[[10.0, 2.0, 3.0, 1.0],
                                    [10.0, 2.0, 1.0, 3.0],
                                    [1.0, 2.0, 3.0, 4.0]]])
        input_ids = torch.tensor([[0, 3, 2]])
        
        deltas = get_token_probability_deltas(logits_ref, logits_abl, input_ids)
        
        # Should return list of length seq_len - 1 (shifted prediction)
        assert isinstance(deltas, list)
        assert len(deltas) == 2  # seq_len=3, so 2 predictions (pos 0 predicts token 1, pos 1 predicts token 2)
    
    def test_identical_logits_zero_delta(self):
        """Identical logits should produce zero deltas."""
        # Logits need seq_len=3 to match input_ids
        logits = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                                [2.0, 3.0, 4.0, 5.0],
                                [3.0, 4.0, 5.0, 6.0]]])
        input_ids = torch.tensor([[0, 3, 2]])
        
        deltas = get_token_probability_deltas(logits, logits.clone(), input_ids)
        
        for delta in deltas:
            assert abs(delta) < 1e-5, f"Expected ~0 delta, got {delta}"
    
    def test_delta_direction(self):
        """When ablation increases a token's probability, delta should be positive."""
        # 3 positions to match 3 input_ids
        logits_ref = torch.tensor([[[1.0, 0.0, 0.0, 0.0],   # favors token 0
                                    [1.0, 0.0, 0.0, 0.0],   # favors token 0
                                    [1.0, 0.0, 0.0, 0.0]]])
        logits_abl = torch.tensor([[[0.0, 10.0, 0.0, 0.0],  # favors token 1
                                    [0.0, 10.0, 0.0, 0.0],  # favors token 1
                                    [0.0, 10.0, 0.0, 0.0]]])
        input_ids = torch.tensor([[0, 1, 1]])  # Target tokens: 1, 1
        
        deltas = get_token_probability_deltas(logits_ref, logits_abl, input_ids)
        
        # Both deltas should be positive (ablation increased target prob)
        for delta in deltas:
            assert delta > 0, f"Expected positive delta, got {delta}"
    
    def test_delta_range(self):
        """Deltas should be bounded by [-1, 1] since they're probability differences."""
        # 3 positions to match input_ids
        logits_ref = torch.tensor([[[100.0, -100.0, -100.0, -100.0],
                                    [-100.0, 100.0, -100.0, -100.0],
                                    [-100.0, -100.0, 100.0, -100.0]]])
        logits_abl = torch.tensor([[[-100.0, 100.0, -100.0, -100.0],
                                    [-100.0, -100.0, 100.0, -100.0],
                                    [-100.0, -100.0, -100.0, 100.0]]])
        input_ids = torch.tensor([[0, 0, 1]])  # Targets: 0, 1
        
        deltas = get_token_probability_deltas(logits_ref, logits_abl, input_ids)
        
        for delta in deltas:
            assert -1.0 <= delta <= 1.0, f"Delta {delta} out of bounds"
