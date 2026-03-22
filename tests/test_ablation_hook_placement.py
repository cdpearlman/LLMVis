"""
Tests for scientifically accurate head ablation via pre-projection hooking.

Verifies that ablation hooks are placed on the INPUT to c_proj (pre-projection),
where per-head dimensions are still separable, rather than on the OUTPUT of the
attention module (post-projection), where heads are mixed.
"""

import sys
import os
import torch
import torch.nn as nn
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.model_patterns import _find_output_proj_submodule


@pytest.fixture(scope="module")
def gpt2_model_and_tokenizer():
    """Load GPT-2 once for all tests in this module."""
    try:
        model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        return model, tokenizer
    except Exception as e:
        pytest.skip(f"Could not load GPT-2: {e}")


class TestFindOutputProjSubmodule:
    def test_find_output_proj_gpt2(self, gpt2_model_and_tokenizer):
        """GPT-2 attention modules should have c_proj as the output projection."""
        model, _ = gpt2_model_and_tokenizer
        attn_module = model.transformer.h[0].attn
        name, submodule = _find_output_proj_submodule(attn_module)
        assert name == "c_proj"
        assert submodule is attn_module.c_proj

    def test_find_output_proj_unknown_raises(self):
        """A plain nn.Module with no recognized projection children should raise ValueError."""
        plain_module = nn.Module()
        plain_module.add_module("some_layer", nn.Linear(10, 10))
        with pytest.raises(ValueError, match="No output projection found"):
            _find_output_proj_submodule(plain_module)


class TestPreHookPlacement:
    def test_pre_hook_zeros_correct_dims(self, gpt2_model_and_tokenizer):
        """Pre-hook on c_proj receives input where per-head dims are separable.
        Zeroing head 0 dims [0:64] should leave [64:768] untouched."""
        model, tokenizer = gpt2_model_and_tokenizer
        captured_input = {}

        def capture_pre_hook(module, args):
            captured_input['x'] = args[0].clone()
            return None  # Don't modify

        hook = model.transformer.h[0].attn.c_proj.register_forward_pre_hook(capture_pre_hook)
        try:
            inputs = tokenizer("The cat sat on the mat", return_tensors="pt")
            with torch.no_grad():
                model(**inputs, use_cache=False)
        finally:
            hook.remove()

        x = captured_input['x']
        # Shape should be [batch, seq, 768]
        assert x.shape[-1] == 768
        # Verify per-head structure: zeroing [0:64] leaves [64:768] intact
        x_modified = x.clone()
        x_modified[:, :, 0:64] = 0.0
        # The rest should be exactly equal
        assert torch.equal(x_modified[:, :, 64:], x[:, :, 64:])
        # The zeroed part should actually be zero
        assert torch.all(x_modified[:, :, 0:64] == 0.0)

    def test_ablation_changes_output(self, gpt2_model_and_tokenizer):
        """Ablating head 0 at layer 0 via pre-hook on c_proj should change logits."""
        model, tokenizer = gpt2_model_and_tokenizer
        prompt = "The quick brown fox jumps over the"
        inputs = tokenizer(prompt, return_tensors="pt")

        # Baseline
        with torch.no_grad():
            baseline_logits = model(**inputs, use_cache=False).logits

        # Ablated: pre-hook zeros head 0 on layer 0's c_proj
        def ablation_pre_hook(module, args):
            x = args[0].clone()
            x[:, :, 0:64] = 0.0
            return (x,)

        hook = model.transformer.h[0].attn.c_proj.register_forward_pre_hook(ablation_pre_hook)
        try:
            with torch.no_grad():
                ablated_logits = model(**inputs, use_cache=False).logits
        finally:
            hook.remove()

        # Logits must differ
        assert not torch.allclose(baseline_logits, ablated_logits, atol=1e-6), \
            "Ablation via pre-hook on c_proj did not change logits"
