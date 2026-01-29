
import sys
import os
import torch
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.model_patterns import execute_forward_pass, execute_forward_pass_with_multi_layer_head_ablation

def test_multi_layer_ablation():
    """
    Verify that ablating heads across multiple layers works.
    """
    model_name = "gpt2"
    prompt = "The quick brown fox jumps over the"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    config = {
        "attention_modules": ["transformer.h.0.attn", "transformer.h.1.attn"],
        "block_modules": ["transformer.h.0", "transformer.h.1"],
        "norm_parameters": [],
        "logit_lens_parameter": "transformer.ln_f.weight" 
    }

    # 1. Baseline
    baseline = execute_forward_pass(model, tokenizer, prompt, config)
    baseline_prob = baseline['actual_output']['probability']
    
    # 2. Ablate L0H0 and L1H1
    # Note: heads_by_layer expects {layer_num: [head_indices]}
    heads_to_ablate = {
        0: [0],
        1: [1]
    }
    
    ablated = execute_forward_pass_with_multi_layer_head_ablation(
        model, tokenizer, prompt, config, heads_to_ablate
    )
    ablated_prob = ablated['actual_output']['probability']
    
    print(f"Baseline: {baseline_prob}, Ablated: {ablated_prob}")
    
    # Assert change
    assert abs(baseline_prob - ablated_prob) > 1e-6
    
    # Assert return structure contains ablation info
    assert ablated['ablated_heads_by_layer'] == heads_to_ablate

if __name__ == "__main__":
    test_multi_layer_ablation()
