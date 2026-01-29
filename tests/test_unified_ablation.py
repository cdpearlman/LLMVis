
import sys
import os
import torch
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.model_patterns import execute_forward_pass

def test_unified_ablation():
    """
    Verify that execute_forward_pass can handle ablation configuration.
    """
    model_name = "gpt2"
    prompt = "The quick brown fox jumps over the"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    config = {
        "attention_modules": ["transformer.h.0.attn"],
        "block_modules": ["transformer.h.0"],
        "norm_parameters": [],
        "logit_lens_parameter": "transformer.ln_f.weight" 
    }

    # 1. Baseline
    baseline = execute_forward_pass(model, tokenizer, prompt, config)
    baseline_prob = baseline['actual_output']['probability']
    
    # 2. Ablated via execute_forward_pass (New API we want to support)
    heads_to_ablate = {0: [0]} # Layer 0, Head 0
    
    # We expect this to fail currently as the argument doesn't exist
    try:
        ablated = execute_forward_pass(
            model, tokenizer, prompt, config, 
            ablation_config=heads_to_ablate
        )
        ablated_prob = ablated['actual_output']['probability']
        
        print(f"Baseline: {baseline_prob}, Ablated: {ablated_prob}")
        
        # Assert change
        assert abs(baseline_prob - ablated_prob) > 1e-6
        assert ablated.get('ablated_heads_by_layer') == heads_to_ablate
        
    except TypeError:
        pytest.fail("execute_forward_pass does not accept ablation_config argument yet")

if __name__ == "__main__":
    test_unified_ablation()
