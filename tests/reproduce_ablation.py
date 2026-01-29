
import sys
import os
import torch
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.model_patterns import execute_forward_pass, execute_forward_pass_with_head_ablation, load_model_and_get_patterns

def test_ablation_changes_output():
    """
    Verify that ablating a head changes the model output compared to a baseline run.
    """
    model_name = "gpt2"  # Small model for testing
    prompt = "The quick brown fox jumps over the"
    
    print(f"Loading model: {model_name}")
    # We can use the utility to load, but it prints a lot. 
    # Let's just use the load_model_and_get_patterns which handles config too
    try:
        module_patterns, param_patterns = load_model_and_get_patterns(model_name)
    except Exception as e:
        pytest.skip(f"Could not load model {model_name}: {e}")
        return

    # Re-load model/tokenizer locally to have direct access if needed, 
    # but the utils need the model object. 
    # load_model_and_get_patterns returns patterns, but we need the model object.
    # Actually, execute_forward_pass takes (model, tokenizer, ...).
    # load_model_and_get_patterns DOES NOT return the model. It loads it internally to get patterns.
    # I need to load the model myself to pass it to execute_forward_pass.
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    # Define config for capture (we need to capture something to make the function work)
    # For GPT-2, attention is transformer.h.{N}.attn
    # We'll stick to the "module_patterns" logic implicitly or explicitly.
    # Let's capture L0 attention
    config = {
        "attention_modules": ["transformer.h.0.attn"],
        "block_modules": ["transformer.h.0"],
        "norm_parameters": [],
        "logit_lens_parameter": "transformer.ln_f.weight" 
    }

    # 1. Baseline Run
    print("Running baseline...")
    baseline_result = execute_forward_pass(model, tokenizer, prompt, config)
    baseline_top_token = baseline_result['actual_output']['token']
    baseline_top_prob = baseline_result['actual_output']['probability']
    print(f"Baseline Output: '{baseline_top_token}' ({baseline_top_prob:.4f})")

    # 2. Ablated Run (Layer 0, Head 0)
    print("Running ablation (L0H0)...")
    ablation_result = execute_forward_pass_with_head_ablation(
        model, tokenizer, prompt, config,
        ablate_layer_num=0,
        ablate_head_indices=[0]
    )
    ablated_top_token = ablation_result['actual_output']['token']
    ablated_top_prob = ablation_result['actual_output']['probability']
    print(f"Ablated Output: '{ablated_top_token}' ({ablated_top_prob:.4f})")
    
    # 3. Assertions
    # We expect the probability to change, even if the token doesn't (depending on head importance)
    # Ideally, exact logit match should be false.
    
    # Check if probabilities are different (using a small epsilon)
    prob_diff = abs(baseline_top_prob - ablated_top_prob)
    print(f"Probability Difference: {prob_diff}")
    
    # We assert that there IS a difference. 
    # Note: If L0H0 is completely useless, this might fail. But usually it does something.
    assert prob_diff > 1e-6, "Ablation of L0H0 did not change the top token probability at all!"

    # Verify that the structure returned contains ablation info
    assert ablation_result['ablated_layer'] == 0
    assert ablation_result['ablated_heads'] == [0]

if __name__ == "__main__":
    test_ablation_changes_output()
