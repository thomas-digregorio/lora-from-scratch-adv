import torch
import torch.nn as nn
from lora_from_scratch.layers import LoRALinear

def test_lora_linear():
    print("Testing LoRALinear...")
    
    # Setup
    in_dim, out_dim = 10, 5
    batch_size = 2
    x = torch.randn(batch_size, in_dim)
    
    # 1. Test Instantiation from Linear
    original = nn.Linear(in_dim, out_dim)
    lora = LoRALinear.from_linear(original, rank=4, alpha=8)
    
    # Verify weights are frozen
    assert not lora.weight.requires_grad
    assert not lora.bias.requires_grad
    assert lora.lora_A.requires_grad
    assert lora.lora_B.requires_grad
    print("[PASS] Initialization & Freezing")
    
    # 2. Test Forward Pass (Unmerged)
    # At init, B is zero, so output should match original
    out_orig = original(x)
    out_lora = lora(x)
    assert torch.allclose(out_orig, out_lora, atol=1e-6)
    print("[PASS] Initial Forward Pass (Identity)")
    
    # 3. Test Merging logic
    # Manually change lora_B to be non-zero to see effect
    nn.init.ones_(lora.lora_B) 
    
    # Output should now be different
    out_unmerged = lora(x)
    assert not torch.allclose(out_orig, out_unmerged)
    
    # Merge
    lora.merge()
    assert lora.merged
    out_merged = lora(x)
    
    # Output should be identical to unmerged
    assert torch.allclose(out_unmerged, out_merged, atol=1e-6)
    print("[PASS] Merge Correctness")
    
    # 4. Test Unmerge
    lora.unmerge()
    assert not lora.merged
    # Weight should be restored to original
    assert torch.allclose(lora.weight, original.weight)
    print("[PASS] Unmerge Correctness")
    
    print("All tests passed!")

if __name__ == "__main__":
    test_lora_linear()
