import torch
import torch.nn as nn
from lora_from_scratch.model import inject_lora, mark_only_lora_as_trainable
from lora_from_scratch.layers import LoRALinear

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # A mock transformer-like structure
        self.transformer = nn.ModuleDict({
            'layer_0': nn.ModuleDict({
                'attention': nn.ModuleDict({
                    'query': nn.Linear(32, 32),
                    'key': nn.Linear(32, 32),
                    'value': nn.Linear(32, 32),
                    'output': nn.Linear(32, 32)
                }),
                'mlp': nn.Sequential(
                    nn.Linear(32, 128),
                    nn.ReLU(),
                    nn.Linear(128, 32)
                )
            })
        })
        self.classifier = nn.Linear(32, 2)

def test_injection():
    print("Testing LoRA Injection...")
    
    model = DummyModel()
    
    # Target only query and value projections
    target_modules = ["query", "value"]
    
    print(f"Injecting into: {target_modules}")
    inject_lora(model, target_modules, rank=4, alpha=8, verbose=True)
    
    # Verify replacements
    # 1. Check Query
    query_layer = model.transformer.layer_0.attention.query
    assert isinstance(query_layer, LoRALinear)
    print("[PASS] replaced 'query'")
    
    # 2. Check Value
    value_layer = model.transformer.layer_0.attention.value
    assert isinstance(value_layer, LoRALinear)
    print("[PASS] replaced 'value'")
    
    # 3. Check Key (Should NOT be replaced)
    key_layer = model.transformer.layer_0.attention.key
    assert isinstance(key_layer, nn.Linear)
    assert not isinstance(key_layer, LoRALinear)
    print("[PASS] ignored 'key'")
    
    # 4. Check Classifier (Should NOT be replaced)
    assert isinstance(model.classifier, nn.Linear)
    assert not isinstance(model.classifier, LoRALinear)
    print("[PASS] ignored 'classifier'")
    
    # 5. Check parameter freezing
    mark_only_lora_as_trainable(model)
    
    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    # Should only contain lora_A and lora_B
    for name in trainable_params:
        assert "lora_" in name
    
    # Ensure original weights are frozen
    assert not query_layer.weight.requires_grad
    print("[PASS] Parameter freezing correct")
    
    print("All injection tests passed!")

if __name__ == "__main__":
    test_injection()
