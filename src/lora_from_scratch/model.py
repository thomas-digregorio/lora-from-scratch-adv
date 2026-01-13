import re
from typing import List, Optional, Union

import torch
import torch.nn as nn
from .layers import LoRALinear


def inject_lora(
    model: nn.Module,
    target_modules: List[str],
    rank: int = 8,
    alpha: int = 16,
    verbose: bool = True,
) -> None:
    """
    Recursively injects LoRALinear layers into the model in-place.

    Args:
        model: The PyTorch model to modify.
        target_modules: List of strings. If a module's name ends with any of these
                        strings (e.g. "query", "value"), it will be replaced.
        rank: LoRA rank.
        alpha: LoRA scaling factor.
        verbose: Whether to print replaced modules.
    """
    if verbose:
        print(f"Injecting LoRA (rank={rank}, alpha={alpha}) into modules matching: {target_modules}")

    for name, module in model.named_modules():
        # Skip the root module itself
        if name == "":
            continue

        # Check if this module matches any target suffix
        if any(name.endswith(target) for target in target_modules):
            # We found a target module. Now we need to find its parent to replace it.
            # We iterate named_modules again to find the parent because we need to set the attribute on the parent.
            # A more efficient way is to split the name.
            
            parent_name, child_name = _get_parent_child_names(name)
            
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                # If parent_name is empty, the parent is the model itself
                parent = model

            # Only replace if it's a standard nn.Linear
            if isinstance(module, nn.Linear):
                if verbose:
                    print(f"  Replacing: {name} ({module.in_features} -> {module.out_features})")
                
                # Create the LoRA layer
                lora_layer = LoRALinear.from_linear(module, rank=rank, alpha=alpha)
                
                # Replace in parent
                setattr(parent, child_name, lora_layer)
            elif verbose:
                print(f"  Skipping: {name} (Matched target but is {type(module)}, not nn.Linear)")

def mark_only_lora_as_trainable(model: nn.Module) -> None:
    """
    Freezes all parameters in the model except for LoRA parameters.
    """
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

def lora_state_dict(model: nn.Module) -> dict:
    """
    Returns a state dictionary containing only the LoRA parameters.
    Useful for saving lightweight checkpoints.
    """
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}

def _get_parent_child_names(full_name: str) -> tuple[str, str]:
    """
    Splits a full module name (e.g., "transformer.layer.0.attention")
    into parent path ("transformer.layer.0") and child name ("attention").
    """
    if "." not in full_name:
        return "", full_name
    
    parts = full_name.rsplit(".", 1)
    return parts[0], parts[1]
