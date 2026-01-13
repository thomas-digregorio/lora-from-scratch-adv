import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    A LoRA-augmented Linear Layer that wraps an existing linear layer or creates one.
    
    This layer freezes the pretrained weight and adds a parallel low-rank adapter
    (A * B) where A is initialized closely to random noise and B is initialized to zero.
    
    Attributes:
        weight (nn.Parameter): The frozen pretrained weight matrix.
        bias (Optional[nn.Parameter]): The frozen pretrained bias vector.
        lora_A (nn.Parameter): The trainable low-rank down-projection matrix.
        lora_B (nn.Parameter): The trainable low-rank up-projection matrix.
        scaling (float): The scaling factor alpha / rank.
        merged (bool): Whether the adapters are currently merged into the main weight.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        bias: bool = True,
    ) -> None:
        """
        Initializes the LoRALinear layer.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            rank: The rank 'r' of the low-rank adapters.
            alpha: Scaling factor for the adapter output.
            bias: If set to False, the layer will not learn an additive bias.
        """
        super().__init__()
        
        # 1. The Pre-trained Weight (Frozen)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight.requires_grad = False
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            self.bias.requires_grad = False
        else:
            self.register_parameter('bias', None)

        # 2. The Low-Rank Adapters (Trainable)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Matrix A: Down-projection (In -> Rank)
        self.lora_A = nn.Parameter(torch.Tensor(rank, in_features))
        # Matrix B: Up-projection (Rank -> Out)
        self.lora_B = nn.Parameter(torch.Tensor(out_features, rank))
        
        self.merged = False
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset all parameters to their default initialization."""
        # Initialize frozen weight like a standard Linear layer
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # Initialize LoRA matrices
        # A: Kaiming/He initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B: Zero initialization (ensures identity function at start)
        nn.init.zeros_(self.lora_B)

    def merge(self) -> None:
        """
        Merges the LoRA weights into the pretrained weight matrix.
        
        This effectively performs: W_new = W + (B @ A) * scaling
        Used for efficient inference to avoid the extra computational overhead of the adapter branch.
        """
        if self.merged:
            return

        # W = W + BA * scale
        # Note: We compute (B @ A) which is (Out x Rank) @ (Rank x In) -> (Out x In)
        # We use .data to update the tensor in-place without tracking gradients for this op
        self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
        self.merged = True

    def unmerge(self) -> None:
        """
        Unmerges the LoRA weights from the pretrained weight matrix.
        
        Restores W to its original frozen state.
        """
        if not self.merged:
            return

        # W = W - BA * scale
        self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
        self.merged = False

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int = 8, alpha: int = 16) -> "LoRALinear":
        """
        Creates a LoRALinear layer from an existing nn.Linear layer.

        Args:
            linear: The existing nn.Linear layer to wrap.
            rank: The rank of the adapters.
            alpha: The scaling factor.

        Returns:
            A new LoRALinear instance initialized with the linear layer's weights.
        """
        lora_layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            bias=linear.bias is not None
        )
        
        # Copy the weights and bias
        lora_layer.weight.data = linear.weight.data.clone()
        lora_layer.weight.requires_grad = False
        
        if linear.bias is not None:
            lora_layer.bias.data = linear.bias.data.clone()
            lora_layer.bias.requires_grad = False
            
        return lora_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LoRA Linear layer.
        
        Args:
            x: Input tensor of shape (batch_size, *, in_features).
            
        Returns:
            Output tensor of shape (batch_size, *, out_features).
        """
        # If merged, we just use the standard linear forward pass
        if self.merged:
            return F.linear(x, self.weight, self.bias)
            
        # 1. The "Standard" Path (Frozen W)
        pretrained_out = F.linear(x, self.weight, self.bias)
        
        # 2. The "Adapter" Path
        # x shape: (B, *, In)
        # lora_A shape: (Rank, In)
        # lora_B shape: (Out, Rank)
        #
        # (x @ A.T) @ B.T
        # We can implement this compactly:
        # dropout is typically applied to x before lora_A, but we omit it for simplicity as per original snippet
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        
        # 3. Combine
        return pretrained_out + (lora_out * self.scaling)
