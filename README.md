# LoRA From Scratch

A clean implementation of Low-Rank Adaptation (LoRA) for PyTorch models.

## Features
- **LoRALinear**: A drop-in replacement for `nn.Linear`.
- **Dynamic Injection**: Automatically inject LoRA adapters into specific layers (e.g., Attention Q/V).
- **Weight Merging**: Merge adapter weights into the base model for zero-overhead inference.
- **Benchmarking**: Scripts to compare Full Fine-Tuning vs. LoRA performance.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repo_url>
   cd lora_from_scratch
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -e .
   ```

## Usage

### Basic Usage
```python
from lora_from_scratch.layers import LoRALinear
import torch.nn as nn

# Replace a standard layer
original_layer = nn.Linear(768, 768)
lora_layer = LoRALinear.from_linear(original_layer, rank=8, alpha=16)

# Forward pass
output = lora_layer(input_tensor)
```
