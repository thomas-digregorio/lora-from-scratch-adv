# LoRA From Scratch


A clean implementation of Low-Rank Adaptation (LoRA) for PyTorch models.

This project demonstrates LoRA by:
1.  **Fine-Tuning**: Fine-tuning `distilbert-base-uncased` performance on the **SST-2** binary sentiment classification task.
2.  **Integration Testing**: Injecting adapters into `facebook/opt-125m` to verify structural integrity and parameter freezing on a generative model.

Its primary goal is to allow efficient fine-tuning of large models by injecting trainable low-rank matrices into linear layers while keeping the original model weights frozen.

Here is a detailed breakdown of the codebase:

## Features
- **LoRALinear**: A drop-in replacement for `nn.Linear`.
- **Dynamic Injection**: Automatically inject LoRA adapters into specific layers (e.g., Attention Q/V).
- **Weight Merging**: Merge adapter weights into the base model for zero-overhead inference.
- **Benchmarking**: Scripts to compare Full Fine-Tuning vs. LoRA performance.

## Directory Structure

```
├── benchmarks/           # Training & Inference scripts (SST-2)
│   ├── train.py          # Main training loop (DistilBERT + LoRA)
│   └── benchmark_inference.py # Latency testing
├── examples/             # Integration demos
│   └── verify_integration.py # OPT-125m integrity check
├── src/                  # Core Library
│   └── lora_from_scratch/
│       ├── layers.py     # LoRALinear implementation
│       └── model.py      # Injection & freezing utilities
└── results_visualization.ipynb # Jupyter notebook for plotting metrics
```

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

## System Design & Verification

### Why separate Verification and Training?

In this project, we separated the **structural verification** from the **task-specific training** to ensure robustness at different levels of abstraction.

#### 1. Integration Verification (`examples/verify_integration.py`)
**Model Choice:** `facebook/opt-125m` (Causal Language Model)
**Why OPT-125m?**
*   **Visual Verification**: As a text-generation model, we can "eye-ball" the output. If the model generates coherent English after LoRA injection, we know the structural integrity of the model is preserved.
*   **Layer Compatibility**: OPT uses standard `nn.Linear` layers for its projections (`q_proj`, `v_proj`), making it an ideal candidate for testing our `LoRALinear` wrapper. We avoided models like GPT-2 because they often use `Conv1D` layers, which introduce unnecessary complexity for a baseline verification.

**What we verified:**
1.  **Injection**: `inject_lora` successfully locates target layers and swaps them.
2.  **Parameter Isolation**: Verified that only ~0.12% of parameters were trainable, confirming the "freezing" logic worked.
3.  **Checkpoint Plumbing**: Confirmed we could save *only* the adapter weights (~3MB) and reload them to reproduce identical results.

#### 2. Performance Benchmarking (`benchmarks/train.py`)
**Model Choice:** `distilbert-base-uncased` (Masked Language Model)
**Dataset:** SST-2 (GLUE Benchmark)
**Why DistilBERT + SST-2?**
*   **Standard Baseline**: Binary sentiment classification on SST-2 is a canonical benchmark for NLP transfer learning.
*   **Speed**: DistilBERT is lightweight, allowing for rapid iteration to prove that LoRA converges to similar accuracy as full fine-tuning with a fraction of the memory.

## Analysis & Results

### What exactly did we do with LoRA?
We implemented **Low-Rank Adaptation (LoRA)**, a method to fine-tune large pre-trained models by injecting trainable rank decomposition matrices into each layer of the Transformer architecture.

Instead of updating the entire pre-trained weight matrix $W$, we freeze $W$ and constrain the update $\Delta W$ by representing it as the product of two low-rank matrices $A$ and $B$:

$$
W_{new} = W + \Delta W = W + BA
$$

where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$, and the rank $r \ll \min(d, k)$.


### Benchmarking Highlights
1.  **Inference Latency**:
    *   **Baseline**: ~3.0 ms
    *   **Unmerged LoRA**: ~3.5 ms (slight overhead from separate branches)
    *   **Merged LoRA**: ~2.7 ms (mathematically identical to baseline, slight varience due to noise)
    *   *Takeaway*: Merging weights (`lora_layer.merge()`) allows us to deploy fine-tuned models with **zero inference latency penalty**.

2.  **Training Efficiency**:
    *   **Full Fine-Tuning**: Updates 100% of parameters (~66M for DistilBERT). High VRAM usage for optimizer states.
    *   **LoRA**: Updates <1% of parameters (~0.1M). Drastically reduced VRAM usage, allowing for larger batch sizes or training on consumer hardware.

### Summary: LoRA vs. Full Fine-Tuning

| Feature | Full Fine-Tuning | LoRA |
| :--- | :--- | :--- |
| **Parameters Updated** | 100% | < 1% (e.g., 0.12%) |
| **VRAM Usage** | Very High | Very Low |
| **Storage per Checkpoint** | Full Model Size (GBs) | Adapter Size (MBs) |
| **Training Speed** | Baseline | Faster |
| **Accuracy** | Baseline (High) | Comparable (High) |

## Citation

If you find this code useful, please cite the original LoRA paper:

```bibtex
@inproceedings{hu2022lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=nZeVKeeFYf9}
}
```
