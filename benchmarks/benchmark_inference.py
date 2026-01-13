import torch
import time
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from lora_from_scratch.model import inject_lora

def benchmark_inference(model, input_ids, attention_mask, n_steps=100, label=""):
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_ids, attention_mask=attention_mask)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(n_steps):
        with torch.no_grad():
            _ = model(input_ids, attention_mask=attention_mask)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / n_steps * 1000 # ms
    print(f"[{label}] Average Latency: {avg_latency:.4f} ms")
    return avg_latency

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CRITICAL: CUDA is not available. This benchmark requires a GPU.")
    
    device = "cuda"
    print(f"Benchmarking on: {torch.cuda.get_device_name(0)}")
    
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Dummy Input
    text = "This is a sample sentence to benchmark the inference latency of the model."
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # 1. Baseline Model
    baseline_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    baseline_model.eval()
    base_lat = benchmark_inference(baseline_model, input_ids, attention_mask, label="Baseline (No LoRA)")
    
    # 2. LoRA (Unmerged)
    lora_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    inject_lora(lora_model, target_modules=["q_lin", "v_lin"], rank=8, alpha=16, verbose=False)
    lora_model.to(device)
    lora_model.eval()
    
    # Ensure it's unmerged
    for m in lora_model.modules():
        if hasattr(m, 'unmerge'):
            m.unmerge()
            
    unmerged_lat = benchmark_inference(lora_model, input_ids, attention_mask, label="LoRA (Unmerged)")
    
    # 3. LoRA (Merged)
    # Merge weights
    count = 0
    for m in lora_model.modules():
        if hasattr(m, 'merge'):
            m.merge()
            count += 1
    print(f"Merged {count} LoRA layers.")
    
    merged_lat = benchmark_inference(lora_model, input_ids, attention_mask, label="LoRA (Merged)")
    
    print("\n--- Summary ---")
    print(f"Baseline:      {base_lat:.2f} ms")
    print(f"LoRA Unmerged: {unmerged_lat:.2f} ms (+{(unmerged_lat-base_lat)/base_lat:.1%} overhead)")
    print(f"LoRA Merged:   {merged_lat:.2f} ms (+{(merged_lat-base_lat)/base_lat:.1%} overhead)")

if __name__ == "__main__":
    main()
