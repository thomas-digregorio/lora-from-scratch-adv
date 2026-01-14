import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lora_from_scratch.model import inject_lora, lora_state_dict, mark_only_lora_as_trainable
import tempfile
import os
import copy

def main():
    print("=== LoRA Integration & Checkpointing Verification ===\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "facebook/opt-125m" # OPT uses nn.Linear, unlike GPT-2's Conv1D
    
    print(f"1. Loading {model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Load model and keep a pristine copy to compare against later
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    
    # 2. Inject LoRA
    print("\n2. Injecting LoRA...")
    # Target 'q_proj', 'v_proj' which are standard Linear layers in OPT
    inject_lora(model, target_modules=["q_proj", "v_proj"], rank=4, alpha=16)
    mark_only_lora_as_trainable(model)
    
    # Verify parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total Params: {total_params:,}")
    print(f"   Trainable Params: {trainable_params:,}")
    print(f"   Trainable %: {trainable_params/total_params:.4%}")
    assert trainable_params < total_params, "LoRA injection failed: all params are trainable!"
    assert trainable_params > 0, "LoRA injection failed: no params are trainable!"
    print("   [PASS] LoRA Injection successful (parameters isolated).")

    # 3. Functional Test (Generation)
    print("\n3. Testing Generation (Forward Pass)...")
    prompt = "Hello, my name is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_lora = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    
    gen_text = tokenizer.decode(output_lora[0])
    print(f"   Generated: {gen_text.strip()}")
    print("   [PASS] Forward pass successful.")

    # 4. Checkpointing (Save/Load)
    print("\n4. Verifying Checkpointing...")
    
    # Simulate "training" by modifying one LoRA weight slightly
    # This ensures we aren't just saving zeros
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.add_(0.01) # Perturb B slightly
                break
    
    # Save adapters
    lora_weights = lora_state_dict(model)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "lora_adapters.pt")
        torch.save(lora_weights, save_path)
        print(f"   Saved adapter weights to {save_path} ({os.path.getsize(save_path)/1024:.2f} KB)")
        
        # Load a FRESH model
        print("   Loading fresh model for verification...")
        fresh_model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        inject_lora(fresh_model, target_modules=["q_proj", "v_proj"], rank=4, alpha=16, verbose=False)
        
        # Load weights
        print("   Loading saved LoRA weights...")
        fresh_model.load_state_dict(torch.load(save_path), strict=False)
        
        # Verify outputs match exactly
        print("   Verifying outputs match...")
        with torch.no_grad():
            output_fresh = fresh_model.generate(**inputs, max_new_tokens=10, do_sample=False)
            
        if torch.equal(output_lora, output_fresh):
            print("   [PASS] Saved/Loaded model output matches exactly.")
        else:
            print("   [FAIL] Outputs do not match!")
            print(f"   Original: {output_lora}")
            print(f"   Restored: {output_fresh}")
            
    print("\n=== Verification Complete ===")

if __name__ == "__main__":
    main()
