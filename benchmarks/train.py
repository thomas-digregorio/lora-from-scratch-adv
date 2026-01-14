import argparse
import time
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import wandb
import numpy as np
from sklearn.metrics import accuracy_score
from lora_from_scratch.model import inject_lora, mark_only_lora_as_trainable

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

def main():
    parser = argparse.ArgumentParser(description="Benchmark LoRA vs Full Fine-Tuning")
    parser.add_argument("--mode", type=str, choices=["full", "lora"], required=True, help="Training mode")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Num epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()

    # STRICT GPU ENFORCEMENT
    if not torch.cuda.is_available():
        raise RuntimeError("CRITICAL: CUDA is not available. Aborting to prevent CPU training.")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Init W&B
    wandb.init(project="lora-from-scratch-benchmarks", config=vars(args), name=f"{args.mode}_r{args.rank}")

    # Load Data
    print("Loading SST-2 dataset...")
    dataset = load_dataset("glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(examples["sentence"], truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Load Model
    print("Loading Model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )

    # Apply LoRA if needed
    if args.mode == "lora":
        print(f"Injecting LoRA (rank={args.rank})...")
        # Target Q, V, and Linear layers in MLP if desired. 
        # For DistilBERT: 'q_lin', 'v_lin', etc. 
        # Let's target main attention projections for standard LoRA practice.
        inject_lora(model, target_modules=["q_lin", "v_lin"], rank=args.rank, alpha=args.alpha)
        
        # Freeze non-LoRA params
        mark_only_lora_as_trainable(model)

    # Log Parameter Counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Stats:")
    print(f"Total Params: {total_params:,}")
    print(f"Trainable Params: {trainable_params:,}")
    print(f"Trainable %: {trainable_params/total_params:.4%}\n")
    
    wandb.log({
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percent": trainable_params/total_params
    })

    # Training Args
    training_args = TrainingArguments(
        output_dir=f"./results/{args.mode}",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="steps", # Evaluate more frequently than just per epoch
        eval_steps=200,      # Evaluate every 200 steps to see the curve
        save_strategy="no", # Save space during benchmarking
        logging_dir='./logs',
        logging_steps=10,
        report_to="wandb",
        fp16=torch.cuda.is_available(), # Use mixed precision if possible
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    print("Starting Training...")
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"Training completed in {duration:.2f} seconds")
    wandb.log({"training_time_seconds": duration})

    # Evaluate
    print("Evaluating...")
    eval_results = trainer.evaluate()
    print(f"Eval Accuracy: {eval_results['eval_accuracy']:.4f}")
    wandb.log({"final_eval_accuracy": eval_results['eval_accuracy']})
    
    wandb.finish()

if __name__ == "__main__":
    main()
