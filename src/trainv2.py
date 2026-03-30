"""Training script with checkpoint management and TensorBoard logging."""

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
import os

from model import TitansMAC
from config import TitansConfig
from trainer import TitansTrainer

# ------------------------------------------------
# GPU settings (important for H100)
# ------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"


def main(use_checkpoint=True):
    # ------------------------------------------------
    # Setup directories and logging
    # ------------------------------------------------
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    log_dir = "runs/trainv2"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs: {log_dir}")
    
    # 1. MODEL CONFIG
    config = TitansConfig(
        vocab_size=50257,
        dim=768,
        num_layers=8,
        chunk_size=512,
        dropout=0.1
    )
    print("Config done")

    model = TitansMAC(config)
    if use_checkpoint:
        checkpoint_path = "titans_mac_padded.pt"
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    print("Model initialized")

    # 2. TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded")

    # 3. DATASET & TOKENIZATION (Dense training with padding + ignore_index)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    print("Dataset loaded")

    # Pre-filter: Remove empty lines and headers to reduce noise
    dataset = dataset.filter(lambda x: len(x["text"]) > 5)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Running tokenizer on dataset"
    )

    # Set labels, marking pad tokens as -100 for loss ignore
    def label_function(examples):
        labels = []
        pad_id = tokenizer.pad_token_id
        for input_ids in examples["input_ids"]:
            labels.append([token_id if token_id != pad_id else -100 for token_id in input_ids])
        examples["labels"] = labels
        return examples

    tokenized_datasets = tokenized_datasets.map(
        label_function,
        batched=True,
        desc="Setting labels with ignore_index for padding"
    )

    tokenized_datasets.set_format(type="torch", columns=["input_ids", "labels"])
    print("Dataset tokenized, padded, and formatted")

    # 4. DATALOADER
    train_loader = DataLoader(
        tokenized_datasets["train"],
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        tokenized_datasets["validation"],
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print("DataLoader created")

    # 5. TRAINER
    trainer = TitansTrainer(
        model=model,
        dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
        lr=3e-4
    )
    print("Trainer initialized")

    # 6. TRAINING LOOP with best model tracking
    EPOCHS = 15
    best_val_loss = float("inf")
    
    for epoch in range(EPOCHS):
        loss = trainer.train_epoch()
        val_loss = trainer.validate()
        
        print(f"\nEpoch {epoch} Loss: {loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}\n")
        
        # TensorBoard logging
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"✓ Best model saved: {best_checkpoint_path} (val_loss: {val_loss:.4f})")
        
        # Save latest checkpoint every epoch
        latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
        torch.save(model.state_dict(), latest_checkpoint_path)

    print("Training loop completed")
    
    # 7. SAVE FINAL MODEL
    final_model_path = "titans_mac_padded.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved: {final_model_path}")
    
    # Close TensorBoard writer
    writer.close()
    print(f"TensorBoard logs saved to: {log_dir}")


if __name__ == "__main__":
    main(use_checkpoint=False)
