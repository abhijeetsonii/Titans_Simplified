from xml.parsers.expat import model

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from transformers import AutoTokenizer

from model import TitansMAC
from config import TitansConfig
from trainer import TitansTrainer

# ------------------------------------------------
# GPU settings (important for H100)
# ------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
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
    checkpoint_path = "titans_mac.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    print("Model initialized")

    # 2. TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded")

    # 3. DATASET & TOKENIZATION (Fixed for Dense Training)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    print("Dataset loaded")

    # Pre-filter: Remove empty lines and headers to reduce noise
    dataset = dataset.filter(lambda x: len(x["text"]) > 5)

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_datasets = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"],
        desc="Running tokenizer on dataset"
    )

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        block_size = 2048 
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
            
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        desc="Grouping texts into chunks of 2048"
    )

    lm_datasets.set_format(type="torch", columns=["input_ids", "labels"])
    print("Dataset tokenized, grouped, and formatted")

    # 4. DATALOADER
    train_loader = DataLoader(
        lm_datasets["train"],
        batch_size=1, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        lm_datasets["validation"],
        batch_size=1,
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

    # TensorBoard setup
    writer = SummaryWriter(log_dir="runs/titans_train")
    writer.add_text("config", str(config))

    # 6. TRAINING LOOP
    EPOCHS = 25
    best_val_loss = float("inf")
    best_checkpoint_path = "best_titans_mac.pt"

    for epoch in range(EPOCHS):
        loss = trainer.train_epoch()
        val_loss = trainer.evaluate()

        print(f"\nEpoch {epoch} Loss: {loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}\n")

        writer.add_scalar("loss/train", loss, epoch)
        writer.add_scalar("loss/validation", val_loss, epoch)

        # Optional additional metrics
        if hasattr(trainer, "optimizer"):
            lr = trainer.optimizer.param_groups[0]["lr"]
            writer.add_scalar("lr", lr, epoch)

        # Checkpointing on best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"New best model saved: {best_checkpoint_path} (val_loss={val_loss:.4f})")

    print("Training loop completed")

    writer.close()

    # 7. Final save for latest weights
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Training finished; final model saved: {checkpoint_path}")
    print(f"Best model saved: {best_checkpoint_path} with val_loss={best_val_loss:.4f}")

if __name__ == "__main__":
    main()
