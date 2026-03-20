import torch
from torch.utils.data import DataLoader
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


# ------------------------------------------------
# MODEL CONFIG
# ------------------------------------------------
def main():
    config = TitansConfig(
        vocab_size=50257,
        dim=768,
        num_layers=8,
        chunk_size=512,
        dropout=0.1
    )
    print("Config done")

    model = TitansMAC(config)
    print("Model initialized")

    # ------------------------------------------------
    # TOKENIZER
    # ------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded")

    # ------------------------------------------------
    # DATASET
    # ------------------------------------------------

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    print("Dataset loaded")

    def tokenize(example):

        tokens = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=2048
        )

        return {"input_ids": tokens["input_ids"]}


    dataset = dataset.map(tokenize, batched=True)

    dataset.set_format(type="torch", columns=["input_ids"])
    print("Dataset tokenized and formatted")

    # ------------------------------------------------
    # DATALOADER
    # ------------------------------------------------

    train_loader = DataLoader(
        dataset["train"],
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    print("DataLoader created")
    # ------------------------------------------------
    # TRAINER
    # ------------------------------------------------

    trainer = TitansTrainer(
        model=model,
        dataloader=train_loader,
        device=device,
        lr=3e-4
    )
    print("Trainer initialized")

    # ------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------

    EPOCHS = 3

    for epoch in range(EPOCHS):

        loss = trainer.train_epoch()

        print(f"\nEpoch {epoch} Loss: {loss:.4f}")
    print("Training loop completed")

    # ------------------------------------------------
    # SAVE MODEL
    # ------------------------------------------------

    torch.save(model.state_dict(), "titans_mac.pt")

    print("Training finished")

if __name__ == "__main__":
    main()