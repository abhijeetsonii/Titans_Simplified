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
# DATASET & TOKENIZATION (Fixed for Dense Training)
# ------------------------------------------------

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
print("Dataset loaded")

# 1. Pre-filter: Remove empty lines and headers to reduce noise
dataset = dataset.filter(lambda x: len(x["text"]) > 5)

def tokenize_function(examples):
    # Tokenize without padding or truncation here
    return tokenizer(examples["text"])

# 2. Map the raw tokenization
tokenized_datasets = dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=["text"],
    desc="Running tokenizer on dataset"
)

def group_texts(examples):
    # Concatenate all input_ids from the batch
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # We drop the small remainder at the end of the dataset
    # block_size should match your model's max sequence length
    block_size = 2048 
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
        
    # Split into chunks of block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    
    # In Causal LM, labels are usually just a copy of input_ids 
    # (The trainer handles the shifting)
    result["labels"] = result["input_ids"].copy()
    return result

    # 3. Group everything into dense 2048-token blocks
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts into chunks of 2048"
    )

    lm_datasets.set_format(type="torch", columns=["input_ids", "labels"])
    print("Dataset tokenized, grouped, and formatted")

    # ------------------------------------------------
    # DATALOADER
    # ------------------------------------------------

    train_loader = DataLoader(
        lm_datasets["train"],
        batch_size=4, # Reduced to 4 for 2048 seq_len on a 4060 (24GB)
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
    