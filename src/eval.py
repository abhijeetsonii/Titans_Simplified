import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

from model import TitansMAC
from config import TitansConfig


def prepare_wikitext_dataset(tokenizer, max_length=512):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    # Remove empty lines and headers
    dataset = dataset.filter(lambda x: len(x["text"]) > 5)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing dataset"
    )

    # Set labels, marking pad tokens as -100 for loss ignore
    def label_function(examples):
        labels = []
        pad_id = tokenizer.pad_token_id
        for input_ids in examples["input_ids"]:
            labels.append([token_id if token_id != pad_id else -100 for token_id in input_ids])
        examples["labels"] = labels
        return examples

    lm_datasets = tokenized.map(
        label_function,
        batched=True,
        desc="Setting labels with ignore_index for padding"
    )

    lm_datasets.set_format(type="torch", columns=["input_ids", "labels"])
    return lm_datasets


def make_dataloader(dataset_split, batch_size=4, shuffle=False):
    return DataLoader(
        dataset_split,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
    )


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            inputs = input_ids[:, :-1]
            labels = input_ids[:, 1:]

            logits, _ = model(inputs)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, perplexity


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = TitansConfig(
        vocab_size=50257,
        dim=768,
        num_layers=8,
        chunk_size=512,
        dropout=0.1,
    )

    model = TitansMAC(config)
    checkpoint_path = "titans_mac_padded.pt"
    print("checkpoint_path:", checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading and preparing Wikitext dataset...")
    lm_datasets = prepare_wikitext_dataset(tokenizer, max_length=512)

    val_loader = make_dataloader(lm_datasets["validation"], batch_size=4, shuffle=False)
    test_loader = make_dataloader(lm_datasets["test"], batch_size=4, shuffle=False)

    print("Evaluating on validation set...")
    val_loss, val_ppl = evaluate(model, val_loader, device)
    print(f"Validation loss: {val_loss:.4f}, perplexity: {val_ppl:.4f}")

    print("Evaluating on test set...")
    test_loss, test_ppl = evaluate(model, test_loader, device)
    print(f"Test loss: {test_loss:.4f}, perplexity: {test_ppl:.4f}")


if __name__ == "__main__":
    main()
