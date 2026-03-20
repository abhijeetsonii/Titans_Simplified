import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm


class TitansTrainer:

    def __init__(self, model, dataloader, device="cuda", lr=3e-4):

        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device

        self.optimizer = AdamW(model.parameters(), lr=lr)

        self.scaler = torch.cuda.amp.GradScaler()

    def train_epoch(self):

        self.model.train()

        states = None
        total_loss = 0

        pbar = tqdm(self.dataloader)

        for batch in pbar:

            input_ids = batch["input_ids"].to(self.device)

            labels = input_ids.clone()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):

                logits, states = self.model(input_ids, states)

                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1)
                )

            self.scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.optimizer.zero_grad()

            total_loss += loss.item()

            # detach memory states
            if states is not None:
                states = [s.detach() if s is not None else None for s in states]

            pbar.set_description(f"loss {loss.item():.4f}")

        return total_loss / len(self.dataloader)