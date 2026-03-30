import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm


class TitansTrainer:

    def __init__(self, model, dataloader, val_dataloader, device="cuda", lr=3e-4):

        self.model = model.to(device)
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.device = device

        self.optimizer = AdamW(model.parameters(), lr=lr)

        self.scaler = torch.amp.GradScaler('cuda')

    def train_epoch(self):
        self.model.train()
        states = None
        total_loss = 0
        pbar = tqdm(self.dataloader)

        for batch in pbar:
            # 1. Prepare Inputs & Labels
            input_ids = batch["input_ids"].to(self.device)
            # Standard LM shifting: Input is [0...n-1], Label is [1...n]
            # If your model doesn't handle shifting internally, do it here:
            inputs = input_ids[:, :-1]
            labels = input_ids[:, 1:]  # Ensure labels are on the same device

            self.optimizer.zero_grad()

            # 2. Forward Pass with Bfloat16
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, states = self.model(inputs, states)

                # Reshape for CrossEntropy
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100 # Good practice
                )

            # 3. Backward Pass
            # Since you're using bfloat16, you can usually skip scaler.scale()
            # but if you want to keep it for compatibility:
            self.scaler.scale(loss).backward()

            # 4. Gradient Clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # 5. Step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            # 6. Correct State Detachment
            # Use the method defined in your dataclass to clear the graph
            if states is not None:
                states = [s.detach() for s in states]  # Detach each state tensor

            pbar.set_description(f"loss {loss.item():.4f}")

        return total_loss / len(self.dataloader)

    def validate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                input_ids = batch["input_ids"].to(self.device)
                inputs = input_ids[:, :-1]
                labels = input_ids[:, 1:]

                logits, _ = self.model(inputs)

                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100
                )

                total_loss += loss.item()

        return total_loss / len(self.val_dataloader)

    def evaluate(self):
        """Alias for evaluate from main() to keep API consistency."""
        return self.validate()