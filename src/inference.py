"""Generator class for the inference phase of Titan."""

import torch
import torch.nn.functional as F


class TitansFastGenerator:

    def __init__(self, model, device="cuda"):

        self.model = model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=100,
        temperature=1.0,
        top_k=50,
        eos_token_id=None
    ):

        input_ids = input_ids.to(self.device)

        # ------------------------------------------------
        # 1. Process the prompt once
        # ------------------------------------------------

        logits, states = self.model(input_ids)

        generated = input_ids

        for _ in range(max_new_tokens):

            next_logits = logits[:, -1, :] / max(temperature, 1e-7)

            # Top-k sampling
            if top_k > 0:
                values, _ = torch.topk(next_logits, top_k)
                min_val = values[:, -1].unsqueeze(-1)
                next_logits[next_logits < min_val] = -float("inf")

            probs = F.softmax(next_logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)

            # Append token
            generated = torch.cat([generated, next_token], dim=1)

            # ------------------------------------------------
            # 2. Only run model on the NEW token
            # ------------------------------------------------

            logits, states = self.model(next_token, states=states)

            # Detach memory states to avoid graph growth
            if states is not None:
                states = [
                    s.detach() if s is not None else None
                    for s in states
                ]

            # Stop if EOS
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return generated