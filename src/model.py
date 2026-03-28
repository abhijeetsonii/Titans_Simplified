"""
Titans Model Architectures.

This module implements the MAC variants of Titans:

MAC (Memory as Context): Memory retrieval concatenated with input before attention

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import SegmentedAttention
from config import TitansConfig
from memory import MemoryState, NeuralLongTermMemory
from persistent import PersistentMemory


class FeedForward(nn.Module):
    """Feed-forward network with gating (following recent architectures).

    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.dim = config.dim
        self.hidden_dim = config.ffn_dim

        self.gate_proj = nn.Linear(config.dim, config.ffn_dim, bias=False)
        self.up_proj = nn.Linear(config.dim, config.ffn_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with SiLU gating."""
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        
        hidden = F.silu(gate) * up

        hidden = self.dropout(hidden)
        return self.down_proj(hidden)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    Supports fused residual add + norm for efficiency.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization."""
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

    def forward_with_residual(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused residual add + RMS normalization.

        Args:
            x: Input tensor
            residual: Residual tensor to add

        Returns:
            Tuple of (hidden, normalized) where hidden = x + residual
        """
        hidden = x + residual
        rms = torch.sqrt(torch.mean(hidden**2, dim=-1, keepdim=True) + self.eps)
        return hidden, hidden / rms * self.weight


# =============================================================================
# MAC: Memory as Context
# =============================================================================


class MACBlock(nn.Module):
    """Memory as Context Block.

    Architecture:
    1. Retrieve from long-term memory using input as query
    2. Concatenate: [persistent] || [memory] || [input]
    3. Apply segmented attention
    4. Feed-forward network

    At test time:
    - Persistent memory parameters are fixed
    - Attention performs in-context learning
    - Long-term memory continues learning (weight updates)
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        # Long-term memory
        self.memory = NeuralLongTermMemory(config)

        # Persistent memory
        self.persistent = PersistentMemory(config)

        # Segmented attention (Core module)
        self.attention = SegmentedAttention(config)

        # Feed-forward
        self.ffn = FeedForward(config)

        # Layer norms
        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)
        self.norm_mem = RMSNorm(config.dim)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: MemoryState | None = None,
    ) -> tuple[torch.Tensor, MemoryState]:
        batch_size, seq_len, _ = x.shape

        if state is None:
            state = self.memory.init_state(batch_size, x.device)

        # Pre-Norm for Attention and Memory Read
        normed_x = self.norm1(x)

        # Step 1: Retrieve from memory
        memory_retrieved = self.memory.retrieve(normed_x, state)
        memory_tokens = self.norm_mem(memory_retrieved)

        # Step 2: Attention
        persistent = self.persistent(batch_size)
        attn_out = self.attention(normed_x, persistent=persistent, memory=memory_tokens)
        
        # y_t is the residual stream after attention
        y_t = x + self.dropout(attn_out) 

        # Step 3: Update memory with updated context
        # We use the normed version for the inner-loop update for stability
        _, new_state = self.memory(self.norm2(y_t), state=state)

        # Step 4: Final output with Gated Memory Read (Eq. 25)
        # Use a new norm or norm2 to read from the updated memory
        mem_out = self.memory.retrieve(self.norm2(y_t), new_state)
        
        # Apply the gated update as a residual to keep the path alive
        gated_output = y_t * torch.sigmoid(mem_out) # Use sigmoid for gating stability
        
        # Step 5: Feed-forward
        ffn_out = self.ffn(self.norm2(gated_output))
        output = gated_output + self.dropout(ffn_out)

        return output, new_state


class TitansMAC(nn.Module):
    """Titans with Memory as Context.

    Segments the sequence into chunks and processes each with MAC blocks.
    Long-term memory persists across chunks within a sequence.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.dim)

        # Stack of MAC blocks
        self.blocks = nn.ModuleList(
            [MACBlock(config) for _ in range(config.num_layers)]
        )

        # Output normalization and head
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Tie weights
        self.head.weight = self.embed.weight

        # Initialize
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.normal_(self.embed.weight, std=self.config.init_std)

def forward(
        self,
        input_ids: torch.Tensor,
        states: list[MemoryState] | None = None,
    ) -> tuple[torch.Tensor, list[MemoryState]]:
        batch_size, seq_len = input_ids.shape
        chunk_size = self.config.chunk_size

        # 1. Initialize states if needed
        if states is None:
            states = [None] * len(self.blocks)

        # 2. Embed
        x = self.embed(input_ids)

        # 3. Process in chunks
        outputs = []
        
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunk = x[:, chunk_start:chunk_end]

            current_chunk_updated_states = [] 
            
            for i, block in enumerate(self.blocks):
                # Pass the state from the PREVIOUS chunk for this layer
                chunk, new_state = block(chunk, state=states[i])
                current_chunk_updated_states.append(new_state)

            outputs.append(chunk)
            
            # CRITICAL: Transfer these states so the NEXT chunk can use them
            states = current_chunk_updated_states 

        # 4. Concatenate outputs
        x = torch.cat(outputs, dim=1)

        # 5. Final Head
        x = self.norm(x)
        logits = self.head(x)

        # 6. Return the LAST states calculated (from the final chunk)
        return logits, states



