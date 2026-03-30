"""
Attention module for the Titans model.
Implements Segmented Attention for the MAC (Memory as Context) variant.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import TitansConfig
from einops import rearrange


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(
        self, dim: int, max_seq_len: int = 8192, base: float = 10000.0
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(
        self, seq_len: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        inv_freq = self.inv_freq
        if device is not None:
            inv_freq = inv_freq.to(device)

        positions = torch.arange(seq_len, device=inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq.float())

        cos = freqs.cos()
        sin = freqs.sin()

        if dtype is not None:
            cos = cos.to(dtype)
            sin = sin.to(dtype)

        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        self._cache_dtype = dtype
        self._cache_device = device

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[2]
        device = q.device

        need_rebuild = (
            seq_offset + seq_len > self.cos_cached.shape[0]
            or getattr(self, "_cache_device", None) != device
            or getattr(self, "_cache_dtype", None) != q.dtype
        )
        if need_rebuild:
            self._build_cache(max(seq_offset + seq_len, self.max_seq_len), device, q.dtype)

        cos = self.cos_cached[seq_offset : seq_offset + seq_len]
        sin = self.sin_cached[seq_offset : seq_offset + seq_len]

        q_rotated = self._apply_rotary(q, cos, sin)
        k_rotated = self._apply_rotary(k, cos, sin)

        return q_rotated, k_rotated

    def _apply_rotary(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x1, x2 = x[..., ::2], x[..., 1::2]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        rotated = torch.stack(
            [x1 * cos - x2 * sin, x1 * sin + x2 * cos],
            dim=-1,
        )
        return rotated.flatten(-2)


class SegmentedAttention(nn.Module):
    """Segmented Attention for MAC variant."""

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        self.proj_q = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_k = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_v = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_out = nn.Linear(config.dim, config.dim, bias=False)

        self.rope = None
        if config.use_rope:
            self.rope = RotaryPositionEmbedding(
                dim=config.head_dim,
                max_seq_len=config.max_seq_len,
            )

        self.dropout = nn.Dropout(config.dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj_out]:
            nn.init.normal_(module.weight, std=self.config.init_std)

    def _get_segmented_mask(
        self, 
        full_len: int, 
        prefix_len: int, 
        device: torch.device
    ) -> torch.Tensor:
        """
        Creates a mask for: [Prefix (Persistent + Memory)] || [Input Chunk]
        Prefix tokens see each other; Input tokens see Prefix + past Input.
        """
        mask = torch.ones((full_len, full_len), device=device, dtype=torch.bool)
        
        # Apply causal masking only to the 'Input' section (bottom-right square)
        input_len = full_len - prefix_len
        causal_mask = torch.tril(torch.ones((input_len, input_len), device=device, dtype=torch.bool))
        
        # Zero out future tokens in the input section
        mask[prefix_len:, prefix_len:] = causal_mask
        
        # Convert to float mask for SDPA if necessary, or stay bool for newer versions
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        x: torch.Tensor,
        persistent: torch.Tensor | None = None,
        memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Build full sequence components
        components = []
        if persistent is not None:
            components.append(persistent)
        if memory is not None:
            components.append(memory)
        
        prefix_len = sum(c.shape[1] for c in components)
        components.append(x)

        full_x = torch.cat(components, dim=1)
        full_len = full_x.shape[1]

        # Project Q, K, V
        q = rearrange(self.proj_q(full_x), "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(self.proj_k(full_x), "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(self.proj_v(full_x), "b s (h d) -> b h s d", h=self.num_heads)

        # Apply RoPE
        if self.rope is not None:
            q_prefix, q_main = q[:, :, :prefix_len], q[:, :, prefix_len:]
            k_prefix, k_main = k[:, :, :prefix_len], k[:, :, prefix_len:]
            q_main, k_main = self.rope(q_main, k_main, seq_offset=0)
            q = torch.cat([q_prefix, q_main], dim=2)
            k = torch.cat([k_prefix, k_main], dim=2)

        # Build the Segmented Mask
        mask = self._get_segmented_mask(full_len, prefix_len, x.device)

        # DETECT NESTED LOOP: If create_graph=True is active, we must use Manual Math
        is_nested_loop = torch.is_grad_enabled() and any(p.requires_grad for p in [q, k, v])

        if is_nested_loop:
            # MANUAL MATH PATH: Fully differentiable for higher-order gradients
            # (Q @ K.T) * Scale
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            # Apply Mask (mask is boolean, True means keep)
            attn_weights = attn_weights.masked_fill(~mask, float("-inf"))
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            output = torch.matmul(attn_weights, v)
        else:
            # OPTIMIZED PATH: Use Flash Attention for the main training pass
            from torch.nn.attention import SDPBackend
            backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
            
            with torch.nn.attention.sdpa_kernel(backends):
                output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=False,
                    scale=self.scale,
                )

        # Reshape back
        output = rearrange(output, "b h s d -> b s (h d)")
        output = self.proj_out(output)

        return output[:, prefix_len:]