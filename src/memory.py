"""
Neural Long-term Memory Module for Titans.

This module implements the core innovation of Titans: a neural memory that
learns to memorize at test time using gradient descent with momentum and
weight decay. The memory is trained with an associative memory loss to
learn key-value associations.

Key equations from the paper:
    Memory update: M_t = (1 - alpha_t) * M_{t-1} + S_t
    Surprise: S_t = eta_t * S_{t-1} - theta_t * grad(loss(M_{t-1}; x_t))
    Loss: loss(M; x) = ||M(k) - v||^2

where:
    - alpha_t: forgetting/decay factor (weight decay)
    - eta_t: surprise decay (momentum coefficient)
    - theta_t: learning rate for momentary surprise
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from config import TitansConfig

def get_activation(name: str) ->nn.Module:
    """Utility function to get activation function by name."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "silu" or name == "swish":
        return nn.SiLU()
    else:
        raise ValueError(f"Unsupported activation function: {name}")
    
@dataclass
class MemoryState:
    """State of the neural long-term memory.

    This encapsulates the memory weights and momentum for continuing
    inference across chunks/segments.

    Attributes:
        weights: List of weight matrices for each memory layer
        momentum: Accumulated surprise momentum (S_t in paper)
    """
    weights: list[torch.Tensor]  
    momentum: list[torch.Tensor]

    def detach(self) -> MemoryState:
        """Detach state from computation graph."""
        return MemoryState(
            weights=[w.detach() for w in self.weights],
            momentum=[m.detach() for m in self.momentum],
        )

    def clone(self) -> MemoryState:
        """Clone the memory state."""
        return MemoryState(
            weights=[w.clone() for w in self.weights],
            momentum=[m.clone() for m in self.momentum],
        )

class MemoryMLP(nn.Module):
    def __init__(self, config: TitansConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.num_memory_layers
        self.dim = config.dim
        self.hidden_dim = config.memory_hidden_dim

        self.layers = nn.ModuleList()
        if self.num_layers == 1:
            self.layers.append(nn.Linear(self.dim, self.dim, bias=False))
        else:
            self.layers.append(nn.Linear(self.dim, self.hidden_dim, bias=False))
            for _ in range(self.num_layers - 2):
                self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=False))
            self.layers.append(nn.Linear(self.hidden_dim, self.dim, bias=False))
        
        self.activation = get_activation(config.activation)
        self._init_weights()

    def _init_weights(self):
        for layer in self.layers:
            nn.init.normal_(layer.weight, std=0.02)

    def forward(self, x: torch.Tensor, weights: list[torch.Tensor] | None = None) -> torch.Tensor:
        """
        Unified forward pass. 
        If 'weights' are provided (Fast Weights), it uses the functional path.
        Otherwise, it uses the module's own parameters.
        """
        if weights is not None:
            return self.functional_forward(x, weights)
        
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

    def functional_forward(self, x: torch.Tensor, weights: list[torch.Tensor]) -> torch.Tensor:
        """
        The 'Meta-Learning' path. This uses F.linear to maintain the 
        computation graph connection to the gradients/projections.
        """
        h = x
        for i, w in enumerate(weights):
            # Crucial: Use F.linear with the specific weight tensor passed in
            h = F.linear(h, w, bias=None)
            if i < len(weights) - 1:
                h = self.activation(h)
        return h

    def compute_loss(self, keys: torch.Tensor, values: torch.Tensor, weights: list[torch.Tensor] | None = None) -> torch.Tensor:
        """
        Computes MSE loss. Can take optional weights to evaluate the 
        'Surprise' of the current memory state.
        """
        pred_values = self.forward(keys, weights=weights)
        return F.mse_loss(pred_values, values, reduction="mean")

    def get_weights(self) -> list[torch.Tensor]:
        # .detach() is used here only for the INITIAL state 
        # to prevent the optimizer from trying to update 'MemoryState' 
        # as a leaf node.
        return [layer.weight.detach().clone() for layer in self.layers]
    
class NeuralLongTermMemory(nn.Module):
    """Neural Long-term Memory Module.

    This is the main memory component of Titans. It learns to memorize
    at test time by treating training as an online learning problem.

    The memory is updated using gradient descent with:
    - Momentum (for past surprise)
    - Weight decay (for forgetting)

    Key features:
    1. Data-dependent learning rate, momentum, and decay
    2. Deep memory MLP for expressive power
    3. Surprise-based update rule
    """

    def __init__(self, config: TitansConfig):
        super().__init__()
        self.config = config
        self.memory = MemoryMLP(config)

        # 2-MLP projections for the keys and values as described in the Nested Learning paper
        self.key_proj = nn.Sequential(
            nn.Linear(config.dim, config.dim, bias = False),
            get_activation(config.activation),
            nn.Linear(config.dim, config.dim, bias = False)
        )
        self.value_proj = nn.Sequential(
            nn.Linear(config.dim, config.dim, bias = False),
            get_activation(config.activation),
            nn.Linear(config.dim, config.dim, bias = False)
        )
        self.query_proj = nn.Sequential(
            nn.Linear(config.dim, config.dim, bias = False),
            get_activation(config.activation),
            nn.Linear(config.dim, config.dim, bias = False)
        )

        # Optional 1D depthwise convolution (following Mamba2/GatedDeltaNet)
        self.use_conv = config.use_conv
        if self.use_conv:
            self.conv_k = nn.Conv1d(
                config.dim,
                config.dim,
                kernel_size=config.conv_kernel_size,
                padding=config.conv_kernel_size - 1,
                groups=config.dim,
            )
            self.conv_v = nn.Conv1d(
                config.dim,
                config.dim,
                kernel_size=config.conv_kernel_size,
                padding=config.conv_kernel_size - 1,
                groups=config.dim,
            )
            self.conv_q = nn.Conv1d(
                config.dim,
                config.dim,
                kernel_size=config.conv_kernel_size,
                padding=config.conv_kernel_size - 1,
                groups=config.dim,
            )

        # Data-dependent gates for learning parameters
        # These produce alpha_t (decay), theta_t (lr), eta_t (momentum)
        self.gate_decay = nn.Sequential(
            nn.Linear(config.dim, config.dim),
            nn.Sigmoid(),
        )
        self.gate_lr = nn.Sequential(
            nn.Linear(config.dim, config.dim),
            nn.Sigmoid(),
        )
        self.gate_momentum = nn.Sequential(
            nn.Linear(config.dim, config.dim),
            nn.Sigmoid(),
        )

        # Output projection
        self.proj_out = nn.Linear(config.dim, config.dim, bias=False)

        #Initialize memory state
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for module in [self.key_proj, self.value_proj, self.query_proj, self.proj_out]:
            module.apply(init_weights)

    def _apply_conv(
        self, k: torch.Tensor, v: torch.Tensor, q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply 1D convolution to K, V, Q."""
        if not self.use_conv:
            return k, v, q

        # Reshape for conv: (batch, seq, dim) -> (batch, dim, seq)
        k = rearrange(k, "b s d -> b d s")
        v = rearrange(v, "b s d -> b d s")
        q = rearrange(q, "b s d -> b d s")

        # Apply causal convolution
        k = self.conv_k(k)[..., : k.shape[-1]]
        v = self.conv_v(v)[..., : v.shape[-1]]
        q = self.conv_q(q)[..., : q.shape[-1]]

        # Reshape back: (batch, dim, seq) -> (batch, seq, dim)
        k = rearrange(k, "b d s -> b s d")
        v = rearrange(v, "b d s -> b s d")
        q = rearrange(q, "b d s -> b s d")

        return k, v, q
    
    def _compute_gradients(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        weights
    ) -> list[torch.Tensor]:
        """
        Compute gradients for memory update without breaking the computation graph.
        """
        # 1. DO NOT detach keys or values. They must remain connected 
        # to the projection layers (key_proj, value_proj).
        
        # 2. Use a context manager to ensure gradients are enabled
        with torch.enable_grad():
            # Compute the associative loss: loss(M; x) = ||M(k) - v||^2
            # Ensure this forward pass is differentiable
            loss = self.memory.compute_loss(keys, values)

            # 3. Compute gradients with create_graph=True
            # This allows us to backpropagate through these gradients later
            grads = torch.autograd.grad(
                loss,
                weights,
                create_graph=True,  # CRITICAL for Meta-Learning/Titans
                allow_unused=True,
            )

        # 4. Handle None gradients (standard safety)
        return [
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(grads, self.memory.parameters(), strict=True)
        ]
    
    def init_state(self, _batch_size: int, _device: torch.device) -> MemoryState:
        """Initialize memory state.

        Args:
            _batch_size: Batch size (reserved for future per-sample memory)
            _device: Device for tensors (reserved for future use)

        Returns:
            Initial memory state
        """
        # Initialize weights from the memory module
        weights = self.memory.get_weights()

        # Expand for batch dimension - weights are shared across batch
        # but we might want per-sample memory in some cases
        weights = [w.clone() for w in weights]

        # Initialize momentum to zeros
        momentum = [torch.zeros_like(w) for w in weights]

        return MemoryState(weights=weights, momentum=momentum)
    
    def forward(
        self,
        x: torch.Tensor,
        state: MemoryState | None = None,
        return_state: bool = True,
    ) -> tuple[torch.Tensor, MemoryState | None]:
        batch_size, seq_len, dim = x.shape
        device = x.device

        if state is None:
            state = self.init_state(batch_size, device)

        # 1. Projections (Keep these attached to the graph!)
        keys = self.key_proj(x) 
        values = self.value_proj(x)
        queries = self.query_proj(x)

        keys, values, queries = self._apply_conv(keys, values, queries)

        # Activations and Normalization
        keys = F.normalize(F.silu(keys), p=2, dim=-1)
        values = F.silu(values)
        queries = F.normalize(F.silu(queries), p=2, dim=-1)

        # 2. READ from current memory (Functional)
        # We use state.weights directly so the graph knows which version of memory we read
        retrieved = self.memory(queries, weights=state.weights)

        # 3. Compute Learning Gates
        x_mean = x.mean(dim=1, keepdim=True)
        alpha = self.gate_decay(x_mean).mean()
        theta = self.gate_lr(x_mean).mean() * self.config.memory_lr
        eta = self.gate_momentum(x_mean).mean() * self.config.memory_momentum

        # 4. WRITE/Update Memory (Differentiable)
        # Passing state.weights here ensures the loss is calculated against the right weights
        grads = self._compute_gradients(keys, values, weights=state.weights)

        new_weights, new_momentum = self._standard_memory_update(
                state.weights, state.momentum, grads, alpha, eta, theta
            )
        
        output = self.proj_out(retrieved)
        new_state = MemoryState(weights=new_weights, momentum=new_momentum)

        # 5. Gradient Truncation (Crucial for Training)
        # During training, if you are within a segment, DO NOT detach.
        # Only detach if you are passing the state to the NEXT independent segment.
        if self.training:
            return (output, new_state) 
        else:
            return (output, new_state.detach())

    def _standard_memory_update(
        self,
        weights: list[torch.Tensor],
        momentum: list[torch.Tensor],
        grads: list[torch.Tensor],
        alpha: torch.Tensor,
        eta: torch.Tensor,
        theta: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        
        new_momentum = []
        new_weights = []

        for w, m, g in zip(weights, momentum, grads, strict=True):
            # S_t = eta * S_{t-1} - theta * grad
            # These operations are now tracked by autograd
            s = (eta * m) - (theta * g)
            new_momentum.append(s)

            # M_t = (1 - alpha) * M_{t-1} + S_t
            # No .data or .copy_() here!
            new_w = ((1.0 - alpha) * w) + s
            new_weights.append(new_w)

        return new_weights, new_momentum
    
    def retrieve(
        self,
        queries: torch.Tensor,
        state: MemoryState,
    ) -> torch.Tensor:
        """Retrieve from memory without updating.

        Args:
            queries: Query vectors (batch, seq, dim)
            state: Memory state to query

        Returns:
            Retrieved values (batch, seq, dim)
        """
        # Set memory weights from state
        # self.memory.set_weights(state.weights)

        #project the queries using the query projection
        queries = self.query_proj(queries)
        
        #apply convolution if enabled
        if self.use_conv:
            queries = rearrange(queries, "b s d -> b d s")
            queries = self.conv_q(queries)[..., : queries.shape[-1]]
            queries = rearrange(queries, "b d s -> b s d")
        
        #apply activation
        queries = F.silu(queries)
        
        #Normalize queries for stability
        queries = F.normalize(queries, p=2, dim=-1)
        
        # Retrieve from memory using queries
        retrieved = self.memory(queries, weights = state.weights)
        
        #output projection
        output = self.proj_out(retrieved)
        return output