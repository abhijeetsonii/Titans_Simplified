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

class MemoryMLP (nn.Module):
    """MLP architecture for the neural memory.

    This is the actual memory module that stores information in its weights.
    It's a simple MLP that learns key-value associations.

    For L_M = 1 (linear memory), this is equivalent to a matrix-valued memory.
    For L_M >= 2 (deep memory), this provides more expressive power.
    """
    
    def __init__(self, config: TitansConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.num_memory_layers
        self.dim = config.dim
        self.hidden_dim = config.memory_hidden_dim

        self.layers = nn.ModuleList()

        if self.num_layers == 1:
            # Linear memory: single weight matrix
            self.layers.append(nn.Linear(self.dim, self.dim, bias=False))
        else:
            # dim to hidden_dim
            self.layers.append(nn.Linear(self.dim, self.hidden_dim, bias=False))

            for _ in range(self.num_layers - 2):
                self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=False))
            
            # hidden_dim to dim
            self.layers.append(nn.Linear(self.hidden_dim, self.dim, bias=False))
        
        self.activation = get_activation(config.activation)

        self._init_weights()

    def _init_weights(self):
        """Initialize memory weights with a normal distribution."""
        for layer in self.layers:
            nn.init.normal_(layer.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:    
        """Forward pass through memory MLP.

        Args:
            x: Input tensor of shape (batch, seq, dim)

        Returns:
            Output tensor of shape (batch, seq, dim)
        """
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            # Apply activation for all but last layer
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h
    
    def get_weights(self) -> list[torch.Tensor]:
        """Get the current memory weights as a list of tensors."""
        return [layer.weight.data.clone() for layer in self.layers]  
    
    def set_weights(self, weights: list[torch.Tensor]) -> None:
        """Set the memory weights from a list of tensors."""
        for layer, w in zip(self.layers, weights, strict=True):
            layer.weight.data.copy_(w)

    def compute_loss(self, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Compute associative memory loss for a batch of key-value pairs.

        Args:
            keys: Tensor of shape (batch, seq, dim) representing the keys.
            values: Tensor of shape (batch, seq, dim) representing the target values.
        Returns:
            Scalar tensor representing the mean squared error loss.
        """

        # Forward pass through memory to get predicted values
        pred_values = self.forward(keys)  # (batch, seq, dim)
        # Compute mean squared error loss
        loss = F.mse_loss(pred_values, values, reduction="mean")
        return loss
    
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
        self.memory_mlp = MemoryMLP(config)

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
    ) -> list[torch.Tensor]:
        """Compute gradients for memory update.

        This computes the gradient of the associative memory loss
        with respect to the memory weights.

        Args:
            keys: Key vectors (batch, seq, dim)
            values: Value vectors (batch, seq, dim)

        Returns:
            List of gradient tensors for each memory layer
        """
        # Use torch.enable_grad() to compute gradients even in inference mode
        # This is essential because Titans learns at test time
        with torch.enable_grad():
            # Enable gradients for weight computation
            for param in self.memory.parameters():
                param.requires_grad_(True)

            # Detach inputs and re-enable gradients for gradient computation
            keys_grad = keys.detach().requires_grad_(True)
            values_grad = values.detach()

            # Compute loss
            loss = self.memory.compute_loss(keys_grad, values_grad)

            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                list(self.memory.parameters()),
                create_graph=False,
                allow_unused=True,
            )

            # Disable gradients
            for param in self.memory.parameters():
                param.requires_grad_(False)

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
        """Forward pass through the neural long-term memory withh online learning(memory update)."""

        """Args:
            x: Input tensor of shape (batch, seq, dim)
            state: Current memory state (weights and momentum)
            return_state: Whether to return the updated memory state
            
            Returns:
            output: Tensor of shape (batch, seq, dim) after memory read"""

        batch_size, seq_len,dim = x.shape
        device = x.device

        # Initialize state if not provided
        if state is None:
            state = self.init_state(batch_size, device)

        # Set memory weights from state
        self.memory.set_weights(state.weights)

        #project input to keys and values
        keys = self.key_proj(x) 
        values = self.value_proj(x)
        queries = self.query_proj(x)

        #applying convolution as said in paper 
        keys, values, queries = self._apply_conv(keys, values, queries)

        #applying activation told in section 4.4 of the paper
        keys = F.silu(keys)
        values = F.silu(values)
        queries = F.silu(queries)

        #Normalize keys and queries for stability
        keys = F.normalize(keys,p=2, dim=-1)
        queries = F.normalize(queries, p=2, dim=-1)

        # Retrieve from memory using queries
        # y_t = M*(q_t) - forward pass without weight update
        retrieved = self.memory(queries)

        # Computing the data dependent learning parameters using the gates
        x_mean = x.mean(dim=1, keepdim=True)  # (batch, 1, dim)
        alpha = self.gate_decay(x_mean)  # scalar decay
        theta = self.gate_lr(x_mean) * self.config.memory_lr  # scalar lr
        eta = (
            self.gate_momentum(x_mean) * self.config.memory_momentum
        )  # scalar momentum

        # Compute gradients for memory update
        grads = self._compute_gradients(keys, values)

        new_weights, new_momentum = self._standard_memory_update(
                state.weights, state.momentum, grads, alpha, eta, theta
            )
        
        #output projection
        output = self.proj_out(retrieved)

        #create new state for the memory after the update
        new_state = MemoryState(weights=new_weights, momentum=new_momentum)

        return (output, new_state.detach()) if return_state else (output, None)

    def _standard_memory_update(
        self,
        weights: list[torch.Tensor],
        momentum: list[torch.Tensor],
        grads: list[torch.Tensor],
        alpha: torch.Tensor | float,
        eta: torch.Tensor | float,
        theta: torch.Tensor | float,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Standard memory update using PyTorch operations.

        Args:
            weights: Current weight tensors
            momentum: Current momentum tensors
            grads: Gradient tensors
            alpha: Decay factor
            eta: Momentum coefficient
            theta: Learning rate

        Returns:
            Tuple of (new_weights, new_momentum)
        """
        # Update momentum: S_t = eta * S_{t-1} - theta * grad
        new_momentum = []
        for m, g in zip( momentum, grads, strict=True):
            s = eta * m - theta * g
            new_momentum.append(s)

        #updating the weights using the update rule M_t = (1 - alpha) * M_{t-1} + S_t

        new_weights = []
        for w, s in zip(weights, new_momentum, strict=True):
            new_w = (1 - alpha) * w + s
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
        self.memory.set_weights(state.weights)

        #project the queries using the query projection
        queries = self.query_proj(queries)
        
        #apply convolution if enabled
        if self.use_conv:
            q = rearrange(q, "b s d -> b d s")
            q = self.conv_q(q)[..., : q.shape[-1]]
            q = rearrange(q, "b d s -> b s d")
        
        #apply activation
        queries = F.silu(queries)
        
        #Normalize queries for stability
        queries = F.normalize(queries, p=2, dim=-1)
        
        # Retrieve from memory using queries
        retrieved = self.memory(queries)
        
        #output projection
        output = self.proj_out(retrieved)
        return output