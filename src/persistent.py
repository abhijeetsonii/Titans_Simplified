import torch
import torch.nn as nn
from config import TitansConfig

class PersistentMemory(nn.Module):
    """Persistent memory module that are learnable data independent parameters
    intialized with a normal distribution with a standard deviation of 0.02.

    Args:
        num_tokens (int): The number of persistent memory tokens.
        token_dim (int): The dimensionality of each persistent memory token.
    """

    def __init__(self, config: TitansConfig):
        super().__init__()
        self.config = config
        self.num_tokens = config.num_persistent_tokens
        self.token_dim = config.dim

        self.memory_tokens = nn.Parameter(
            torch.randn(self.num_tokens, self.dim) * 0.02
        )

    def forward(self, batch_size: int) -> torch.Tensor:
        """Returns the persistent memory tokens expanded to the batch size.

        Args:
            batch_size (int): The batch size for which to expand the persistent memory tokens.

        Returns:
            torch.Tensor: The expanded persistent memory tokens of shape (batch_size, num_tokens, token_dim).
        """
        return self.memory_tokens.unsqueeze(0).expand(batch_size, -1, -1)
    
    def get_tokens(self) -> torch.Tensor:
        """Returns the persistent memory tokens.

        Returns:
            torch.Tensor: The persistent memory tokens of shape (num_tokens, token_dim).
        """
        
        return self.memory_tokens
    
