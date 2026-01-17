import torch
import torch.nn as nn
from torch.nn import functional as F


def time_embedding_fun(
    timestep: torch.Tensor,
    time_embedding_dim: int = 320,
) -> torch.Tensor:
    """
    Computes time embedding using sinusoidal positional encoding.

    Args:
        timestep: Input timestep tensor of shape (B,) or (B, 1) (batch of timesteps)
        time_embedding_dim: Embedding dimension (must be even number)

    Returns:
        Time embedding tensor of shape (B, time_embedding_dim)

    Process Flow:
        1. Create exponential frequency factors
        2. Compute sinusoidal and cosinudoidal components
        3. Concatenate components along embedding dimension
    """
    # Validate even embedding dimension for symmetric sin/cos split
    assert (
        time_embedding_dim % 2 == 0
    ), "time embedding dimension must be divisible by 2"

    timestep = timestep.reshape(-1).to(dtype=torch.float32)
    # Create frequency scaling factors: 10000^(2i/d_model) for i in [0, d_model/2)
    # Generates (time_embedding_dim//2,) shaped tensor of frequency divisors
    factor = 10000 ** (
        (
            torch.arange(  # Create [0, 1, 2,..., (d_model/2 - 1)] tensor
                start=0,
                end=time_embedding_dim // 2,
                dtype=torch.float32,
                device=timestep.device,  # Match device with input tensor
            )
            / (time_embedding_dim // 2)  # Normalize to [0, 1] range
        )
    )

    # Compute base embedding components
    # timestep shape: (B,) -> (B, 1) -> (B, d_model//2) via broadcasting
    t_emb = timestep[:, None] / factor

    # Combine sinusoidal components and return
    # Concatenate along last dimension: (B, d_model//2 * 2) -> (B, d_model)
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


class TimeEmbedding(nn.Module):
    def __init__(self, n_embd: int = 128, use_direct_map: bool = True):
        super().__init__()
        if use_direct_map:
            self.linear_1 = nn.Linear(n_embd, n_embd)
            self.linear_2 = nn.Linear(n_embd, n_embd)
        else:
            self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
            self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, n_embd)

        # (batch_size, n_embd) -> (batch_size, hidden_dim)
        x = self.linear_1(x)

        # (batch_size, hidden_dim) -> (batch_size, hidden_dim)
        x = F.silu(x)

        # (batch_size, hidden_dim) -> (batch_size, hidden_dim)
        x = self.linear_2(x)

        return x
