import torch
import torch.nn as nn


class SharedEventContext(nn.Module):
    """
    Build dense temporal context from sparse event embeddings and event indices.

    Inputs:
        event_embeds: [B, K, D]
        event_indices: [B, K]
        seq_len: int

    Output:
        context: [B, D, T]
    """
    def __init__(
        self,
        event_dim: int,
        sigma: float = 2.0,
        use_event_mlp: bool = True,
    ):
        super().__init__()
        self.sigma = float(sigma)

        if use_event_mlp:
            self.event_proj = nn.Sequential(
                nn.Linear(event_dim, event_dim),
                nn.GELU(),
                nn.Linear(event_dim, event_dim),
            )
        else:
            self.event_proj = nn.Identity()

    def forward(
        self,
        event_embeds: torch.Tensor,   # [B, K, D]
        event_indices: torch.Tensor,  # [B, K]
        seq_len: int,
    ) -> torch.Tensor:
        if event_embeds.dim() != 3:
            raise ValueError(f"event_embeds must be [B, K, D], got {event_embeds.shape}")
        if event_indices.dim() != 2:
            raise ValueError(f"event_indices must be [B, K], got {event_indices.shape}")

        b, k, d = event_embeds.shape
        device = event_embeds.device
        dtype = event_embeds.dtype

        # [B, K, D]
        event_feats = self.event_proj(event_embeds)

        # [1, 1, T]
        t = torch.arange(seq_len, device=device, dtype=dtype).view(1, 1, seq_len)

        # [B, K, 1]
        centers = event_indices.to(dtype).unsqueeze(-1).clamp(0, seq_len - 1)

        sigma = max(self.sigma, 1e-3)

        # Gaussian spread: [B, K, T]
        weights = torch.exp(-0.5 * ((t - centers) / sigma) ** 2)

        # Normalize across events so each timestep gets a weighted mixture
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        # [B, D, T]
        context = torch.einsum("bkt,bkd->bdt", weights, event_feats)
        return context