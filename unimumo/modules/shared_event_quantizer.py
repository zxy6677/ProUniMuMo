import typing as tp
import torch
import torch.nn as nn

from unimumo.audio.audiocraft_.quantization.vq import ResidualVectorQuantizer


class SharedEventQuantizer(nn.Module):
    """
    Adapter around ResidualVectorQuantizer for sparse event embeddings.

    Input:
        sparse_event_embeds: [B, K_event, D]
    Output:
        dict:
            x_q:    [B, K_event, D]      quantized sparse event embeddings
            codes:  [B, n_q, K_event]    discrete shared event codes
            penalty: scalar tensor        commitment penalty
            bandwidth: scalar tensor
    """
    def __init__(
        self,
        event_dim: int = 128,
        n_q: int = 4,
        bins: int = 1024,
        q_dropout: bool = False,
        decay: float = 0.99,
        kmeans_init: bool = True,
        kmeans_iters: int = 10,
        threshold_ema_dead_code: int = 2,
        orthogonal_reg_weight: float = 0.0,
        orthogonal_reg_active_codes_only: bool = False,
        orthogonal_reg_max_codes: tp.Optional[int] = None,
        freeze_codebook: bool = False,
        default_frame_rate: int = 50,
    ):
        super().__init__()
        self.default_frame_rate = default_frame_rate

        self.rvq = ResidualVectorQuantizer(
            dimension=event_dim,
            n_q=n_q,
            q_dropout=q_dropout,
            bins=bins,
            decay=decay,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            threshold_ema_dead_code=threshold_ema_dead_code,
            orthogonal_reg_weight=orthogonal_reg_weight,
            orthogonal_reg_active_codes_only=orthogonal_reg_active_codes_only,
            orthogonal_reg_max_codes=orthogonal_reg_max_codes,
        )
        self.rvq.freeze_codebook = freeze_codebook

    def forward(
        self,
        sparse_event_embeds: torch.Tensor,
        frame_rate: tp.Optional[int] = None,
    ) -> dict:
        """
        sparse_event_embeds: [B, K_event, D]
        """
        assert sparse_event_embeds.dim() == 3, f"Expected [B, K_event, D], got {sparse_event_embeds.shape}"

        x = sparse_event_embeds.transpose(1, 2).contiguous()  # [B, D, K_event]
        q = self.rvq(x, frame_rate or self.default_frame_rate)

        return {
            "x_q": q.x.transpose(1, 2).contiguous(),   # [B, K_event, D]
            "codes": q.codes,                          # [B, n_q, K_event]
            "penalty": q.penalty,
            "bandwidth": q.bandwidth,
        }

    @torch.no_grad()
    def encode(self, sparse_event_embeds: torch.Tensor) -> torch.Tensor:
        """
        sparse_event_embeds: [B, K_event, D]
        return codes: [B, n_q, K_event]
        """
        x = sparse_event_embeds.transpose(1, 2).contiguous()
        return self.rvq.encode(x)

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        codes: [B, n_q, K_event]
        return quantized sparse event embeddings: [B, K_event, D]
        """
        x_q = self.rvq.decode(codes)                   # [B, D, K_event]
        return x_q.transpose(1, 2).contiguous()       # [B, K_event, D]

    @property
    def num_codebooks(self) -> int:
        return self.rvq.num_codebooks

    @property
    def total_codebooks(self) -> int:
        return self.rvq.total_codebooks

    def set_num_codebooks(self, n: int):
        self.rvq.set_num_codebooks(n)