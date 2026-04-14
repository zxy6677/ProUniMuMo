import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _batched_gather_txd(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    x: [B, T, D]
    indices: [B, K]
    return: [B, K, D]
    """
    assert x.dim() == 3, f"Expected x as [B, T, D], got {x.shape}"
    assert indices.dim() == 2, f"Expected indices as [B, K], got {indices.shape}"
    indices = indices.long()
    idx = indices.unsqueeze(-1).expand(-1, -1, x.size(-1))
    return torch.gather(x, dim=1, index=idx)


def build_phase_encoding(phi: torch.Tensor, dim: int) -> torch.Tensor:
    """
    phi: [B, K], normalized to [0, 1]
    return: [B, K, dim]

    Use periodic sin/cos encoding so phase=0 and phase=1 are close.
    """
    device = phi.device
    dtype = phi.dtype

    half_dim = dim // 2
    if half_dim == 0:
        raise ValueError(f"phase encoding dim must be >= 2, got {dim}")

    freq = torch.arange(half_dim, device=device, dtype=dtype)
    if half_dim > 1:
        freq = 1.0 / (10000 ** (freq / (half_dim - 1)))
    else:
        freq = torch.ones_like(freq)

    angles = 2.0 * math.pi * phi.unsqueeze(-1) * freq.unsqueeze(0).unsqueeze(0)
    pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    if pe.size(-1) < dim:
        pe = F.pad(pe, (0, dim - pe.size(-1)))

    return pe


def scatter_sparse_tokens(tokens: torch.Tensor, indices: torch.Tensor, length: int) -> torch.Tensor:
    """
    tokens: [B, K, D]
    indices: [B, K]
    return: [B, D, T]

    Put sparse tokens back onto the frame-level timeline.
    Non-event positions are zero-filled.
    """
    b, k, d = tokens.shape
    dense = tokens.new_zeros(b, length, d)           # [B, T, D]
    idx = indices.unsqueeze(-1).expand(-1, -1, d)   # [B, K, D]
    dense.scatter_(dim=1, index=idx, src=tokens)
    return dense.transpose(1, 2).contiguous()       # [B, D, T]


class EventHead(nn.Module):
    """
    Input:
        x: [B, C, T]

    Output keys:
        content:                [B, T, D]
        saliency_logits:        [B, T]
        saliency_prob:          [B, T]
        phase:                  [B, T]
        role_logits:            [B, T, R]
        indices:                [B, K]
        sparse_content:         [B, K, D]
        sparse_saliency_logits: [B, K]
        sparse_saliency_prob:   [B, K]
        sparse_phase:           [B, K]
        sparse_role_logits:     [B, K, R]
        sparse_role_prob:       [B, K, R]
        sparse_event_embeds:    [B, K, D]   # continuous embeddings before RVQ
    """
    def __init__(
        self,
        input_dim: int,
        event_dim: int,
        hidden_dim: int = 256,
        phase_dim: int = 32,
        role_num: int = 4,
        topk: int = 8,
        window_topk: bool = False,
        window_size: int = 25,
        topk_per_window: int = 2,
    ):
        super().__init__()
        self.event_dim = event_dim
        self.phase_dim = phase_dim
        self.role_num = role_num
        self.topk = topk
        self.window_topk = window_topk
        self.window_size = window_size
        self.topk_per_window = topk_per_window

        self.content_proj = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 1),
            nn.ELU(),
            nn.Conv1d(hidden_dim, event_dim, 1),
        )

        self.saliency_head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 1),
            nn.ELU(),
            nn.Conv1d(hidden_dim, 1, 1),
        )

        self.phase_head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 1),
            nn.ELU(),
            nn.Conv1d(hidden_dim, 1, 1),
        )

        self.role_head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 1),
            nn.ELU(),
            nn.Conv1d(hidden_dim, role_num, 1),
        )

        self.role_embedding = nn.Embedding(role_num, event_dim)

        self.phase_proj = nn.Sequential(
            nn.Linear(phase_dim, event_dim),
            nn.ELU(),
            nn.Linear(event_dim, event_dim),
        )

        self.event_proj = nn.Sequential(
            nn.Linear(event_dim * 3, event_dim),
            nn.ELU(),
            nn.Linear(event_dim, event_dim),
        )

    def _select_event_indices(self, saliency_logits: torch.Tensor) -> torch.Tensor:
        """
        saliency_logits: [B, T]
        return:
            indices: [B, K]
        """
        b, t = saliency_logits.shape

        if not self.window_topk:
            k = min(self.topk, t)
            _, indices = torch.topk(saliency_logits, k=k, dim=1, largest=True, sorted=False)
            indices = torch.sort(indices, dim=1).values
            return indices

        if self.window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {self.window_size}")
        if self.topk_per_window <= 0:
            raise ValueError(f"topk_per_window must be > 0, got {self.topk_per_window}")

        all_indices = []
        for start in range(0, t, self.window_size):
            end = min(start + self.window_size, t)
            local_logits = saliency_logits[:, start:end]   # [B, Tw]
            local_k = min(self.topk_per_window, end - start)

            _, local_idx = torch.topk(
                local_logits,
                k=local_k,
                dim=1,
                largest=True,
                sorted=False,
            )
            all_indices.append(local_idx + start)

        indices = torch.cat(all_indices, dim=1)           # [B, K_total]
        indices = torch.sort(indices, dim=1).values
        return indices

    def forward(self, x: torch.Tensor) -> dict:
        # x: [B, C, T]
        assert x.dim() == 3, f"Expected [B, C, T], got {x.shape}"

        content = self.content_proj(x).transpose(1, 2).contiguous()          # [B, T, D]
        saliency_logits = self.saliency_head(x).squeeze(1)                    # [B, T]
        saliency_prob = torch.sigmoid(saliency_logits)                        # [B, T]

        phase = torch.sigmoid(self.phase_head(x).squeeze(1))                  # [B, T], [0,1]
        role_logits = self.role_head(x).transpose(1, 2).contiguous()          # [B, T, R]

        indices = self._select_event_indices(saliency_logits)

        sparse_content = _batched_gather_txd(content, indices)                # [B, K, D]
        sparse_saliency_logits = _batched_gather_txd(
            saliency_logits.unsqueeze(-1), indices
        ).squeeze(-1)                                                         # [B, K]
        sparse_saliency_prob = torch.sigmoid(sparse_saliency_logits)          # [B, K]
        sparse_phase = _batched_gather_txd(
            phase.unsqueeze(-1), indices
        ).squeeze(-1)                                                         # [B, K]
        sparse_role_logits = _batched_gather_txd(role_logits, indices)        # [B, K, R]
        sparse_role_prob = torch.softmax(sparse_role_logits, dim=-1)          # [B, K, R]

        phase_feat = build_phase_encoding(sparse_phase, self.phase_dim)       # [B, K, P]
        phase_emb = self.phase_proj(phase_feat)                               # [B, K, D]
        role_emb = sparse_role_prob @ self.role_embedding.weight              # [B, K, D]

        sparse_event_embeds = self.event_proj(
            torch.cat([sparse_content, phase_emb, role_emb], dim=-1)
        )                                                                     # [B, K, D]

        # use saliency as a soft gate for event strength
        sparse_event_embeds = sparse_event_embeds * sparse_saliency_prob.unsqueeze(-1)

        return {
            "content": content,
            "saliency_logits": saliency_logits,
            "saliency_prob": saliency_prob,
            "phase": phase,
            "role_logits": role_logits,
            "indices": indices,
            "sparse_content": sparse_content,
            "sparse_saliency_logits": sparse_saliency_logits,
            "sparse_saliency_prob": sparse_saliency_prob,
            "sparse_phase": sparse_phase,
            "sparse_role_logits": sparse_role_logits,
            "sparse_role_prob": sparse_role_prob,
            "sparse_event_embeds": sparse_event_embeds,
            "num_sparse_events": torch.tensor(indices.shape[1], device=indices.device),
        }