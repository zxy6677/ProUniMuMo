import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class SinkhornOTLoss(nn.Module):
    def __init__(self, epsilon: float = 0.1, max_iter: int = 30):
        super().__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x: [B, Kx, D]
        y: [B, Ky, D]
        """
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)

        cost = 1.0 - torch.bmm(x, y.transpose(1, 2))  # [B, Kx, Ky]
        b, kx, ky = cost.shape

        log_a = torch.full((b, kx), -math.log(kx), device=cost.device, dtype=cost.dtype)
        log_b = torch.full((b, ky), -math.log(ky), device=cost.device, dtype=cost.dtype)

        u = torch.zeros_like(log_a)
        v = torch.zeros_like(log_b)
        K = -cost / self.epsilon

        for _ in range(self.max_iter):
            u = log_a - torch.logsumexp(K + v.unsqueeze(1), dim=2)
            v = log_b - torch.logsumexp(K + u.unsqueeze(2), dim=1)

        pi = torch.exp(K + u.unsqueeze(2) + v.unsqueeze(1))
        return (pi * cost).sum(dim=(1, 2)).mean()


class SparseEventLoss(nn.Module):
    def __init__(
        self,
        rec_weight: float = 1.0,
        commitment_weight: float = 0.02,
        ot_weight: float = 0.1,
        inv_weight: float = 0.0,
        sinkhorn_epsilon: float = 0.1,
        sinkhorn_iter: int = 30,
    ):
        super().__init__()
        self.rec_weight = rec_weight
        self.commitment_weight = commitment_weight
        self.ot_weight = ot_weight
        self.inv_weight = inv_weight

        self.recon_loss = nn.MSELoss()
        self.inv_loss = nn.MSELoss()
        self.ot = SinkhornOTLoss(
            epsilon=sinkhorn_epsilon,
            max_iter=sinkhorn_iter,
        )

    def forward(self, motion_gt: torch.Tensor, model_output: dict, split: str = "train"):
        motion_recon = model_output["motion_recon"]
        commitment_loss = model_output["commitment_loss"]

        # 在 shared sparse quantized event space 里做 OT
        music_shared = model_output["music_quantized_sparse_events"]
        motion_shared = model_output["motion_quantized_sparse_events"]

        rec_loss = self.recon_loss(motion_recon.contiguous(), motion_gt.contiguous())
        ot_loss = self.ot(music_shared, motion_shared)

        if (
            "music_quantized_sparse_events_aug" in model_output and
            "motion_quantized_sparse_events_aug" in model_output and
            model_output["music_quantized_sparse_events_aug"] is not None and
            model_output["motion_quantized_sparse_events_aug"] is not None
        ):
            inv_loss = 0.5 * (
                self.inv_loss(
                    model_output["music_quantized_sparse_events"],
                    model_output["music_quantized_sparse_events_aug"]
                ) +
                self.inv_loss(
                    model_output["motion_quantized_sparse_events"],
                    model_output["motion_quantized_sparse_events_aug"]
                )
            )
        else:
            inv_loss = torch.zeros_like(rec_loss)

        total = (
            self.rec_weight * rec_loss +
            self.commitment_weight * commitment_loss +
            self.ot_weight * ot_loss +
            self.inv_weight * inv_loss
        )

        log = {
            f"{split}/total_loss": total.detach().mean(),
            f"{split}/rec_loss": rec_loss.detach().mean(),
            f"{split}/commitment_loss": commitment_loss.detach().mean(),
            f"{split}/ot_loss": ot_loss.detach().mean(),
            f"{split}/inv_loss": inv_loss.detach().mean(),
        }
        return total, log



class MotionVqVaeLoss(nn.Module):
    def __init__(self, commitment_loss_weight: float = 1.0, motion_weight: float = 1.0):
        super().__init__()
        self.commitment_loss_weight = commitment_loss_weight
        self.motion_weight = motion_weight
        self.recon_loss = nn.MSELoss()

    def forward(self, motion_gt: torch.Tensor, motion_recon: torch.Tensor, commitment_loss: torch.Tensor, split: str = "train"):
        motion_rec_loss = self.recon_loss(motion_recon.contiguous(), motion_gt.contiguous())

        loss = self.motion_weight * motion_rec_loss + self.commitment_loss_weight * commitment_loss
        rec_loss = self.motion_weight * motion_rec_loss

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
               "{}/commitment_loss".format(split): commitment_loss.detach().mean(),
               "{}/rec_loss".format(split): rec_loss.detach().mean(),
               }
        return loss, log
