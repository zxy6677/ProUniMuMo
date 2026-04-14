import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class SinkhornOTLoss(nn.Module):
    def __init__(
        self,
        epsilon: float = 0.1,
        max_iter: int = 30,
        lambda_idx: float = 0.0,
        lambda_phase: float = 0.0,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.lambda_idx = lambda_idx
        self.lambda_phase = lambda_phase

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_idx: torch.Tensor = None,
        y_idx: torch.Tensor = None,
        x_phase: torch.Tensor = None,
        y_phase: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        x: [B, Kx, D]
        y: [B, Ky, D]
        """
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)

        cost = 1.0 - torch.bmm(x, y.transpose(1, 2))

        if x_idx is not None and y_idx is not None and self.lambda_idx > 0:
            idx_cost = torch.abs(
                x_idx.float().unsqueeze(2) - y_idx.float().unsqueeze(1)
            )
            idx_cost = idx_cost / idx_cost.max().clamp_min(1.0)
            cost = cost + self.lambda_idx * idx_cost

        if x_phase is not None and y_phase is not None and self.lambda_phase > 0:
            if x_phase.dim() == 3 and y_phase.dim() == 3:
                # phase 是向量 / embedding: [B, K, D_phase]
                x_phase = F.normalize(x_phase, dim=-1)
                y_phase = F.normalize(y_phase, dim=-1)
                phase_cost = 1.0 - torch.bmm(x_phase, y_phase.transpose(1, 2))
            elif x_phase.dim() == 2 and y_phase.dim() == 2:
                # phase 是标量 / id: [B, K]
                phase_cost = torch.abs(
                    x_phase.float().unsqueeze(2) - y_phase.float().unsqueeze(1)
                )
                phase_cost = phase_cost / phase_cost.max().clamp_min(1.0)
            else:
                raise ValueError(
                    f"Unexpected phase shapes: x_phase={x_phase.shape}, y_phase={y_phase.shape}"
                )

            cost = cost + self.lambda_phase * phase_cost
        
        
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
        meta_weight: float = 0.02,
        sinkhorn_epsilon: float = 0.1,
        sinkhorn_iter: int = 30,
        lambda_idx: float = 0.0,
        lambda_phase: float = 0.0,
    ):
        super().__init__()
        self.meta_weight = meta_weight
        self.meta_loss = nn.MSELoss()
        self.rec_weight = rec_weight
        self.commitment_weight = commitment_weight
        self.ot_weight = ot_weight
        self.inv_weight = inv_weight

        self.recon_loss = nn.MSELoss()
        self.inv_loss = nn.MSELoss()
        self.ot = SinkhornOTLoss(
            epsilon=sinkhorn_epsilon,
            max_iter=sinkhorn_iter,
            lambda_idx=lambda_idx,
            lambda_phase=lambda_phase,
        )

    def forward(self, motion_gt: torch.Tensor, model_output: dict, split: str = "train"):
        motion_recon = model_output["motion_recon"]
        commitment_loss = model_output["commitment_loss"]

        # 在 shared sparse quantized event space 里做 OT
        music_shared = model_output["music_quantized_sparse_events"]
        motion_shared = model_output["motion_quantized_sparse_events"]

        rec_loss = self.recon_loss(motion_recon.contiguous(), motion_gt.contiguous())
        ot_loss = self.ot(
            music_shared,
            motion_shared,
            x_idx=model_output["music_event_indices"],
            y_idx=model_output["motion_event_indices"],
            x_phase=model_output["music_phase"],
            y_phase=model_output["motion_phase"],
        )

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

        if (
            "music_meta_pred" in model_output and
            "motion_meta_pred" in model_output and
            "music_meta_target" in model_output and
            "motion_meta_target" in model_output
        ):
            music_meta_loss = self.meta_loss(
                model_output["music_meta_pred"],
                model_output["music_meta_target"],
            )
            motion_meta_loss = self.meta_loss(
                model_output["motion_meta_pred"],
                model_output["motion_meta_target"],
            )
            meta_loss = 0.5 * (music_meta_loss + motion_meta_loss)
        else:
            music_meta_loss = torch.zeros_like(rec_loss)
            motion_meta_loss = torch.zeros_like(rec_loss)
            meta_loss = torch.zeros_like(rec_loss)

        total = (
            self.rec_weight * rec_loss +
            self.commitment_weight * commitment_loss +
            self.ot_weight * ot_loss +
            self.inv_weight * inv_loss +
            self.meta_weight * meta_loss
        )

        log = {
            f"{split}/total_loss": total.detach().mean(),
            f"{split}/rec_loss": rec_loss.detach().mean(),
            f"{split}/commitment_loss": commitment_loss.detach().mean(),
            f"{split}/ot_loss": ot_loss.detach().mean(),
            f"{split}/inv_loss": inv_loss.detach().mean(),
            f"{split}/meta_loss": meta_loss.detach().mean(),
            f"{split}/music_meta_loss": music_meta_loss.detach().mean(),
            f"{split}/motion_meta_loss": motion_meta_loss.detach().mean(),
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
