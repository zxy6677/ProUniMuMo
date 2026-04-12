import os.path
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import typing as tp
import torch.nn.functional as F

from einops import rearrange
from omegaconf import OmegaConf, DictConfig

from unimumo.util import instantiate_from_config
from unimumo.audio.audiocraft_.models.builders import get_compression_model
from unimumo.audio.audiocraft_.modules.seanet import SEANetEncoder
from unimumo.motion.motion_process import recover_from_ric
from unimumo.modules.motion_vqvae_module import Encoder, Decoder
from unimumo.modules.event_head import EventHead, scatter_sparse_tokens


class SparseSharedEventVQVAE(pl.LightningModule):
    def __init__(
        self,
        music_config: dict,
        motion_config: dict,
        shared_event_config: dict,
        shared_quantizer_config: dict,
        loss_config: dict,
        ot_warmup_start_step: int = 0,
        ot_warmup_steps: int = 0,
        ot_warmup_init_weight: float = 0.0,
        ot_warmup_target_weight: tp.Optional[float] = None,
        ckpt_path: tp.Optional[str] = None,
        ignore_keys: tp.Optional[tp.List[str]] = None,
        music_key: str = "waveform",
        motion_key: str = "motion",
        monitor: tp.Optional[str] = None,
    ):
        super().__init__()
        self.motion_key = motion_key
        self.music_key = music_key
        self.quantize_fps = shared_quantizer_config.get("params", {}).get("default_frame_rate", 50)
        self.shared_bins = shared_quantizer_config.get("params", {}).get("bins", 128)
        self.private_mask_ratio = shared_event_config.get("private_mask_ratio", 0.3)
        self.shared_recon_weight = shared_event_config.get("shared_recon_weight", 0.1)
        self.ot_warmup_start_step = ot_warmup_start_step
        self.ot_warmup_steps = ot_warmup_steps
        self.ot_warmup_init_weight = ot_warmup_init_weight
        self.ot_warmup_target_weight = (
            ot_warmup_target_weight
            if ot_warmup_target_weight is not None
            else loss_config.get("params", {}).get("ot_weight", 0.0)
        )
                # allow optional pretrained motion init
        motion_config = dict(motion_config)
        motion_vqvae_ckpt = motion_config.pop("vqvae_ckpt", None)

        self.music_encoder = self.instantiate_music_encoder(**music_config)

        self.motion_encoder = Encoder(**motion_config)
        self.motion_decoder = Decoder(**motion_config)

        self.music_shared_alpha = nn.Parameter(torch.tensor(-1.5))

        if motion_vqvae_ckpt is not None:
            self.init_motion_backbone_from_ckpt(motion_vqvae_ckpt)

        latent_dim = motion_config["output_dim"]
        music_latent_dim = 128

        self.music_private_proj = nn.Sequential(
            nn.Conv1d(music_latent_dim, latent_dim, 1),
            nn.ELU(),
            nn.Conv1d(latent_dim, latent_dim, 1),
        )

        self.motion_private_proj = nn.Sequential(
            nn.Conv1d(latent_dim, latent_dim, 1),
            nn.ELU(),
            nn.Conv1d(latent_dim, latent_dim, 1),
        )

        self.music_event_head = EventHead(
            input_dim=music_latent_dim,
            event_dim=latent_dim,
            hidden_dim=shared_event_config.get("hidden_dim", 256),
            phase_dim=shared_event_config.get("phase_dim", 32),
            role_num=shared_event_config.get("role_num", 4),
            topk=shared_event_config.get("topk", 8),
        )

        self.motion_event_head = EventHead(
            input_dim=latent_dim,
            event_dim=latent_dim,
            hidden_dim=shared_event_config.get("hidden_dim", 256),
            phase_dim=shared_event_config.get("phase_dim", 32),
            role_num=shared_event_config.get("role_num", 4),
            topk=shared_event_config.get("topk", 8),
        )

        # NEW: dedicated shared sparse-event codebook
        self.shared_event_quantizer = instantiate_from_config(shared_quantizer_config)

        # scatter sparse events -> dense timeline, then smooth / spread locally
        self.motion_shared_refiner = nn.Sequential(
            nn.Conv1d(latent_dim, latent_dim, 3, 1, 1),
            nn.ELU(),
            nn.Conv1d(latent_dim, latent_dim, 3, 1, 1),
        )

        self.music_shared_refiner = nn.Sequential(
            nn.Conv1d(latent_dim, latent_dim, 3, 1, 1),
            nn.ELU(),
            nn.Conv1d(latent_dim, latent_dim, 3, 1, 1),
        )

        self.motion_shared_proj = nn.Conv1d(latent_dim, latent_dim, 1)
        self.music_shared_proj = nn.Conv1d(latent_dim, latent_dim, 1)

        self.motion_gate_motion = nn.Sequential(
            nn.Conv1d(latent_dim * 3, latent_dim, 1),
            nn.ELU(),
            nn.Conv1d(latent_dim, latent_dim, 1),
        )

        self.motion_gate_music = nn.Sequential(
            nn.Conv1d(latent_dim * 3, latent_dim, 1),
            nn.ELU(),
            nn.Conv1d(latent_dim, latent_dim, 1),
        )

        self.loss = instantiate_from_config(loss_config)

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)


    def _compute_code_stats(self, codes: torch.Tensor) -> dict:
        """
        codes: [B, n_q, K] or similar integer tensor
        """
        flat = codes.detach().reshape(-1).long()
        flat = flat[(flat >= 0) & (flat < self.shared_bins)]

        if flat.numel() == 0:
            zero = torch.tensor(0.0, device=self.device)
            return {
                "usage": zero,
                "perplexity": zero,
                "perplexity_norm": zero,
                "dead_codes": torch.tensor(float(self.shared_bins), device=self.device),
            }

        hist = torch.bincount(flat, minlength=self.shared_bins).float()
        probs = hist / hist.sum().clamp_min(1.0)

        used = (hist > 0).float()
        usage = used.mean()

        nz = probs > 0
        entropy = -(probs[nz] * torch.log(probs[nz] + 1e-12)).sum()
        perplexity = torch.exp(entropy)
        perplexity_norm = perplexity / float(self.shared_bins)

        dead_codes = (hist == 0).float().sum()

        return {
            "usage": usage,
            "perplexity": perplexity,
            "perplexity_norm": perplexity_norm,
            "dead_codes": dead_codes,
        }

    def _get_current_ot_weight(self) -> float:
        step = int(self.global_step)

        if step < self.ot_warmup_start_step:
            return float(self.ot_warmup_init_weight)

        if self.ot_warmup_steps <= 0:
            return float(self.ot_warmup_target_weight)

        progress = (step - self.ot_warmup_start_step) / float(self.ot_warmup_steps)
        progress = max(0.0, min(1.0, progress))

        return float(
            self.ot_warmup_init_weight
            + progress * (self.ot_warmup_target_weight - self.ot_warmup_init_weight)
        )

    def _set_current_ot_weight(self):
        current_ot_weight = self._get_current_ot_weight()

        if hasattr(self.loss, "ot_weight"):
            self.loss.ot_weight = current_ot_weight
        else:
            raise AttributeError(
                "SparseEventLoss does not expose `ot_weight`, cannot apply OT warmup."
            )

        return current_ot_weight

    def init_from_ckpt(self, path: str, ignore_keys: tp.Optional[tp.List[str]] = None):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        if ignore_keys is not None:
            keys = list(sd.keys())
            for k in keys:
                for ik in ignore_keys:
                    if k.startswith(ik):
                        print(f"Deleting key {k} from state_dict.")
                        del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def _safe_torch_load(self, path: str):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def instantiate_music_encoder(
        self,
        vqvae_ckpt: str,
        vqvae_config: tp.Optional[tp.Any] = None,
    ) -> SEANetEncoder:
        if os.path.exists(vqvae_ckpt):
            pkg = self._safe_torch_load(vqvae_ckpt)

            if not isinstance(pkg, dict):
                raise TypeError(f"Unexpected checkpoint type: {type(pkg)}")

            # Case 1: old audiocraft-style checkpoint
            if "xp.cfg" in pkg and "best_state" in pkg:
                cfg = OmegaConf.create(pkg["xp.cfg"])
                state_dict = pkg["best_state"]

            # Case 2: UniMuMo packaged full.ckpt
            elif "music_vqvae_config" in pkg and "music_vqvae_weight" in pkg:
                cfg = pkg["music_vqvae_config"]
                if isinstance(cfg, dict):
                    cfg = OmegaConf.create(cfg)
                elif not isinstance(cfg, DictConfig):
                    cfg = OmegaConf.create(cfg)
                state_dict = pkg["music_vqvae_weight"]

            else:
                raise KeyError(
                    "Checkpoint format mismatch. "
                    f"Expected keys ['xp.cfg', 'best_state'] or "
                    f"['music_vqvae_config', 'music_vqvae_weight'], "
                    f"got keys: {list(pkg.keys())[:20]}"
                )

            model = get_compression_model(cfg)
            model.load_state_dict(state_dict, strict=True)
        else:
            assert vqvae_config is not None
            model = get_compression_model(vqvae_config)

        encoder = model.encoder
        for p in encoder.parameters():
            p.requires_grad = False
        return encoder

    def init_motion_backbone_from_ckpt(self, ckpt_path: str):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"motion vqvae ckpt not found: {ckpt_path}")

        pkg = self._safe_torch_load(ckpt_path)

        if not isinstance(pkg, dict):
            raise TypeError(f"Unexpected checkpoint type: {type(pkg)}")

        if "motion_vqvae_weight" not in pkg:
            raise KeyError(
                f"Expected 'motion_vqvae_weight' in checkpoint, got keys: {list(pkg.keys())[:20]}"
            )

        state_dict = pkg["motion_vqvae_weight"]

        encoder_sd = {}
        decoder_sd = {}

        # common case: saved keys prefixed with encoder. / decoder.
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                encoder_sd[k[len("encoder."):]] = v
            elif k.startswith("decoder."):
                decoder_sd[k[len("decoder."):]] = v

        # fallback: sparse-event-style names
        if len(encoder_sd) == 0 and len(decoder_sd) == 0:
            for k, v in state_dict.items():
                if k.startswith("motion_encoder."):
                    encoder_sd[k[len("motion_encoder."):]] = v
                elif k.startswith("motion_decoder."):
                    decoder_sd[k[len("motion_decoder."):]] = v

        if len(encoder_sd) == 0:
            raise KeyError("Could not find encoder weights in motion_vqvae_weight")
        if len(decoder_sd) == 0:
            raise KeyError("Could not find decoder weights in motion_vqvae_weight")

        missing_e, unexpected_e = self.motion_encoder.load_state_dict(encoder_sd, strict=False)
        missing_d, unexpected_d = self.motion_decoder.load_state_dict(decoder_sd, strict=False)

        print("[motion init] encoder missing:", missing_e)
        print("[motion init] encoder unexpected:", unexpected_e)
        print("[motion init] decoder missing:", missing_d)
        print("[motion init] decoder unexpected:", unexpected_d)
        print(f"[motion init] restored motion encoder/decoder from {ckpt_path}")


    def encode_backbone(
        self,
        x_music: torch.Tensor,
        x_motion: torch.Tensor,
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        # x_music: [B, 1, 32000*T]
        # x_motion: [B, 60*T, 263]
        with torch.no_grad():
            music_emb = self.music_encoder(x_music)                # [B, 128, 50*T]

        x_motion = rearrange(x_motion, "b t d -> b d t")
        motion_emb = self.motion_encoder(x_motion)                 # [B, D, 50*T]
        return music_emb, motion_emb

    def build_shared_private(
        self,
        music_emb: torch.Tensor,
        motion_emb: torch.Tensor,
    ) -> dict:
        seq_len = motion_emb.shape[-1]

        music_private = self.music_private_proj(music_emb)         # [B, D, T]
        motion_private = self.motion_private_proj(motion_emb)      # [B, D, T]

        music_event = self.music_event_head(music_emb)
        motion_event = self.motion_event_head(motion_emb)

        music_q = self.shared_event_quantizer(
            music_event["sparse_event_embeds"],
            frame_rate=self.quantize_fps,
        )
        motion_q = self.shared_event_quantizer(
            motion_event["sparse_event_embeds"],
            frame_rate=self.quantize_fps,
        )

        music_shared_dense = scatter_sparse_tokens(
            music_q["x_q"], music_event["indices"], seq_len
        )                                                          # [B, D, T]

        motion_shared_dense = scatter_sparse_tokens(
            motion_q["x_q"], motion_event["indices"], seq_len
        )                                                          # [B, D, T]

        music_shared_dense = self.music_shared_refiner(music_shared_dense)
        motion_shared_dense = self.motion_shared_refiner(motion_shared_dense)

        return {
            "music_private": music_private,
            "motion_private": motion_private,
            "music_event": music_event,
            "motion_event": motion_event,
            "music_q": music_q,
            "motion_q": motion_q,
            "music_shared_dense": music_shared_dense,
            "motion_shared_dense": motion_shared_dense,
        }

    def decode_motion(
        self,
        motion_private: torch.Tensor,
        motion_shared_dense: torch.Tensor,
        music_shared_dense: torch.Tensor,
        force_shared_only: bool = False,
    ) -> torch.Tensor:
        if force_shared_only:
            motion_private = torch.zeros_like(motion_private)
        elif self.training and self.private_mask_ratio > 0:
            keep = (
                torch.rand(motion_private.shape[0], 1, 1, device=motion_private.device)
                > self.private_mask_ratio
            ).float()
            motion_private = motion_private * keep

        gate_in = torch.cat(
            [motion_private, motion_shared_dense, music_shared_dense],
            dim=1,
        )

        gate_motion = torch.sigmoid(self.motion_gate_motion(gate_in))
        gate_music = torch.sigmoid(self.motion_gate_music(gate_in))

        alpha = torch.sigmoid(self.music_shared_alpha)

        motion_shared = self.motion_shared_proj(motion_shared_dense)
        music_shared = self.music_shared_proj(music_shared_dense)

        fused_motion = (
            motion_private
            + gate_motion * motion_shared
            + gate_music * (alpha * music_shared)
        )

        motion_recon = self.motion_decoder(fused_motion)
        motion_recon = rearrange(motion_recon, "b d t -> b t d")
        return motion_recon

    def forward(self, batch: tp.Dict[str, torch.Tensor]) -> dict:
        music_emb, motion_emb = self.encode_backbone(
            batch[self.music_key],
            batch[self.motion_key],
        )

        shared_private = self.build_shared_private(music_emb, motion_emb)

        motion_recon = self.decode_motion(
            shared_private["motion_private"],
            shared_private["motion_shared_dense"],
            shared_private["music_shared_dense"],
        )

        shared_only_recon = self.decode_motion(
            shared_private["motion_private"],
            shared_private["motion_shared_dense"],
            shared_private["music_shared_dense"],
            force_shared_only=True,
        )

        commitment_loss = 0.5 * (
            shared_private["music_q"]["penalty"] +
            shared_private["motion_q"]["penalty"]
        )

        return {
            "motion_recon": motion_recon,
            "commitment_loss": commitment_loss,

            "shared_only_recon": shared_only_recon,

            # continuous sparse events before RVQ
            "music_sparse_event_embeds": shared_private["music_event"]["sparse_event_embeds"],
            "motion_sparse_event_embeds": shared_private["motion_event"]["sparse_event_embeds"],

            # quantized sparse shared events after RVQ
            "music_quantized_sparse_events": shared_private["music_q"]["x_q"],
            "motion_quantized_sparse_events": shared_private["motion_q"]["x_q"],

            # discrete codes from the shared event codebook
            "music_shared_codes": shared_private["music_q"]["codes"],      # [B, n_q, K_event]
            "motion_shared_codes": shared_private["motion_q"]["codes"],    # [B, n_q, K_event]

            # event metadata
            "music_event_indices": shared_private["music_event"]["indices"],
            "motion_event_indices": shared_private["motion_event"]["indices"],
            "music_saliency": shared_private["music_event"]["sparse_saliency_prob"],
            "motion_saliency": shared_private["motion_event"]["sparse_saliency_prob"],
            "music_phase": shared_private["music_event"]["sparse_phase"],
            "motion_phase": shared_private["motion_event"]["sparse_phase"],
            "music_role_prob": shared_private["music_event"]["sparse_role_prob"],
            "motion_role_prob": shared_private["motion_event"]["sparse_role_prob"],
        }

    @torch.no_grad()
    def encode_shared_codes(self, batch: tp.Dict[str, torch.Tensor]) -> dict:
        music_emb, motion_emb = self.encode_backbone(
            batch[self.music_key],
            batch[self.motion_key],
        )
        shared_private = self.build_shared_private(music_emb, motion_emb)
        return {
            "music_shared_codes": shared_private["music_q"]["codes"],
            "motion_shared_codes": shared_private["motion_q"]["codes"],
            "music_event_indices": shared_private["music_event"]["indices"],
            "motion_event_indices": shared_private["motion_event"]["indices"],
        }

    def motion_vec_to_joint(self, vec: torch.Tensor, motion_mean: np.ndarray, motion_std: np.ndarray) -> np.ndarray:
        mean = torch.tensor(motion_mean).to(vec)
        std = torch.tensor(motion_std).to(vec)
        vec = vec * std + mean
        joint = recover_from_ric(vec, joints_num=22)
        joint = joint.cpu().detach().numpy()
        return joint

    def training_step(self, batch: tp.Dict[str, torch.Tensor], batch_idx: int):
        model_output = self(batch)
        current_ot_weight = self._set_current_ot_weight()
        loss, log_dict = self.loss(batch[self.motion_key], model_output, split="train")
        music_stats = self._compute_code_stats(model_output["music_shared_codes"])
        motion_stats = self._compute_code_stats(model_output["motion_shared_codes"])
        shared_recon_loss = F.mse_loss(model_output["shared_only_recon"], batch[self.motion_key])
        loss = loss + self.shared_recon_weight * shared_recon_loss
        log_dict["train/shared_recon_loss"] = shared_recon_loss.detach()
        log_dict["train/total_loss_with_shared"] = loss.detach()
        log_dict["train/ot_weight"] = torch.tensor(
        current_ot_weight, device=self.device, dtype=torch.float32
    )
        log_dict["train/music_shared_alpha"] = torch.sigmoid(self.music_shared_alpha).detach()


        self.log("train/music_code_usage", music_stats["usage"], prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/music_code_perplexity", music_stats["perplexity"], prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/music_code_perplexity_norm", music_stats["perplexity_norm"], prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/music_dead_codes", music_stats["dead_codes"], prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        self.log("train/motion_code_usage", motion_stats["usage"], prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/motion_code_perplexity", motion_stats["perplexity"], prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/motion_code_perplexity_norm", motion_stats["perplexity_norm"], prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/motion_dead_codes", motion_stats["dead_codes"], prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        self.log("aeloss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def validation_step(self, batch: tp.Dict[str, torch.Tensor], batch_idx: int):
        model_output = self(batch)
        current_ot_weight = self._set_current_ot_weight()
        loss, log_dict = self.loss(batch[self.motion_key], model_output, split="val")
        music_stats = self._compute_code_stats(model_output["music_shared_codes"])
        motion_stats = self._compute_code_stats(model_output["motion_shared_codes"])
        shared_recon_loss = F.mse_loss(model_output["shared_only_recon"], batch[self.motion_key])
        val_total_loss_with_shared = loss + self.shared_recon_weight * shared_recon_loss
        self.log(
            "val/total_loss_with_shared",
            val_total_loss_with_shared,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log("val/shared_recon_loss", shared_recon_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)

        self.log(
            "val/ot_weight",
            torch.tensor(current_ot_weight, device=self.device, dtype=torch.float32),
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        self.log(
            "val/music_shared_alpha",
            torch.sigmoid(self.music_shared_alpha).detach(),
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        self.log("val/music_code_usage", music_stats["usage"], prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/music_code_perplexity", music_stats["perplexity"], prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/music_code_perplexity_norm", music_stats["perplexity_norm"], prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/music_dead_codes", music_stats["dead_codes"], prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)

        self.log("val/motion_code_usage", motion_stats["usage"], prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/motion_code_perplexity", motion_stats["perplexity"], prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/motion_code_perplexity_norm", motion_stats["perplexity_norm"], prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/motion_dead_codes", motion_stats["dead_codes"], prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)

        self.log("val/rec_loss", log_dict["val/rec_loss"], prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        return val_total_loss_with_shared

    def configure_optimizers(self):
        lr = self.learning_rate
        opt = torch.optim.AdamW(self.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0)
        return [opt], []

    @torch.no_grad()
    def log_videos(
        self,
        batch: tp.Dict[str, torch.Tensor],
        motion_mean: np.ndarray,
        motion_std: np.ndarray,
    ):
        model_output = self(batch)
        motion_recon = model_output["motion_recon"]
        waveform = batch[self.music_key].detach().cpu().numpy()

        joint = self.motion_vec_to_joint(motion_recon, motion_mean, motion_std)
        gt_joint = self.motion_vec_to_joint(batch[self.motion_key], motion_mean, motion_std)
        return waveform, joint, gt_joint

        