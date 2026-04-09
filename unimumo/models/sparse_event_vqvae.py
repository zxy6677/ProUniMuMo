import os.path
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import typing as tp

from einops import rearrange
from omegaconf import OmegaConf

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

        self.music_encoder = self.instantiate_music_encoder(**music_config)

        self.motion_encoder = Encoder(**motion_config)
        self.motion_decoder = Decoder(**motion_config)

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

        self.motion_fuse = nn.Sequential(
            nn.Conv1d(latent_dim * 2, latent_dim * 2, 1),
            nn.ELU(),
            nn.Conv1d(latent_dim * 2, latent_dim, 3, 1, 1),
            nn.ELU(),
            nn.Conv1d(latent_dim, latent_dim, 1),
        )

        self.loss = instantiate_from_config(loss_config)

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

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

    def instantiate_music_encoder(
        self,
        vqvae_ckpt: str,
        vqvae_config: tp.Optional[tp.Any] = None,
    ) -> SEANetEncoder:
        if os.path.exists(vqvae_ckpt):
            pkg = torch.load(vqvae_ckpt, map_location="cpu")
            cfg = OmegaConf.create(pkg["xp.cfg"])
            model = get_compression_model(cfg)
            model.load_state_dict(pkg["best_state"])
        else:
            assert vqvae_config is not None
            model = get_compression_model(vqvae_config)

        encoder = model.encoder
        for p in encoder.parameters():
            p.requires_grad = False
        return encoder

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
    ) -> torch.Tensor:
        fused_motion = self.motion_fuse(
            torch.cat([motion_private, motion_shared_dense], dim=1)
        )                                                          # [B, D, T]
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
        )

        commitment_loss = 0.5 * (
            shared_private["music_q"]["penalty"] +
            shared_private["motion_q"]["penalty"]
        )

        return {
            "motion_recon": motion_recon,
            "commitment_loss": commitment_loss,

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
        loss, log_dict = self.loss(batch[self.motion_key], model_output, split="train")

        self.log("aeloss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: tp.Dict[str, torch.Tensor], batch_idx: int):
        model_output = self(batch)
        loss, log_dict = self.loss(batch[self.motion_key], model_output, split="val")

        self.log("val/rec_loss", log_dict["val/rec_loss"], prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

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