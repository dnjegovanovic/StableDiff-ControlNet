from __future__ import annotations

from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from ControlNet.models.LinearNoiseScheduler import LinearNoiseScheduler
from ControlNet.modules.UNet import UNet
from ControlNet.modules.VQVAE import VectorQuantizedVAE
from ControlNet.utils.config import (
    get_config_value,
    validate_class_config,
    validate_image_config,
    validate_text_config,
)
from ControlNet.utils.diff_utils import (
    drop_class_condition,
    drop_image_condition,
    drop_text_condition,
)


class DDPMLightning(pl.LightningModule):
    def __init__(
        self,
        unet_params: Dict[str, Any],
        vqvae_params: Dict[str, Any],
        ddpm_params: Optional[Dict[str, Any]] = None,
        input_channels: Optional[int] = None,
        vqvae_ckpt_path: Optional[str] = None,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        condition_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        if input_channels is None:
            input_channels = vqvae_params.get("im_channels")
        if input_channels is None:
            raise ValueError("input_channels must be provided for VQ-VAE.")

        self._unet_params = self._build_unet_params(
            unet_params, vqvae_params, condition_config
        )
        self.model = UNet(UnetParams=self._unet_params)
        self.vqvae = VectorQuantizedVAE(input_channels, VQVAE=vqvae_params)
        self.vqvae.requires_grad_(False)

        self.scheduler: Optional[LinearNoiseScheduler] = None
        self.condition_config = self._get_condition_config(self._unet_params)
        self.condition_types = (
            [] if self.condition_config is None else self.condition_config["condition_types"]
        )

        self.use_class = "class" in self.condition_types
        self.use_image = "image" in self.condition_types
        self.use_text = "text" in self.condition_types

        self.num_classes = None
        self.class_drop_prob = 0.0
        self.image_drop_prob = 0.0
        self.text_drop_prob = 0.0
        self.text_embed_model = None
        self.text_tokenizer = None
        self.text_model = None
        self.empty_text_embed = None

        if self.use_class:
            validate_class_config(self.condition_config)
            class_cfg = self.condition_config["class_condition_config"]
            self.num_classes = int(class_cfg["num_classes"])
            self.class_drop_prob = float(get_config_value(class_cfg, "cond_drop_prob", 0.0))
        if self.use_image:
            validate_image_config(self.condition_config)
            image_cfg = self.condition_config["image_condition_config"]
            self.image_drop_prob = float(get_config_value(image_cfg, "cond_drop_prob", 0.0))
        if self.use_text:
            validate_text_config(self.condition_config)
            text_cfg = self.condition_config["text_condition_config"]
            self.text_drop_prob = float(get_config_value(text_cfg, "cond_drop_prob", 0.0))
            self.text_embed_model = text_cfg.get("text_embed_model")

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond_input=None):
        return self.model(x, t, cond_input=cond_input)

    def on_fit_start(self) -> None:
        self._load_vqvae_checkpoint()
        self._ensure_scheduler()
        self._freeze_vqvae()
        self._ensure_text_model()

    def on_train_epoch_start(self) -> None:
        self._freeze_vqvae()

    def training_step(self, batch, batch_idx):
        self._ensure_scheduler()
        images, raw_cond = self._unpack_batch(batch)
        images = images.to(self.device, non_blocking=True).float()

        with torch.no_grad():
            latents, _ = self.vqvae.encode(images)
        latents = latents.detach()

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.scheduler.num_timesteps,
            (latents.shape[0],),
            device=latents.device,
            dtype=torch.long,
        )
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        cond_input = self._build_cond_input(raw_cond, latents)
        noise_pred = self(noisy_latents, timesteps, cond_input=cond_input)
        loss = F.mse_loss(noise_pred, noise)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self._ensure_scheduler()
        images, raw_cond = self._unpack_batch(batch)
        images = images.to(self.device, non_blocking=True).float()

        with torch.no_grad():
            latents, _ = self.vqvae.encode(images)
        latents = latents.detach()

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.scheduler.num_timesteps,
            (latents.shape[0],),
            device=latents.device,
            dtype=torch.long,
        )
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        cond_input = self._build_cond_input(raw_cond, latents)
        noise_pred = self(noisy_latents, timesteps, cond_input=cond_input)
        loss = F.mse_loss(noise_pred, noise)

        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    @staticmethod
    def _build_unet_params(
        unet_params: Dict[str, Any],
        vqvae_params: Dict[str, Any],
        condition_config: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        params = dict(unet_params)
        latent_channels = vqvae_params.get("z_channels")
        if params.get("im_channels") is None:
            params["im_channels"] = latent_channels
        if latent_channels is not None and params["im_channels"] != latent_channels:
            raise ValueError("UNet im_channels must match VQ-VAE z_channels.")
        if condition_config is not None:
            model_config = dict(params.get("model_config", {}))
            model_config["condition_config"] = condition_config
            params["model_config"] = model_config
        return params

    @staticmethod
    def _get_condition_config(unet_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        model_config = get_config_value(unet_params, "model_config", unet_params)
        return get_config_value(model_config, "condition_config", None)

    def _ensure_scheduler(self) -> None:
        device = getattr(self, "device", torch.device("cpu"))
        if self.scheduler is None or getattr(self.scheduler, "device", None) != device:
            ddpm_params = self.hparams.ddpm_params or {}
            self.scheduler = LinearNoiseScheduler(device=device, DDPMParams=ddpm_params)

    def _freeze_vqvae(self) -> None:
        self.vqvae.eval()
        self.vqvae.requires_grad_(False)

    def _load_vqvae_checkpoint(self) -> None:
        ckpt_path = self.hparams.vqvae_ckpt_path
        if not ckpt_path:
            return
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        for prefix in ("module.", "model.", "vqvae."):
            state_dict = self._strip_prefix(state_dict, prefix)

        vqvae_state = self.vqvae.state_dict()
        filtered_state = {k: v for k, v in state_dict.items() if k in vqvae_state}
        if not filtered_state:
            raise RuntimeError(f"No VQ-VAE weights found in checkpoint: {ckpt_path}")
        missing_keys = set(vqvae_state) - set(filtered_state)
        if missing_keys:
            preview = ", ".join(sorted(missing_keys)[:10])
            raise RuntimeError(
                "VQ-VAE checkpoint is missing "
                f"{len(missing_keys)} keys. Example(s): {preview}"
            )
        self.vqvae.load_state_dict(filtered_state, strict=True)

    @staticmethod
    def _strip_prefix(state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        if not state_dict:
            return state_dict
        if not all(isinstance(key, str) for key in state_dict.keys()):
            return state_dict
        if not any(key.startswith(prefix) for key in state_dict):
            return state_dict
        stripped_state: Dict[str, Any] = {}
        for key, value in state_dict.items():
            if key.startswith(prefix):
                stripped_state[key[len(prefix) :]] = value
            else:
                stripped_state[key] = value
        return stripped_state

    def _ensure_text_model(self) -> None:
        if not self.use_text:
            return
        if self.text_model is not None and self.text_tokenizer is not None:
            return
        if not self.text_embed_model:
            raise ValueError("text_embed_model must be set for text conditioning.")
        from ControlNet.utils.text_model_utils import (
            get_text_representation,
            get_tokenizer_and_model,
        )

        self.text_tokenizer, self.text_model = get_tokenizer_and_model(
            self.text_embed_model, device=self.device
        )
        with torch.no_grad():
            self.empty_text_embed = get_text_representation(
                [""], self.text_tokenizer, self.text_model, self.device
            )

    def _unpack_batch(self, batch):
        if isinstance(batch, dict):
            return batch["x"], batch.get("cond_input", {})
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch[0], batch[1] or {}
        return batch, {}

    def _build_cond_input(
        self, raw_cond: Dict[str, Any], latents: torch.Tensor
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not self.condition_config:
            return None
        if not isinstance(raw_cond, dict):
            raw_cond = {}

        cond_input: Dict[str, torch.Tensor] = {}

        if self.use_class:
            class_indices = raw_cond.get("class")
            if class_indices is None:
                raise ValueError("Class conditioning enabled but batch has no 'class'.")
            class_indices = torch.as_tensor(class_indices, device=self.device).view(-1)
            class_one_hot = F.one_hot(
                class_indices.long(), num_classes=self.num_classes
            ).float()
            if self.class_drop_prob > 0.0:
                class_one_hot = drop_class_condition(
                    class_one_hot, self.class_drop_prob, latents
                )
            cond_input["class"] = class_one_hot

        if self.use_image:
            image_cond = raw_cond.get("image")
            if image_cond is None:
                raise ValueError("Image conditioning enabled but batch has no 'image'.")
            image_cond = torch.as_tensor(image_cond, device=self.device).float()
            if self.image_drop_prob > 0.0:
                image_cond = drop_image_condition(
                    image_cond, latents, self.image_drop_prob
                )
            cond_input["image"] = image_cond

        if self.use_text:
            if self.text_model is None or self.text_tokenizer is None:
                self._ensure_text_model()
            from ControlNet.utils.text_model_utils import get_text_representation

            text_cond = raw_cond.get("text")
            if text_cond is None:
                raise ValueError("Text conditioning enabled but batch has no 'text'.")
            with torch.no_grad():
                text_embed = get_text_representation(
                    text_cond, self.text_tokenizer, self.text_model, self.device
                )
                if self.text_drop_prob > 0.0:
                    text_embed = drop_text_condition(
                        text_embed, latents, self.empty_text_embed, self.text_drop_prob
                    )
            cond_input["text"] = text_embed

        return cond_input or None
