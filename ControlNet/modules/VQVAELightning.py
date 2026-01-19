import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from ControlNet.models.Discriminator import Discriminator
from ControlNet.models.LPIPS import LPIPS
from ControlNet.modules.VQVAE import VectorQuantizedVAE


class VQVAELightning(pl.LightningModule):
    def __init__(
        self,
        vqvae_params,
        input_channels=None,
        lr=1e-4,
        weight_decay=0.0,
        recon_loss="mse",
        codebook_weight=1.0,
        commitment_weight=1.0,
        grad_accum_steps=1,
        use_discriminator=False,
        disc_start=0,
        disc_weight=1.0,
        disc_lr=None,
        disc_betas=(0.5, 0.999),
        use_lpips=False,
        perceptual_weight=1.0,
        lpips_auto_download=True,
        lpips_weights_url=None,
        lpips_backbone_pretrained=True,
    ):
        super().__init__()
        lr = _as_float(lr, "lr")
        weight_decay = _as_float(weight_decay, "weight_decay")
        codebook_weight = _as_float(codebook_weight, "codebook_weight")
        commitment_weight = _as_float(commitment_weight, "commitment_weight")
        disc_weight = _as_float(disc_weight, "disc_weight")
        disc_lr = None if disc_lr is None else _as_float(disc_lr, "disc_lr")
        perceptual_weight = _as_float(perceptual_weight, "perceptual_weight")
        disc_betas = tuple(_as_float(beta, "disc_betas") for beta in disc_betas)
        grad_accum_steps = int(grad_accum_steps)
        disc_start = int(disc_start)
        self.save_hyperparameters()
        self.automatic_optimization = False
        self._train_step_count = 0
        # Resolve input channels from explicit arg or config for convenience.
        if input_channels is None:
            input_channels = vqvae_params.get("im_channels")
        assert (
            input_channels is not None
        ), "input_channels must be provided or present as vqvae_params['im_channels']."

        # Core VQ-VAE model used by the Lightning wrapper.
        self.model = VectorQuantizedVAE(input_channels, VQVAE=vqvae_params)

        # Pick reconstruction loss function for training/validation.
        if recon_loss not in ("mse", "l1"):
            raise ValueError("recon_loss must be 'mse' or 'l1'.")
        self._recon_loss_fn = F.mse_loss if recon_loss == "mse" else F.l1_loss
        self.grad_accum_steps = max(1, int(grad_accum_steps))

        # Optional discriminator (LSGAN) and LPIPS perceptual loss.
        self.use_discriminator = use_discriminator
        self.disc_start = int(disc_start)
        self.disc_weight = float(disc_weight)
        self.disc_lr = disc_lr
        self.disc_betas = disc_betas

        self.use_lpips = use_lpips
        self.perceptual_weight = float(perceptual_weight)

        if self.use_discriminator:
            self.discriminator = Discriminator(im_channels=input_channels)
            self.adv_loss_fn = F.mse_loss
        else:
            self.discriminator = None
            self.adv_loss_fn = None

        if self.use_lpips:
            self.lpips = LPIPS(
                auto_download=lpips_auto_download,
                weights_url=lpips_weights_url,
                backbone_pretrained=lpips_backbone_pretrained,
            )
            self.lpips.eval()
        else:
            self.lpips = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if self.use_discriminator:
            opt_g, opt_d = self.optimizers()
        else:
            opt_g = self.optimizers()
            opt_d = None

        if self._train_step_count % self.grad_accum_steps == 0:
            opt_g.zero_grad(set_to_none=True)
            if opt_d is not None:
                opt_d.zero_grad(set_to_none=True)

        x = batch["x"]
        recon, _, losses = self(x)

        # Combine reconstruction and quantization losses.
        recon_loss = self._recon_loss_fn(recon, x)
        codebook_loss = losses["codebook"]
        commitment_loss = losses["commitment"]
        loss = (
            recon_loss
            + self.hparams.codebook_weight * codebook_loss
            + self.hparams.commitment_weight * commitment_loss
        )

        if self.use_lpips:
            recon_lpips, x_lpips = self._lpips_inputs(recon, x)
            perceptual = self.lpips(recon_lpips, x_lpips).mean()
            loss = loss + self.perceptual_weight * perceptual
        else:
            perceptual = None

        g_adv = None
        disc_loss = None
        if self.use_discriminator and self._train_step_count >= self.disc_start:
            self._set_requires_grad(self.discriminator, False)
            fake_pred = self.discriminator(recon)
            adv_target = torch.ones_like(fake_pred, device=fake_pred.device)
            g_adv = self.adv_loss_fn(fake_pred, adv_target)
            loss = loss + self.disc_weight * g_adv

        loss_scaled = loss / self.grad_accum_steps
        self.manual_backward(loss_scaled)

        if self.use_discriminator and self._train_step_count >= self.disc_start:
            self._set_requires_grad(self.discriminator, True)
            real_pred = self.discriminator(x.detach())
            fake_pred = self.discriminator(recon.detach())
            real_labels = torch.ones_like(real_pred, device=real_pred.device)
            fake_labels = torch.zeros_like(fake_pred, device=fake_pred.device)
            real_loss = self.adv_loss_fn(real_pred, real_labels)
            fake_loss = self.adv_loss_fn(fake_pred, fake_labels)
            disc_loss = self.disc_weight * 0.5 * (real_loss + fake_loss)
            self.manual_backward(disc_loss / self.grad_accum_steps)

        is_accum_boundary = (
            (self._train_step_count + 1) % self.grad_accum_steps == 0
        )
        is_last_batch = (batch_idx + 1) == self.trainer.num_training_batches
        if is_accum_boundary or is_last_batch:
            opt_g.step()
            opt_g.zero_grad(set_to_none=True)
            if opt_d is not None and self._train_step_count >= self.disc_start:
                opt_d.step()
                opt_d.zero_grad(set_to_none=True)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_recon_loss", recon_loss, prog_bar=False)
        self.log("train_codebook_loss", codebook_loss, prog_bar=False)
        self.log("train_commitment_loss", commitment_loss, prog_bar=False)
        if perceptual is not None:
            self.log("train_perceptual_loss", perceptual, prog_bar=False)
        if g_adv is not None:
            self.log("train_g_adv_loss", g_adv, prog_bar=False)
        if disc_loss is not None:
            self.log("train_d_loss", disc_loss, prog_bar=False)

        self._train_step_count += 1
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["x"]
        recon, _, losses = self(x)

        # Mirror training loss composition for validation metrics.
        recon_loss = self._recon_loss_fn(recon, x)
        codebook_loss = losses["codebook"]
        commitment_loss = losses["commitment"]
        loss = (
            recon_loss
            + self.hparams.codebook_weight * codebook_loss
            + self.hparams.commitment_weight * commitment_loss
        )

        if self.use_lpips:
            recon_lpips, x_lpips = self._lpips_inputs(recon, x)
            perceptual = self.lpips(recon_lpips, x_lpips).mean()
            loss = loss + self.perceptual_weight * perceptual
        else:
            perceptual = None

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_recon_loss", recon_loss, prog_bar=False)
        self.log("val_codebook_loss", codebook_loss, prog_bar=False)
        self.log("val_commitment_loss", commitment_loss, prog_bar=False)
        if perceptual is not None:
            self.log("val_perceptual_loss", perceptual, prog_bar=False)

    def configure_optimizers(self):
        disc_lr = self.disc_lr if self.disc_lr is not None else self.hparams.lr
        disc_betas = tuple(self.disc_betas)
        if self.use_discriminator:
            opt_g = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
            opt_d = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=disc_lr,
                weight_decay=self.hparams.weight_decay,
                betas=disc_betas,
            )
            return [opt_g, opt_d]
        # Default AdamW optimizer for VQ-VAE training.
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    @staticmethod
    def _lpips_inputs(recon: torch.Tensor, x: torch.Tensor) -> tuple:
        if recon.shape[1] == 1:
            recon = recon.repeat(1, 3, 1, 1)
            x = x.repeat(1, 3, 1, 1)
        return recon, x

    @staticmethod
    def _set_requires_grad(module, requires_grad: bool) -> None:
        for param in module.parameters():
            param.requires_grad = requires_grad


def _as_float(value, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a float, got {value!r}.") from exc
