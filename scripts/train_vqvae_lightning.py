"""Train VQ-VAE with PyTorch Lightning."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torchvision.utils as vutils
import yaml
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from ControlNet.datasets.MnistDatasets import MNISTDataset
from ControlNet.modules.VQVAELightning import VQVAELightning


class VQVAEDataModule(pl.LightningDataModule):
    def __init__(self, dataset_cfg: Dict[str, Any]):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.batch_size = dataset_cfg.get("batch_size", 8)
        self.num_workers = dataset_cfg.get("num_workers", 0)
        self.train_root = dataset_cfg["train_root"]
        self.val_root = dataset_cfg.get("val_root")
        self.val_split = float(dataset_cfg.get("val_split", 0.1))
        self.seed = int(dataset_cfg.get("seed", 0))
        self.image_extension = dataset_cfg.get("image_extension", "png")
        self.condition_config = dataset_cfg.get("condition_config")
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        # Build datasets and optionally split train into train/val.
        self.train_dataset = MNISTDataset(
            dataset_split="train",
            data_root=self.train_root,
            image_extension=self.image_extension,
            condition_config=self.condition_config,
        )
        if self.val_root is not None:
            self.val_dataset = MNISTDataset(
                dataset_split="val",
                data_root=self.val_root,
                image_extension=self.image_extension,
                condition_config=self.condition_config,
            )
        elif self.val_split > 0:
            total = len(self.train_dataset)
            val_size = max(1, int(total * self.val_split))
            train_size = max(1, total - val_size)
            if train_size + val_size > total:
                val_size = total - train_size
            generator = torch.Generator().manual_seed(self.seed)
            self.train_dataset, self.val_dataset = random_split(
                self.train_dataset, [train_size, val_size], generator=generator
            )

    @staticmethod
    def _collate(batch):
        # Support datasets that return (image, conditioning) tuples.
        if isinstance(batch[0], tuple):
            images = torch.stack([item[0] for item in batch], dim=0)
        else:
            images = torch.stack(batch, dim=0)
        return {"x": images}

    def train_dataloader(self):
        # Shuffle training data and enable pinned memory when CUDA is available.
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self._collate,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        # Deterministic validation loader.
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self._collate,
        )


class ReconstructionCallback(pl.Callback):
    def __init__(
        self,
        output_dir: Path,
        every_n_epochs: int,
        max_images: int = 8,
        log_to_tensorboard: bool = True,
        tensorboard_tag: str = "vqvae/reconstructions",
    ):
        self.output_dir = output_dir
        self.every_n_epochs = every_n_epochs
        self.max_images = max_images
        self.log_to_tensorboard = log_to_tensorboard
        self.tensorboard_tag = tensorboard_tag

    def on_validation_epoch_end(self, trainer, pl_module):
        # Log reconstructions at a fixed epoch cadence.
        if self.every_n_epochs <= 0:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        batch = _get_first_batch(trainer)
        if batch is None:
            return
        inputs = batch["x"].to(pl_module.device)

        with torch.no_grad():
            recons, _, _ = pl_module(inputs)

        # Arrange input and reconstruction pairs into a single image grid.
        count = min(self.max_images, inputs.size(0))
        inputs_vis = (inputs[:count].detach().cpu() + 1) / 2
        recons_vis = (recons[:count].detach().cpu().clamp(-1, 1) + 1) / 2
        grid = vutils.make_grid(torch.cat([inputs_vis, recons_vis], dim=0), nrow=count)
        output_path = self.output_dir / f"vqvae_recon_epoch_{trainer.current_epoch + 1}.png"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        vutils.save_image(grid, output_path)
        if self.log_to_tensorboard:
            tb_experiment = _get_tensorboard_experiment(trainer)
            if tb_experiment is not None:
                # Lightning exposes a SummaryWriter-compatible experiment for TensorBoard.
                tb_experiment.add_image(
                    self.tensorboard_tag,
                    grid,
                    global_step=trainer.current_epoch + 1,
                    dataformats="CHW",
                )


def _get_first_batch(trainer: pl.Trainer) -> Optional[Dict[str, Any]]:
    # Prefer validation batches for visualization, with a training fallback.
    dataloaders = None
    if trainer.val_dataloaders:
        dataloaders = trainer.val_dataloaders
    elif trainer.train_dataloader is not None:
        dataloaders = trainer.train_dataloader
    if dataloaders is None:
        return None
    if isinstance(dataloaders, (list, tuple)):
        dataloader = dataloaders[0]
    else:
        dataloader = dataloaders
    return next(iter(dataloader))


def _get_tensorboard_experiment(trainer: pl.Trainer):
    # Safely access the underlying TensorBoard SummaryWriter.
    logger = trainer.logger
    if logger is None or not hasattr(logger, "experiment"):
        return None
    experiment = logger.experiment
    if hasattr(experiment, "add_image"):
        return experiment
    return None


def load_config(path: Path) -> Dict[str, Any]:
    # Load YAML config with a friendly error if parsing fails.
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        try:
            return yaml.safe_load(handle)
        except yaml.YAMLError as error:
            raise ValueError(f"Failed to parse config: {error}") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VQ-VAE with Lightning.")
    parser.add_argument("--config", dest="config_path", required=True, type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Split the config into dataset/model/train sections.
    config = load_config(Path(args.config_path))

    dataset_cfg = config["dataset_params"]
    vqvae_cfg = config["VQVAE"]
    train_cfg = config["train_params"]

    pl.seed_everything(train_cfg.get("seed", 0), workers=True)

    # Build the LightningModule wrapper around the VQ-VAE model.
    model = VQVAELightning(
        vqvae_params=vqvae_cfg,
        input_channels=dataset_cfg.get("im_channels"),
        lr=train_cfg.get("autoencoder_lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        recon_loss=train_cfg.get("recon_loss", "mse"),
        codebook_weight=train_cfg.get("codebook_weight", 1.0),
        commitment_weight=train_cfg.get("commitment_beta", 0.25),
        grad_accum_steps=train_cfg.get("autoencoder_acc_steps", 1),
        use_discriminator=train_cfg.get("use_discriminator", False),
        disc_start=train_cfg.get("disc_start", 0),
        disc_weight=train_cfg.get("disc_weight", 1.0),
        disc_lr=train_cfg.get("disc_lr"),
        disc_betas=tuple(train_cfg.get("disc_betas", (0.5, 0.999))),
        use_lpips=train_cfg.get("use_lpips", False),
        perceptual_weight=train_cfg.get("perceptual_weight", 1.0),
        lpips_auto_download=train_cfg.get("lpips_auto_download", True),
        lpips_weights_url=train_cfg.get("lpips_weights_url"),
        lpips_backbone_pretrained=train_cfg.get("lpips_backbone_pretrained", True),
    )

    # DataModule handles dataset creation and batching.
    datamodule = VQVAEDataModule(dataset_cfg)

    output_dir = Path(train_cfg.get("output_dir", "outputs"))
    use_tensorboard = bool(train_cfg.get("use_tensorboard", True))
    tb_logger = None
    if use_tensorboard:
        # Write TensorBoard logs under output_dir/tensorboard_dir.
        tb_logger = TensorBoardLogger(
            save_dir=str(output_dir),
            name=str(train_cfg.get("tensorboard_dir", "tensorboard")),
        )
    callbacks = []
    if train_cfg.get("save_recon_every_n_epochs", 0) > 0:
        # Save reconstruction grids to disk and optionally to TensorBoard.
        callbacks.append(
            ReconstructionCallback(
                output_dir / "vqvae_reconstructions",
                every_n_epochs=train_cfg["save_recon_every_n_epochs"],
                max_images=train_cfg.get("max_recon_images", 8),
                log_to_tensorboard=use_tensorboard,
            )
        )

    trainer = pl.Trainer(
        max_epochs=train_cfg.get("autoencoder_epochs", 1),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=train_cfg.get("devices", 1),
        accumulate_grad_batches=1,
        precision=train_cfg.get("precision", 32),
        log_every_n_steps=train_cfg.get("log_every_n_steps", 50),
        default_root_dir=str(output_dir),
        callbacks=callbacks,
        logger=tb_logger,
    )

    # Launch training loop.
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
