"""Train a DDPM UNet on VQ-VAE latents with PyTorch Lightning."""

from __future__ import annotations

import argparse
import os
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
from ControlNet.modules.DDPMLightning import DDPMLightning


class DDPMDataModule(pl.LightningDataModule):
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
        if isinstance(batch[0], tuple):
            images = torch.stack([item[0] for item in batch], dim=0)
            conds = [item[1] for item in batch]
            cond_input = DDPMDataModule._merge_conditions(conds)
            if cond_input:
                return {"x": images, "cond_input": cond_input}
            return {"x": images}
        images = torch.stack(batch, dim=0)
        return {"x": images}

    @staticmethod
    def _merge_conditions(conds):
        merged: Dict[str, Any] = {}
        for cond in conds:
            if not isinstance(cond, dict):
                continue
            for key, value in cond.items():
                merged.setdefault(key, []).append(value)
        if not merged:
            return {}

        output: Dict[str, Any] = {}
        for key, values in merged.items():
            first = values[0]
            if torch.is_tensor(first):
                output[key] = torch.stack(values, dim=0)
            elif isinstance(first, (int, float, bool)):
                output[key] = torch.tensor(values)
            elif isinstance(first, str):
                output[key] = list(values)
            else:
                try:
                    output[key] = torch.stack(
                        [torch.as_tensor(v) for v in values], dim=0
                    )
                except (TypeError, ValueError):
                    output[key] = list(values)
        return output

    def train_dataloader(self):
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
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self._collate,
        )


class DDPMSampleCallback(pl.Callback):
    def __init__(
        self,
        output_dir: Path,
        every_n_epochs: int,
        max_images: int = 8,
        log_to_tensorboard: bool = True,
        tensorboard_tag: str = "ddpm/samples",
    ):
        self.output_dir = output_dir
        self.every_n_epochs = int(every_n_epochs)
        self.max_images = int(max_images)
        self.log_to_tensorboard = log_to_tensorboard
        self.tensorboard_tag = tensorboard_tag

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: DDPMLightning):
        if self.every_n_epochs <= 0:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        if trainer.sanity_checking:
            return
        batch = _get_first_batch(trainer)
        if batch is None or "x" not in batch:
            return

        images = batch["x"]
        if not torch.is_tensor(images):
            return
        count = min(self.max_images, images.shape[0])
        if count <= 0:
            return
        images = images[:count].to(pl_module.device)
        raw_cond = batch.get("cond_input", {})

        was_training = pl_module.training
        pl_module.eval()
        pl_module._freeze_vqvae()
        with torch.no_grad():
            latents, _ = pl_module.vqvae.encode(images)
            cond_input = None
            if pl_module.condition_config:
                cond_input = pl_module._build_cond_input(raw_cond, latents)
            sample_latents = pl_module.sample_latents(latents.shape, cond_input=cond_input)
            decoded = pl_module.vqvae.decode(sample_latents)
        if was_training:
            pl_module.train()

        decoded_vis = (decoded.detach().cpu().clamp(-1, 1) + 1) / 2
        grid = vutils.make_grid(decoded_vis, nrow=count)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"ddpm_sample_epoch_{trainer.current_epoch + 1}.png"
        vutils.save_image(grid, output_path)
        if self.log_to_tensorboard:
            tb_experiment = _get_tensorboard_experiment(trainer)
            if tb_experiment is not None:
                tb_experiment.add_image(
                    self.tensorboard_tag,
                    grid,
                    global_step=trainer.current_epoch + 1,
                    dataformats="CHW",
                )


def _get_first_batch(trainer: pl.Trainer) -> Optional[Dict[str, Any]]:
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
    logger = trainer.logger
    if logger is None or not hasattr(logger, "experiment"):
        return None
    experiment = logger.experiment
    if hasattr(experiment, "add_image"):
        return experiment
    return None


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        try:
            return yaml.safe_load(handle)
        except yaml.YAMLError as error:
            raise ValueError(f"Failed to parse config: {error}") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DDPM UNet with Lightning.")
    parser.add_argument("--config", dest="config_path", required=True, type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config_path))

    dataset_cfg = config["dataset_params"]
    vqvae_cfg = config["VQVAE"]
    unet_cfg = config["UnetParams"]
    ddpm_cfg = config.get("ddpm_params", {})
    train_cfg = config["train_params"]

    pl.seed_everything(train_cfg.get("seed", 0), workers=True)

    vqvae_ckpt_path = train_cfg.get("vqvae_ckpt_path")
    if not vqvae_ckpt_path:
        raise ValueError("train_params.vqvae_ckpt_path must be set for DDPM training.")
    ckpt_path = Path(vqvae_ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"VQ-VAE checkpoint not found: {ckpt_path}")

    model = DDPMLightning(
        unet_params=unet_cfg,
        vqvae_params=vqvae_cfg,
        ddpm_params=ddpm_cfg,
        input_channels=dataset_cfg.get("im_channels"),
        vqvae_ckpt_path=str(ckpt_path),
        lr=train_cfg.get("ddpm_lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        condition_config=dataset_cfg.get("condition_config"),
    )

    datamodule = DDPMDataModule(dataset_cfg)

    output_dir = Path(train_cfg.get("output_dir", "outputs"))
    if output_dir.name != "ddpm":
        output_dir = output_dir / "ddpm"
    run_id = os.environ.get("CONTROLNET_RUN_ID")
    samples_dir = output_dir / "ddpm_samples"
    if run_id:
        samples_dir = samples_dir / run_id
    use_tensorboard = bool(train_cfg.get("use_tensorboard", True))
    tb_logger = None
    if use_tensorboard:
        tb_logger = TensorBoardLogger(
            save_dir=str(output_dir),
            name=str(train_cfg.get("tensorboard_dir", "tensorboard")),
        )

    callbacks = []
    if train_cfg.get("save_samples_every_n_epochs", 0) > 0:
        callbacks.append(
            DDPMSampleCallback(
                samples_dir,
                every_n_epochs=train_cfg["save_samples_every_n_epochs"],
                max_images=train_cfg.get("max_sample_images", 8),
                log_to_tensorboard=use_tensorboard,
            )
        )
    trainer = pl.Trainer(
        max_epochs=train_cfg.get("ddpm_epochs", 1),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=train_cfg.get("devices", 1),
        accumulate_grad_batches=train_cfg.get("accumulate_grad_batches", 1),
        precision=train_cfg.get("precision", 32),
        log_every_n_steps=train_cfg.get("log_every_n_steps", 50),
        default_root_dir=str(output_dir),
        logger=tb_logger,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
