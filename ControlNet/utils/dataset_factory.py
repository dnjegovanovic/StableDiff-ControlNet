from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from con_stable_diff_model.datasets.MnistDatasets import MNISTDataset
from con_stable_diff_model.datasets.CelebDataset import CelebDataset


def build_dataset(
    dataset_cfg: Dict[str, Any],
    *,
    split: str = "train",
    condition_config: Optional[Dict[str, Any]] = None,
    use_latents_override: Optional[bool] = None,
    latent_path_override: Optional[str | Path] = None,
):
    """Instantiate a dataset defined in the configuration dictionary."""

    name = dataset_cfg["name"].lower()

    if name == "mnist":
        return MNISTDataset(
            dataset_split=split,
            data_root=dataset_cfg["im_path"],
            image_extension=dataset_cfg.get("im_ext", "png"),
            condition_config=condition_config,
        )

    if name in {"celeb", "celebahq", "celebhq"}:
        use_latents = (
            use_latents_override
            if use_latents_override is not None
            else dataset_cfg.get("use_latents", False)
        )
        latent_path = (
            latent_path_override
            if latent_path_override is not None
            else dataset_cfg.get("latent_path")
        )

        return CelebDataset(
            split=split,
            im_path=dataset_cfg["im_path"],
            im_size=dataset_cfg.get("im_size", 256),
            im_channels=dataset_cfg.get("im_channels", 3),
            im_ext=dataset_cfg.get("im_ext", "jpg"),
            use_latents=use_latents,
            latent_path=latent_path,
            condition_config=condition_config,
        )

    raise ValueError(f"Unsupported dataset '{dataset_cfg['name']}'.")
