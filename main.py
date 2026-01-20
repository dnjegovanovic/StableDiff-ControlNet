"""Project CLI entrypoint."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def _run_train_vqvae_lightning(config_path: str) -> None:
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    script_path = Path("scripts/train_vqvae_lightning.py")
    if not script_path.exists():
        raise FileNotFoundError(f"Training script not found: {script_path}")

    # Reuse the existing CLI script in-process for consistent behavior.
    sys.argv = [str(script_path), "--config", str(config_file)]
    runpy.run_path(str(script_path), run_name="__main__")


def _run_train_ddpm_lightning(config_path: str) -> None:
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    script_path = Path("scripts/train_ddpm_lightning.py")
    if not script_path.exists():
        raise FileNotFoundError(f"Training script not found: {script_path}")

    # Reuse the existing CLI script in-process for consistent behavior.
    sys.argv = [str(script_path), "--config", str(config_file)]
    runpy.run_path(str(script_path), run_name="__main__")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_vqvae = subparsers.add_parser(
        "train-vqvae-lightning",
        help="Run VQ-VAE Lightning training with a YAML config.",
    )
    train_vqvae.add_argument(
        "--config",
        required=True,
        help="Path to the VQ-VAE training config YAML.",
    )

    train_ddpm = subparsers.add_parser(
        "train-ddpm-lightning",
        help="Run DDPM Lightning training with a YAML config.",
    )
    train_ddpm.add_argument(
        "--config",
        required=True,
        help="Path to the DDPM training config YAML.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "train-vqvae-lightning":
        _run_train_vqvae_lightning(args.config)
    elif args.command == "train-ddpm-lightning":
        _run_train_ddpm_lightning(args.config)


if __name__ == "__main__":
    main()
