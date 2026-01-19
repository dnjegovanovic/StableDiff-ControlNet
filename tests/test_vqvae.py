import torch

from ControlNet.modules.VQVAE import VectorQuantizedVAE


def _make_vqvae_config():
    return {
        "down_channels": [8, 16],
        "mid_channels": [16, 16],
        "down_sample": [True],
        "num_down_layers": 1,
        "num_mid_layers": 1,
        "num_up_layers": 1,
        "attn_down": [False],
        "z_channels": 4,
        "codebook_size": 8,
        "norm_channels": 4,
        "num_heads": 2,
    }


def test_vqvae_forward_shapes():
    model = VectorQuantizedVAE(3, VQVAE=_make_vqvae_config())
    x = torch.randn(2, 3, 8, 8)
    recon, quantized, losses = model(x)

    assert recon.shape == x.shape
    assert quantized.shape == (2, 4, 4, 4)
    assert set(losses.keys()) == {"codebook", "commitment"}


def test_vqvae_encode_with_indices():
    model = VectorQuantizedVAE(3, VQVAE=_make_vqvae_config())
    x = torch.randn(1, 3, 8, 8)
    quantized, losses, indices = model.encode_with_indices(x)

    assert quantized.shape == (1, 4, 4, 4)
    assert indices.shape == (1, 4, 4)
    assert indices.max().item() < model.codebook_size
    assert set(losses.keys()) == {"codebook", "commitment"}


def test_vqvae_ema_updates_codebook():
    config = _make_vqvae_config()
    config["use_ema"] = True
    config["ema_decay"] = 0.0
    model = VectorQuantizedVAE(3, VQVAE=config)
    model.train()
    x = torch.randn(2, 3, 8, 8)
    initial = model.codebook.weight.detach().clone()
    model.encode(x)
    assert not torch.allclose(model.codebook.weight, initial)
