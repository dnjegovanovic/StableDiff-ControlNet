import math

import pytest
import torch

from ControlNet.models.TimeEmbedding import TimeEmbedding, time_embedding_fun


def test_time_embedding_fun_shape():
    timesteps = torch.tensor([0, 1, 2])
    emb = time_embedding_fun(timesteps, time_embedding_dim=8)
    assert emb.shape == (3, 8)


def test_time_embedding_fun_values():
    timesteps = torch.tensor([1])
    emb = time_embedding_fun(timesteps, time_embedding_dim=4)
    expected = torch.tensor(
        [
            math.sin(1.0),
            math.sin(0.01),
            math.cos(1.0),
            math.cos(0.01),
        ],
        dtype=emb.dtype,
    )
    torch.testing.assert_close(emb.squeeze(0), expected, rtol=1e-6, atol=1e-7)


def test_time_embedding_fun_odd_dim_raises():
    with pytest.raises(AssertionError):
        time_embedding_fun(torch.tensor([1]), time_embedding_dim=3)


def test_time_embedding_module_direct_map_shapes():
    module = TimeEmbedding(n_embd=16, use_direct_map=True)
    x = torch.randn(4, 16)
    out = module(x)
    assert out.shape == (4, 16)


def test_time_embedding_module_projected_shapes():
    module = TimeEmbedding(n_embd=16, use_direct_map=False)
    x = torch.randn(2, 16)
    out = module(x)
    assert out.shape == (2, 64)
