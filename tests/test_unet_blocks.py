import pytest
import torch

from ControlNet.models.UNetBlocks import BottleNeck, DownSamplingBlock, UpSamplingBlock


def test_downsampling_block_output_shape():
    block = DownSamplingBlock(
        in_channels=4,
        out_channels=8,
        time_emb_dim=None,
        num_heads=4,
        down_sample=True,
        use_attn=False,
        grp_norm_chanels=4,
    )
    x = torch.randn(2, 4, 8, 8)
    out = block(x)
    assert out.shape == (2, 8, 4, 4)


def test_downsampling_block_requires_time_emb():
    block = DownSamplingBlock(
        in_channels=4,
        out_channels=8,
        time_emb_dim=16,
        num_heads=4,
        down_sample=False,
        use_attn=False,
        grp_norm_chanels=4,
    )
    x = torch.randn(1, 4, 8, 8)
    with pytest.raises(AssertionError):
        block(x, time_emb=None)


def test_downsampling_block_with_cross_attention():
    block = DownSamplingBlock(
        in_channels=4,
        out_channels=8,
        time_emb_dim=None,
        num_heads=4,
        down_sample=False,
        use_attn=False,
        cross_attn=True,
        cross_cont_dim=6,
        grp_norm_chanels=4,
    )
    x = torch.randn(1, 4, 8, 8)
    context = torch.randn(1, 5, 6)
    out = block(x, context=context)
    assert out.shape == (1, 8, 8, 8)


def test_bottleneck_with_cross_attention():
    block = BottleNeck(
        in_channels=8,
        out_channels=8,
        time_emb_dim=16,
        num_heads=4,
        num_layers=1,
        cross_attn=True,
        cross_cont_dim=6,
    )
    x = torch.randn(2, 8, 4, 4)
    time_emb = torch.randn(2, 16)
    context = torch.randn(2, 7, 6)
    out = block(x, time_emb=time_emb, context=context)
    assert out.shape == (2, 8, 4, 4)


def test_upsampling_block_output_shape_with_skip():
    block = UpSamplingBlock(
        in_channels=8,
        out_channels=8,
        skip_channels=4,
        time_emb_dim=None,
        num_heads=4,
        up_sample=True,
        use_attn=False,
        grp_norm_chanels=4,
    )
    x = torch.randn(1, 8, 4, 4)
    skip = torch.randn(1, 4, 8, 8)
    out = block(x, out_down=skip)
    assert out.shape == (1, 8, 8, 8)
