import torch

from ControlNet.modules.UNet import UNet


def _make_unet_params(
    *,
    down_channels=(16, 32),
    mid_channels=(32, 16),
    time_emb_dim=32,
    im_channels=3,
    condition_config=None,
):
    params = {
        "down_channels": list(down_channels),
        "mid_channels": list(mid_channels),
        "down_sample": [True] * (len(down_channels) - 1),
        "attn_down": [False] * (len(down_channels) - 1),
        "time_emb_dim": time_emb_dim,
        "im_channels": im_channels,
        "num_down_layers": 1,
        "num_mid_layers": 1,
        "num_up_layers": 1,
    }
    if condition_config is not None:
        params["condition_config"] = condition_config
    return params


def test_unet_forward_no_condition():
    params = _make_unet_params()
    model = UNet(UnetParams=params)
    x = torch.randn(2, 3, 8, 8)
    out = model(x, t=1)
    assert out.shape == x.shape


def test_unet_output_channels_follow_down_channels():
    params = _make_unet_params(down_channels=(32, 64), mid_channels=(64, 32))
    model = UNet(UnetParams=params)
    assert model.norm_out.num_channels == 32


def test_unet_forward_with_class_condition():
    condition_config = {
        "condition_types": ["class"],
        "class_condition_config": {"num_classes": 5},
    }
    params = _make_unet_params(condition_config=condition_config)
    model = UNet(UnetParams=params)
    x = torch.randn(2, 3, 8, 8)
    cond_input = {"class": torch.rand(2, 5)}
    out = model(x, t=1, cond_input=cond_input)
    assert out.shape == x.shape


def test_unet_forward_with_text_condition():
    condition_config = {
        "condition_types": ["text"],
        "text_condition_config": {"text_embed_dim": 32},
    }
    params = _make_unet_params(condition_config=condition_config)
    model = UNet(UnetParams=params)
    x = torch.randn(1, 3, 8, 8)
    cond_input = {"text": torch.randn(1, 4, 32)}
    out = model(x, t=1, cond_input=cond_input)
    assert out.shape == x.shape


def test_unet_forward_with_image_condition():
    condition_config = {
        "condition_types": ["image"],
        "image_condition_config": {
            "image_condition_input_channels": 1,
            "image_condition_output_channels": 2,
        },
    }
    params = _make_unet_params(condition_config=condition_config)
    model = UNet(UnetParams=params)
    x = torch.randn(1, 3, 8, 8)
    cond_input = {"image": torch.randn(1, 1, 16, 16)}
    out = model(x, t=1, cond_input=cond_input)
    assert out.shape == x.shape
