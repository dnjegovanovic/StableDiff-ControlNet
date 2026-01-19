# StableDiff-ControlNet
This repo implements core ControlNet-style building blocks in PyTorch: time embeddings, multi-head attention, UNet blocks, and a configurable UNet with optional class/text/image conditioning.

The focus is on the model components themselves (not full training/inference pipelines).

## Implemented Components

### 1) Time Embedding (`ControlNet/models/TimeEmbedding.py`)
Theory: Diffusion models use a timestep embedding to condition the network on the noise level. A standard approach is sinusoidal embeddings (sin/cos at log-spaced frequencies), followed by an MLP to give the model a learnable projection.

Practice:
- `time_embedding_fun(timestep, time_embedding_dim)` returns a `(B, time_embedding_dim)` tensor.
- `TimeEmbedding(n_embd, use_direct_map)` applies two linear layers with SiLU.

```python
import torch

from ControlNet.models.TimeEmbedding import time_embedding_fun, TimeEmbedding

t = torch.tensor([1, 10, 100])
emb = time_embedding_fun(t, time_embedding_dim=320)  # (3, 320)
proj = TimeEmbedding(n_embd=320, use_direct_map=True)
emb_proj = proj(emb)  # (3, 320)
```

### 2) Multi-Head Attention (`ControlNet/models/MultiHeadAttention.py`)
Theory: Multi-head attention computes attention weights per head using scaled dot-product attention, letting the model attend to different subspaces.
- Self-attention: query, key, value come from the same input.
- Cross-attention: query comes from one stream, key/value from another (for example, text context).

Practice: Implemented with `torch.nn.functional.scaled_dot_product_attention`.

```python
import torch

from ControlNet.models.MultiHeadAttention import (
    MultiHeadSelfAttention,
    MultiHeadCrossAttention,
)

self_attn = MultiHeadSelfAttention(num_heads=4, embed_dim=64)
x = torch.randn(2, 16, 64)
y = self_attn(x, apply_causal_mask=False)  # (2, 16, 64)

cross_attn = MultiHeadCrossAttention(num_heads=4, embed_dim=64, cross_dim=32)
context = torch.randn(2, 10, 32)
y = cross_attn(x, context)  # (2, 16, 64)
```

### 3) UNet Blocks (`ControlNet/models/UNetBlocks.py`)
Theory: A UNet block combines residual convolutions and optional attention.
- DownSamplingBlock: residual convs + time embedding + optional attention + optional downsample.
- BottleNeck: residual blocks with attention at the lowest resolution.
- UpSamplingBlock: optional upsample, skip concatenation, residual convs, optional attention.

Practice: Each block accepts `(x, time_emb, context)` and returns a tensor with updated channels/spatial size.

### 4) UNet Module (`ControlNet/modules/UNet.py`)
Theory: Encoder-decoder UNet with skip connections, conditioned on time.
- Class conditioning: class embedding added to time embedding.
- Text conditioning: cross-attention uses text hidden states.
- Image conditioning: image condition is projected and concatenated at the input.

Practice: `UNet` expects a `UnetParams` dictionary with the keys shown below.

```python
import torch

from ControlNet.modules.UNet import UNet

params = {
    "down_channels": [16, 32],
    "mid_channels": [32, 16],
    "down_sample": [True],
    "attn_down": [False],
    "time_emb_dim": 32,
    "im_channels": 3,
    "num_down_layers": 1,
    "num_mid_layers": 1,
    "num_up_layers": 1,
}

model = UNet(UnetParams=params)
x = torch.randn(2, 3, 32, 32)
out = model(x, t=10)
```

### 5) Config Helpers (`ControlNet/utils/config.py`)
Small validation helpers are used by the UNet to check conditioning config and inputs:
- `validate_class_config`, `validate_text_config`, `validate_image_config`
- `validate_class_conditional_input`

## Conditioning Examples

### Class Conditioning
```python
import torch

from ControlNet.modules.UNet import UNet

params = {
    "down_channels": [16, 32],
    "mid_channels": [32, 16],
    "down_sample": [True],
    "attn_down": [False],
    "time_emb_dim": 32,
    "im_channels": 3,
    "num_down_layers": 1,
    "num_mid_layers": 1,
    "num_up_layers": 1,
    "condition_config": {
        "condition_types": ["class"],
        "class_condition_config": {"num_classes": 5},
    },
}
model = UNet(UnetParams=params)
cond_input = {"class": torch.rand(2, 5)}
out = model(torch.randn(2, 3, 32, 32), t=5, cond_input=cond_input)
```

### Text Conditioning
```python
import torch

from ControlNet.modules.UNet import UNet

params = {
    "down_channels": [16, 32],
    "mid_channels": [32, 16],
    "down_sample": [True],
    "attn_down": [False],
    "time_emb_dim": 32,
    "im_channels": 3,
    "num_down_layers": 1,
    "num_mid_layers": 1,
    "num_up_layers": 1,
    "condition_config": {
        "condition_types": ["text"],
        "text_condition_config": {"text_embed_dim": 64},
    },
}
model = UNet(UnetParams=params)
cond_input = {"text": torch.randn(2, 10, 64)}
out = model(torch.randn(2, 3, 32, 32), t=5, cond_input=cond_input)
```

### Image Conditioning
```python
import torch

from ControlNet.modules.UNet import UNet

params = {
    "down_channels": [16, 32],
    "mid_channels": [32, 16],
    "down_sample": [True],
    "attn_down": [False],
    "time_emb_dim": 32,
    "im_channels": 3,
    "num_down_layers": 1,
    "num_mid_layers": 1,
    "num_up_layers": 1,
    "condition_config": {
        "condition_types": ["image"],
        "image_condition_config": {
            "image_condition_input_channels": 1,
            "image_condition_output_channels": 2,
        },
    },
}
model = UNet(UnetParams=params)
cond_input = {"image": torch.randn(2, 1, 64, 64)}
out = model(torch.randn(2, 3, 32, 32), t=5, cond_input=cond_input)
```

## Tests
```bash
pytest tests
```

## VQ-VAE Lightning Training
Use the provided config template and training script:

```bash
python scripts/train_vqvae_lightning.py --config config_vqvae.yml
```

Config template: `config_vqvae.yml`

## Notes
- `time_emb_dim` must be even for sinusoidal embeddings.
- `down_channels[0]` is used as the decoder output channel width.
- `torch` and `einops` are required for the UNet module.
