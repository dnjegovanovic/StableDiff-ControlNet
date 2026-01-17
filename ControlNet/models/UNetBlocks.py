import torch
import torch.nn as nn

from .MultiHeadAttention import MultiHeadSelfAttention, MultiHeadCrossAttention

"""
UNet Architecture with Attention Blocks for Diffusion Models

This module implements a UNet-like architecture with residual blocks, attention mechanisms,
and time embedding for diffusion models. It consists of downsampling blocks, a bottleneck,
and upsampling blocks with skip connections.
"""


def _assert_group_norm_divisible(num_groups: int, channels: int, label: str) -> None:
    assert channels % num_groups == 0, (
        f"{label} ({channels}) must be divisible by grp_norm_chanels ({num_groups})."
    )


class DownSamplingBlock(nn.Module):
    """
    A UNet-style downsampling block that combines:
    - Residual blocks with time embeddings
    - Optional attention mechanisms
    - Optional spatial downsampling
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int = None,
        num_heads: int = 4,
        down_sample: bool = True,
        custom_mha: bool = True,
        num_layers: int = 1,
        use_attn: bool = True,
        grp_norm_chanels: int = 8,
        cross_attn=False,
        cross_cont_dim=None,
        custom_cross_attn=True,
    ):
        super().__init__()

        # Store configuration parameters
        self.down_sample = down_sample  # Whether to reduce spatial dimensions
        self.custom_mha = custom_mha  # Use custom or PyTorch's MHA implementation
        self.num_layers = num_layers  # Number of residual/attention layers in block
        self.time_emb_dim = time_emb_dim
        self.use_attn = use_attn
        self.cross_attn = cross_attn
        self.cross_cont_dim = cross_cont_dim
        self.custom_cross_attn = custom_cross_attn

        _assert_group_norm_divisible(grp_norm_chanels, in_channels, "in_channels")
        _assert_group_norm_divisible(grp_norm_chanels, out_channels, "out_channels")

        # First part of residual block (per layer)
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(
                        grp_norm_chanels, in_channels if i == 0 else out_channels
                    ),  # Normalize input
                    nn.SiLU(),  # Activation function
                    nn.Conv2d(  # Channel transformation
                        in_channels if i == 0 else out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
                for i in range(num_layers)
            ]
        )

        # Time embedding processing (per layer)
        if self.time_emb_dim:
            self.time_emb_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.SiLU(),  # Activation for time embedding
                        nn.Linear(
                            time_emb_dim, out_channels
                        ),  # Project time emb to channel space
                    )
                    for _ in range(num_layers)
                ]
            )

        # Second part of residual block (per layer)
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(grp_norm_chanels, out_channels),  # Normalization
                    nn.SiLU(),  # Activation
                    nn.Conv2d(  # Final convolution in residual path
                        out_channels, out_channels, kernel_size=3, stride=1, padding=1
                    ),
                )
                for _ in range(num_layers)
            ]
        )

        # Attention mechanism components (per layer)
        if self.use_attn:
            self.attention_norm = nn.ModuleList(
                [
                    nn.GroupNorm(
                        grp_norm_chanels, out_channels
                    )  # Normalization before attention
                    for _ in range(num_layers)
                ]
            )

            # Choose attention implementation
            if custom_mha:
                self.attention = nn.ModuleList(
                    [
                        MultiHeadSelfAttention(
                            num_heads, out_channels, input_proj_bias=False
                        )
                        for _ in range(num_layers)
                    ]
                )
            else:
                self.attention = nn.ModuleList(
                    [
                        nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                        for _ in range(num_layers)
                    ]
                )

        #### Cross attention ###
        if self.cross_attn:
            assert (
                self.cross_cont_dim is not None
            ), "Context dim must be passed for cross attention."
            self.cross_attention_norm = nn.ModuleList(
                [
                    nn.GroupNorm(
                        grp_norm_chanels, out_channels
                    )  # Normalization before attention
                    for _ in range(num_layers)
                ]
            )

            # Choose attention implementation
            if custom_cross_attn:
                self.cross_attention = nn.ModuleList(
                    [
                        MultiHeadCrossAttention(
                            num_heads,
                            out_channels,
                            self.cross_cont_dim,
                            input_proj_bias=False,
                        )
                        for _ in range(num_layers)
                    ]
                )
            else:
                self.context_proj = nn.ModuleList(
                    [
                        nn.Linear(self.cross_cont_dim, out_channels)
                        for _ in range(num_layers)
                    ]
                )
                self.cross_attention = nn.ModuleList(
                    [
                        nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                        for _ in range(num_layers)
                    ]
                )

        # Residual connection projection (per layer)
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(  # 1x1 conv for channel matching
                    in_channels if i == 0 else out_channels, out_channels, kernel_size=1
                )
                for i in range(num_layers)
            ]
        )

        # Final downsampling layer
        self.down_sample_conv = (
            nn.Conv2d(  # Spatial downsampling (halves resolution)
                out_channels, out_channels, kernel_size=4, stride=2, padding=1
            )
            if self.down_sample
            else nn.Identity()  # Skip downsampling
        )

    def forward(
        self, x: torch.Tensor, time_emb: torch.Tensor = None, context=None
    ) -> torch.Tensor:
        if self.time_emb_dim:
            assert time_emb is not None, "time_emb must be provided when time_emb_dim is set."
        out = x  # Preserve original input for residual connection

        for i in range(self.num_layers):
            # Residual block processing
            resnet_input = out  # Store input for residual connection

            # First convolution path
            out = self.resnet_conv_first[i](out)

            if self.time_emb_dim:
                # Add time embedding (broadcasted to spatial dimensions)
                out = out + self.time_emb_layers[i](time_emb)[:, :, None, None]

            # Second convolution path
            out = self.resnet_conv_second[i](out)

            # Residual connection with projection
            out = out + self.residual_input_conv[i](resnet_input)

            # Attention processing
            if self.use_attn:
                batch_size, chanels, h, w = out.shape
                in_attn = out.reshape(
                    batch_size, chanels, h * w
                )  # Flatten spatial dims
                in_attn = self.attention_norm[i](in_attn)
                if self.custom_mha:
                    # Custom attention expects (batch, seq_len, channels)
                    in_attn = in_attn.transpose(-1, -2)
                    out_attn = self.attention[i](in_attn)
                    out_attn = out_attn.transpose(-1, -2)
                else:
                    # PyTorch MHA expects (batch, seq_len, channels)
                    in_attn = in_attn.transpose(1, 2)
                    out_attn, _ = self.attention[i](in_attn, in_attn, in_attn)
                    out_attn = out_attn.transpose(1, 2)

                # Reshape back to original dimensions
                out_attn = out_attn.reshape(batch_size, chanels, h, w)
                out = out + out_attn  # Add attention output

            # Cross Attention processing
            if self.cross_attn:
                assert context is not None, "Cross attention requires a context tensor."
                batch_size, chanels, h, w = out.shape
                in_attn = out.reshape(
                    batch_size, chanels, h * w
                )  # Flatten spatial dims
                in_attn = self.cross_attention_norm[i](in_attn)
                if self.custom_mha:
                    # Custom attention expects (batch, seq_len, channels)
                    in_attn = in_attn.transpose(-1, -2)
                    out_attn = self.cross_attention[i](in_attn, context)
                    out_attn = out_attn.transpose(-1, -2)
                else:
                    # PyTorch MHA expects (batch, seq_len, channels)
                    in_attn = in_attn.transpose(1, 2)
                    contex_proj = self.context_proj[i](context)
                    out_attn, _ = self.cross_attention[i](
                        in_attn, contex_proj, contex_proj
                    )
                    out_attn = out_attn.transpose(1, 2)

                # Reshape back to original dimensions
                out_attn = out_attn.reshape(batch_size, chanels, h, w)
                out = out + out_attn  # Add attention output

        # Final downsampling
        out = self.down_sample_conv(out)
        assert (
            out.ndim == 4
        ), "Donwsample output dont haher 4 dim [batch, channels, H, W]"
        return out


class BottleNeck(nn.Module):
    """
    UNet bottleneck block that processes features at lowest resolution with:
    - Multiple residual blocks with time embeddings
    - Attention mechanisms between residual blocks
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int = None,
        num_heads: int = 4,
        custom_mha: bool = True,
        num_layers: int = 1,
        grp_norm_chanels: int = 8,
        cross_attn=False,
        cross_cont_dim=None,
        custom_cross_attn=True,
    ):
        super().__init__()

        # Configuration parameters
        self.custom_mha = custom_mha  # Choice of attention implementation
        self.num_heads = num_heads  # Number of attention heads
        self.num_layers = num_layers  # Number of residual-attention layers
        self.time_emb_dim = time_emb_dim
        self.cross_attn = cross_attn
        self.cross_cont_dim = cross_cont_dim
        self.custom_cross_attn = custom_cross_attn

        _assert_group_norm_divisible(grp_norm_chanels, in_channels, "in_channels")
        _assert_group_norm_divisible(grp_norm_chanels, out_channels, "out_channels")

        # First residual block components (N+1 layers)
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(
                        grp_norm_chanels, in_channels if i == 0 else out_channels
                    ),  # Normalization
                    nn.SiLU(),  # Activation
                    nn.Conv2d(  # Channel transformation
                        in_channels if i == 0 else out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
                for i in range(num_layers + 1)  # +1 for initial block
            ]
        )

        if self.time_emb_dim:
            # Time embedding processing (N+1 layers)
            self.time_emb_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.SiLU(),  # Time embedding activation
                        nn.Linear(
                            time_emb_dim, out_channels
                        ),  # Project to channel space
                    )
                    for _ in range(num_layers + 1)
                ]
            )

        # Second residual path components (N+1 layers)
        self.resnet_conv_sec = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(grp_norm_chanels, out_channels),  # Normalization
                    nn.SiLU(),  # Activation
                    nn.Conv2d(  # Final convolution
                        out_channels, out_channels, kernel_size=3, stride=1, padding=1
                    ),
                )
                for _ in range(num_layers + 1)
            ]
        )

        # Attention normalization (N layers)
        self.attention_norm = nn.ModuleList(
            [
                nn.GroupNorm(
                    grp_norm_chanels, out_channels
                )  # Pre-attention normalization
                for _ in range(num_layers)
            ]
        )

        # Attention mechanism (N layers)
        if custom_mha:
            self.attention = nn.ModuleList(
                [
                    MultiHeadSelfAttention(  # Custom implementation
                        num_heads, out_channels, input_proj_bias=False
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.attention = nn.ModuleList(
                [
                    nn.MultiheadAttention(  # PyTorch native
                        out_channels, num_heads, batch_first=True
                    )
                    for _ in range(num_layers)
                ]
            )

        #### Cross attention ###
        if self.cross_attn:
            assert (
                self.cross_cont_dim is not None
            ), "Context dim must be passed for cross attention."
            self.cross_attention_norm = nn.ModuleList(
                [
                    nn.GroupNorm(
                        grp_norm_chanels, out_channels
                    )  # Normalization before attention
                    for _ in range(num_layers)
                ]
            )

            # Choose attention implementation
            if custom_cross_attn:
                self.cross_attention = nn.ModuleList(
                    [
                        MultiHeadCrossAttention(
                            num_heads,
                            out_channels,
                            self.cross_cont_dim,
                            input_proj_bias=False,
                        )
                        for _ in range(num_layers)
                    ]
                )
            else:
                self.context_proj = nn.ModuleList(
                    [
                        nn.Linear(self.cross_cont_dim, out_channels)
                        for _ in range(num_layers)
                    ]
                )
                self.cross_attention = nn.ModuleList(
                    [
                        nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                        for _ in range(num_layers)
                    ]
                )

        # Residual connection projections (N+1 layers)
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(  # 1x1 conv for channel matching
                    in_channels if i == 0 else out_channels, out_channels, kernel_size=1
                )
                for i in range(num_layers + 1)
            ]
        )

    def forward(
        self, x: torch.Tensor, time_emb: torch.Tensor = None, context=None
    ) -> torch.Tensor:
        if self.time_emb_dim:
            assert time_emb is not None, "time_emb must be provided when time_emb_dim is set."
        out = x  # Initial feature map

        # First residual block
        resnet_in = out  # Save input for residual connection
        out = self.resnet_conv_first[0](out)  # First convolution path
        if self.time_emb_dim:
            out = (
                out + self.time_emb_layers[0](time_emb)[:, :, None, None]
            )  # Add time emb
        out = self.resnet_conv_sec[0](out)  # Second convolution path
        out = out + self.residual_input_conv[0](resnet_in)  # Residual connection

        # Subsequent layers
        for i in range(self.num_layers):
            # Attention processing
            batch_size, chanels, h, w = out.shape
            in_attn = out.reshape(batch_size, chanels, h * w)  # Flatten spatial dims
            in_attn = self.attention_norm[i](in_attn)  # Normalize

            # Attention implementation choice
            if self.custom_mha:
                in_attn = in_attn.transpose(-1, -2)  # (B, C, H*W) -> (B, H*W, C)
                out_attn = self.attention[i](in_attn)  # Custom MHA
                out_attn = out_attn.transpose(-1, -2)  # (B, C, H*W)
            else:
                in_attn = in_attn.transpose(1, 2)  # (B, C, H*W) -> (B, H*W, C)
                out_attn, _ = self.attention[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2)  # (B, C, H*W)

            # Reshape and add attention output
            out_attn = out_attn.reshape(batch_size, chanels, h, w)
            out = out + out_attn  # Residual attention

            # Cross Attention processing
            if self.cross_attn:
                assert context is not None, "Cross attention requires a context tensor."
                batch_size, chanels, h, w = out.shape
                in_attn = out.reshape(
                    batch_size, chanels, h * w
                )  # Flatten spatial dims
                in_attn = self.cross_attention_norm[i](in_attn)
                if self.custom_mha:
                    # Custom attention expects (batch, seq_len, channels)
                    in_attn = in_attn.transpose(-1, -2)
                    out_attn = self.cross_attention[i](in_attn, context)
                    out_attn = out_attn.transpose(-1, -2)
                else:
                    # PyTorch MHA expects (batch, seq_len, channels)
                    in_attn = in_attn.transpose(1, 2)
                    contex_proj = self.context_proj[i](context)
                    out_attn, _ = self.cross_attention[i](
                        in_attn, contex_proj, contex_proj
                    )
                    out_attn = out_attn.transpose(1, 2)

                # Reshape back to original dimensions
                out_attn = out_attn.reshape(batch_size, chanels, h, w)
                out = out + out_attn  # Add attention output

            # Subsequent residual block
            resnet_in = out  # Save input
            out = self.resnet_conv_first[i + 1](out)  # Conv path
            if self.time_emb_dim:
                out = (
                    out + self.time_emb_layers[i + 1](time_emb)[:, :, None, None]
                )  # Time emb
            out = self.resnet_conv_sec[i + 1](out)  # Second conv path
            out = out + self.residual_input_conv[i + 1](resnet_in)  # Residual

        return out


class UpSamplingBlock(nn.Module):
    """
    UNet upsampling block that combines:
    - Feature upsampling with skip connections
    - Residual blocks with time embeddings
    - Optional attention mechanisms
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int = 0,
        time_emb_dim: int = None,
        num_heads: int = 4,
        up_sample: bool = True,
        custom_mha: bool = True,
        num_layers: int = 1,
        use_attn: bool = True,
        grp_norm_chanels: int = 8,
        cross_attn=False,
        cross_cont_dim=None,
        custom_cross_attn=True,
    ):
        super().__init__()

        # Configuration parameters
        self.up_sample = up_sample  # Whether to upsample features
        self.custom_mha = custom_mha  # Attention implementation choice
        self.num_heads = num_heads  # Number of attention heads
        self.num_layers = num_layers  # Number of residual-attention layers
        self.time_emb_dim = time_emb_dim
        self.use_attn = use_attn
        self.pre_channels = in_channels  # channels before concatenation
        self.skip_channels = skip_channels  # channels coming from skip connection
        self.first_in_channels = in_channels + skip_channels  # channels after concat
        self.cross_attn = cross_attn
        self.cross_cont_dim = cross_cont_dim
        self.custom_cross_attn = custom_cross_attn

        _assert_group_norm_divisible(
            grp_norm_chanels, self.first_in_channels, "first_in_channels"
        )
        _assert_group_norm_divisible(grp_norm_chanels, out_channels, "out_channels")

        # Residual block components (per layer)
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(
                        grp_norm_chanels,
                        self.first_in_channels if i == 0 else out_channels,
                    ),  # Input normalization
                    nn.SiLU(),  # Activation
                    nn.Conv2d(  # Channel transformation
                        self.first_in_channels if i == 0 else out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
                for i in range(num_layers)
            ]
        )

        # Time embedding processing (per layer)
        if time_emb_dim:
            self.time_emb_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.SiLU(),  # Time embedding activation
                        nn.Linear(
                            time_emb_dim, out_channels
                        ),  # Project to channel space
                    )
                    for _ in range(num_layers)
                ]
            )

        # Second residual path (per layer)
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(grp_norm_chanels, out_channels),  # Normalization
                    nn.SiLU(),  # Activation
                    nn.Conv2d(  # Final convolution
                        out_channels, out_channels, kernel_size=3, stride=1, padding=1
                    ),
                )
                for _ in range(num_layers)
            ]
        )

        if use_attn:
            # Attention components (per layer)
            self.attention_norm = nn.ModuleList(
                [
                    nn.GroupNorm(
                        grp_norm_chanels, out_channels
                    )  # Pre-attention normalization
                    for _ in range(num_layers)
                ]
            )

            # Attention implementation choice
            if custom_mha:
                self.attention = nn.ModuleList(
                    [
                        MultiHeadSelfAttention(  # Custom implementation
                            num_heads, out_channels, input_proj_bias=False
                        )
                        for _ in range(num_layers)
                    ]
                )
            else:
                self.attention = nn.ModuleList(
                    [
                        nn.MultiheadAttention(  # PyTorch native
                            out_channels, num_heads, batch_first=True
                        )
                        for _ in range(num_layers)
                    ]
                )

        #### Cross attention ###
        if self.cross_attn:
            assert (
                self.cross_cont_dim is not None
            ), "Context dim must be passed for cross attention."
            self.cross_attention_norm = nn.ModuleList(
                [
                    nn.GroupNorm(
                        grp_norm_chanels, out_channels
                    )  # Normalization before attention
                    for _ in range(num_layers)
                ]
            )

            # Choose attention implementation
            if custom_cross_attn:
                self.cross_attention = nn.ModuleList(
                    [
                        MultiHeadCrossAttention(
                            num_heads,
                            out_channels,
                            self.cross_cont_dim,
                            input_proj_bias=False,
                        )
                        for _ in range(num_layers)
                    ]
                )
            else:
                self.context_proj = nn.ModuleList(
                    [
                        nn.Linear(self.cross_cont_dim, out_channels)
                        for _ in range(num_layers)
                    ]
                )
                self.cross_attention = nn.ModuleList(
                    [
                        nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                        for _ in range(num_layers)
                    ]
                )

        # Residual projections (per layer)
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(  # 1x1 conv for channel matching
                    self.first_in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=1,
                )
                for i in range(num_layers)
            ]
        )

        # Upsampling layer
        self.up_sample_conv = (
            nn.ConvTranspose2d(  # Transposed convolution for upsampling
                self.pre_channels,
                self.pre_channels,  # maintain pre-concat channels
                kernel_size=4,
                stride=2,
                padding=1,
            )
            if self.up_sample
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor = None,
        out_down: torch.Tensor = None,
        context=None,
    ) -> torch.Tensor:
        if self.time_emb_dim:
            assert time_emb is not None, "time_emb must be provided when time_emb_dim is set."
        # Upsample and combine with skip connection
        x = self.up_sample_conv(
            x
        )  # Increase spatial resolution (channels = pre_channels)
        if out_down is not None:
            x = torch.cat([x, out_down], dim=1)  # Concatenate with skip features

        out = x  # Initialize processing
        for i in range(self.num_layers):
            # Residual block processing
            resnet_input = out  # Save for residual connection
            out = self.resnet_conv_first[i](out)  # First conv path
            if self.time_emb_dim:
                out = (
                    out + self.time_emb_layers[i](time_emb)[:, :, None, None]
                )  # Add time embedding
            out = self.resnet_conv_second[i](out)  # Second conv path
            out = out + self.residual_input_conv[i](resnet_input)  # Residual connection

            # Attention processing
            if self.use_attn:
                batch_size, chanels, h, w = out.shape
                in_attn = out.reshape(
                    batch_size, chanels, h * w
                )  # Flatten spatial dims
                in_attn = self.attention_norm[i](in_attn)  # Normalize

                if self.custom_mha:
                    in_attn = in_attn.transpose(-1, -2)  # (B, C, N) -> (B, N, C)
                    out_attn = self.attention[i](in_attn)
                    out_attn = out_attn.transpose(-1, -2)  # (B, C, N)
                else:
                    in_attn = in_attn.transpose(1, 2)  # (B, C, N) -> (B, N, C)
                    out_attn, _ = self.attention[i](in_attn, in_attn, in_attn)
                    out_attn = out_attn.transpose(1, 2)  # (B, C, N)

                # Reshape and add attention output
                out_attn = out_attn.reshape(batch_size, chanels, h, w)
                out = out + out_attn  # Residual attention

            # Cross Attention processing
            if self.cross_attn:
                assert context is not None, "Cross attention requires a context tensor."
                batch_size, chanels, h, w = out.shape
                in_attn = out.reshape(
                    batch_size, chanels, h * w
                )  # Flatten spatial dims
                in_attn = self.cross_attention_norm[i](in_attn)
                if self.custom_mha:
                    # Custom attention expects (batch, seq_len, channels)
                    in_attn = in_attn.transpose(-1, -2)
                    out_attn = self.cross_attention[i](in_attn, context)
                    out_attn = out_attn.transpose(-1, -2)
                else:
                    # PyTorch MHA expects (batch, seq_len, channels)
                    in_attn = in_attn.transpose(1, 2)
                    contex_proj = self.context_proj[i](context)
                    out_attn, _ = self.cross_attention[i](
                        in_attn, contex_proj, contex_proj
                    )
                    out_attn = out_attn.transpose(1, 2)

                # Reshape back to original dimensions
                out_attn = out_attn.reshape(batch_size, chanels, h, w)
                out = out + out_attn  # Add attention output

        return out
