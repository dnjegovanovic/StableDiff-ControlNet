import torch  # Provides core tensor functionality for PyTorch models
import torch.nn as nn  # Exposes neural network layers and modules from PyTorch
import torch.nn.functional as F  # Supplies interpolation utilities for resizing condition inputs
from ControlNet.models.UNetBlocks import (
    DownSamplingBlock,
    BottleNeck,
    UpSamplingBlock,
)  # Imports UNet building blocks defined in this project
from ControlNet.models.TimeEmbedding import (
    TimeEmbedding,
    time_embedding_fun,
)  # Brings in sinusoidal time embedding utilities for diffusion timesteps
from ControlNet.utils.config import (
    get_config_value,
    validate_class_config,
    validate_class_conditional_input,
    validate_image_config,
    validate_text_config,
)  # Loads helper functions for validation and configuration access
from einops import (
    einsum,
)  # Allows concise tensor contractions used for class conditioning


class UNet(nn.Module):  # Defines the core UNet architecture used by the diffusion model
    """
    UNet-like architecture with attention and time embedding for diffusion models.

    The architecture consists of:
    1. A series of downsampling blocks
    2. A bottleneck block
    3. A series of upsampling blocks with skip connections

    Args:
        in_chanels: Number of input channels
    """

    def __init__(
        self, *args, **kwargs
    ):  # Accepts flexible arguments so configs can be injected directly
        super().__init__()  # Initializes parent nn.Module state
        self.__dict__.update(
            kwargs
        )  # Copies keyword arguments (like UnetParams) into instance attributes
        assert hasattr(
            self, "UnetParams"
        ), "UNet expects an 'UnetParams' configuration dictionary."  # Ensures required configuration is present

        # Channel configuration -------------------------------------------------------------
        unet_config = (
            self.UnetParams
        )  # Caches the UNet hyperparameter dictionary for easier access
        self.down_channels = unet_config[
            "down_channels"
        ]  # Channel widths for each encoder stage
        self.mid_channels = unet_config[
            "mid_channels"
        ]  # Channel widths for bottleneck stages
        self.down_sample = unet_config[
            "down_sample"
        ]  # Flags indicating whether to downsample in each encoder stage
        self.attns = unet_config[
            "attn_down"
        ]  # Flags controlling attention usage in encoder stages
        self.time_emb_dim = unet_config[
            "time_emb_dim"
        ]  # Dimensionality of the timestep embedding vector
        self.in_channels = unet_config[
            "im_channels"
        ]  # Number of channels in the input image representation
        self.num_down_layers = unet_config[
            "num_down_layers"
        ]  # Number of residual layers per encoder block
        self.num_mid_layers = unet_config[
            "num_mid_layers"
        ]  # Number of residual layers in the bottleneck block
        self.num_up_layers = unet_config[
            "num_up_layers"
        ]  # Number of residual layers per decoder block
        self.num_heads = unet_config.get(
            "num_heads", 4
        )  # Attention heads used across UNet blocks

        assert (
            self.mid_channels[0] == self.down_channels[-1]
        ), "Bottleneck in_channels must match deepest encoder output."  # Verifies encoder-to-bottleneck interface
        assert (
            self.mid_channels[-1] == self.down_channels[-2]
        ), "Bottleneck out_channels must match penultimate encoder output."  # Verifies bottleneck-to-decoder interface
        assert (
            len(self.down_sample) == len(self.down_channels) - 1
        ), "down_sample length must be one less than down_channels length."  # Ensures downsample flags align with transitions
        assert (
            len(self.attns) == len(self.down_channels) - 1
        ), "attn_down length must be one less than down_channels length."  # Ensures attention flags align with transitions
        assert (
            self.time_emb_dim % 2 == 0
        ), "time_emb_dim must be divisible by 2 for sinusoidal embeddings."

        # Conditioning configuration --------------------------------------------------------
        self.class_cond = False  # Tracks whether class conditioning is enabled
        self.text_cond = False  # Tracks whether text conditioning is enabled
        self.image_cond = False  # Tracks whether image conditioning is enabled
        self.text_embed_dim = (
            None  # Placeholder for text embedding dimensionality when configured
        )

        model_config = get_config_value(
            unet_config, "model_config", unet_config
        )  # Supports nested model_config wrappers while defaulting to the same dictionary
        self.condition_config = get_config_value(
            model_config, "condition_config", None
        )  # Retrieves optional conditioning configuration section

        if (
            self.condition_config is not None
        ):  # Only parse conditioning blocks if provided
            assert (
                "condition_types" in self.condition_config
            ), "Condition configuration requires 'condition_types' field."  # Validates presence of condition types
            condition_types = self.condition_config[
                "condition_types"
            ]  # Reads which conditioning modalities are requested

            if "class" in condition_types:  # Handles class conditioning setup
                validate_class_config(
                    self.condition_config
                )  # Validates class conditioning configuration
                self.class_cond = True  # Marks that class conditioning is active
                class_cfg = self.condition_config[
                    "class_condition_config"
                ]  # Accesses class conditioning hyperparameters
                self.num_classes = class_cfg[
                    "num_classes"
                ]  # Stores number of classes for embedding construction
                self.class_drop_prob = class_cfg.get(
                    "cond_drop_prob", 0.0
                )  # Optional classifier-free guidance drop probability

            if "text" in condition_types:  # Handles text conditioning setup
                validate_text_config(
                    self.condition_config
                )  # Validates text conditioning configuration
                self.text_cond = True  # Marks that text conditioning is active
                self.text_embed_dim = self.condition_config["text_condition_config"][
                    "text_embed_dim"
                ]  # Stores expected text embedding dimensionality

            if "image" in condition_types:  # Handles image or mask conditioning setup
                validate_image_config(
                    self.condition_config
                )  # Validates image conditioning configuration
                image_cfg = self.condition_config[
                    "image_condition_config"
                ]  # Accesses image conditioning hyperparameters
                self.image_cond = True  # Marks that image conditioning is active
                self.im_cond_input_ch = image_cfg[
                    "image_condition_input_channels"
                ]  # Reads conditioning image input channel count
                self.im_cond_output_ch = image_cfg[
                    "image_condition_output_channels"
                ]  # Reads number of channels produced after preprocessing

        if (
            self.class_cond
        ):  # Builds class embedding table when class conditioning is enabled
            self.class_emb = nn.Embedding(
                self.num_classes, self.time_emb_dim
            )  # Translates class ids into embeddings aligned with timestep dimension

        if self.image_cond:  # Prepares image conditioning layers when requested
            self.cond_conv_in = nn.Conv2d(
                self.im_cond_input_ch, self.im_cond_output_ch, kernel_size=1, bias=False
            )  # Projects conditioning image into desired channel dimension
            self.conv_in = nn.Conv2d(
                self.in_channels + self.im_cond_output_ch,
                self.down_channels[0],
                kernel_size=3,
                padding=1,
            )  # Combines original and conditioning images before encoding
        else:  # Handles the standard non-image-conditioned case
            self.conv_in = nn.Conv2d(
                self.in_channels, self.down_channels[0], kernel_size=3, padding=1
            )  # Converts raw input channels to first encoder channel count

        # Time embedding projection ---------------------------------------------------------
        self.time_proj = TimeEmbedding(
            self.time_emb_dim, True
        )  # Projects sinusoidal timestep encodings through an MLP

        # Downsampling path ----------------------------------------------------------------
        self.down_sampling = (
            nn.ModuleList()
        )  # Holds encoder blocks for each resolution stage
        for idx in range(
            len(self.down_channels) - 1
        ):  # Iterates over transitions between encoder stages
            self.down_sampling.append(  # Appends a new DownSamplingBlock configured for this stage
                DownSamplingBlock(
                    in_channels=self.down_channels[
                        idx
                    ],  # Number of channels entering the block
                    out_channels=self.down_channels[
                        idx + 1
                    ],  # Number of channels after the block
                    time_emb_dim=self.time_emb_dim,  # Dimension of timestep embedding injected into the block
                    num_heads=self.num_heads,
                    down_sample=self.down_sample[
                        idx
                    ],  # Whether to reduce spatial resolution at the end of the block
                    num_layers=self.num_down_layers,  # Number of residual layers to stack in the block
                    use_attn=self.attns[
                        idx
                    ],  # Whether to include attention in this block
                    cross_attn=self.text_cond,
                    cross_cont_dim=self.text_embed_dim,
                    custom_cross_attn=True
                )
            )

        # Bottleneck path ------------------------------------------------------------------
        self.bottleneck = (
            nn.ModuleList()
        )  # Holds the series of bottleneck residual blocks
        for idx in range(
            len(self.mid_channels) - 1
        ):  # Iterates across bottleneck transitions
            self.bottleneck.append(  # Appends a BottleNeck block responsible for deepest feature processing
                BottleNeck(
                    self.mid_channels[
                        idx
                    ],  # Input channel count to the bottleneck block
                    self.mid_channels[
                        idx + 1
                    ],  # Output channel count from the bottleneck block
                    self.time_emb_dim,  # Dimension of timestep embedding used in the block
                    num_heads=self.num_heads,
                    num_layers=self.num_mid_layers,  # Number of residual layers stacked inside the bottleneck
                    cross_attn=self.text_cond,
                    cross_cont_dim=self.text_embed_dim,
                    custom_cross_attn=True
                )
            )

        # Upsampling path ------------------------------------------------------------------
        self.up_sampling = (
            nn.ModuleList()
        )  # Holds decoder blocks that reconstruct high-resolution features
        current_in_channels = self.mid_channels[
            -1
        ]  # Tracks channel dimensionality entering the decoder
        for idx in reversed(
            range(len(self.down_channels) - 1)
        ):  # Walks encoder stages in reverse to mirror skip connections
            skip_channels = self.down_channels[
                idx
            ]  # Number of channels provided by the corresponding skip connection
            out_channels = (
                self.down_channels[idx - 1] if idx != 0 else self.down_channels[0]
            )  # Defines decoder output channels, matching the first encoder stage
            self.up_sampling.append(  # Appends a configured UpSamplingBlock to the decoder
                UpSamplingBlock(
                    in_channels=current_in_channels,  # Channels entering from the previous decoder stage
                    out_channels=out_channels,  # Channels produced by this decoder stage
                    skip_channels=skip_channels,  # Channels from the skip connection to concatenate
                    time_emb_dim=self.time_emb_dim,  # Timestep embedding dimensionality injected into the block
                    num_heads=self.num_heads,
                    up_sample=self.down_sample[
                        idx
                    ],  # Mirrors encoder downsampling flag to decide on upsampling
                    num_layers=self.num_up_layers,  # Number of residual layers in the decoder block
                    use_attn=True,  # Enables attention in decoder blocks for richer feature fusion
                    cross_attn=self.text_cond,
                    cross_cont_dim=self.text_embed_dim,
                    custom_cross_attn=True
                )
            )
            current_in_channels = out_channels  # Updates the incoming channel count for the next decoder stage

        # Output projection ----------------------------------------------------------------
        assert (
            self.down_channels[0] % 8 == 0
        ), "down_channels[0] must be divisible by 8 for output GroupNorm."
        self.norm_out = nn.GroupNorm(
            8, self.down_channels[0]
        )  # Normalizes features before final projection using group normalization
        self.activation = (
            nn.SiLU()
        )  # Stores the SiLU activation to avoid repeated instantiation
        self.conv_out = nn.Conv2d(
            self.down_channels[0], self.in_channels, kernel_size=3, padding=1
        )  # Maps processed features back to original channel count

    def forward(
        self, x: torch.Tensor, t, cond_input=None
    ):  # Performs a forward pass through the UNet
        """
        Forward pass of the UNet.

        Args:
            x: Input tensor of shape (B, C, H, W)
            t: Timestep for time embedding

        Returns:
            Output tensor of same shape as input
        """
        batch_size = x.shape[0]  # Captures the current batch size from the input tensor

        if (
            self.image_cond
        ):  # Handles optional image conditioning during the forward pass
            assert (
                cond_input is not None and "image" in cond_input
            ), "Image conditioning requested but 'image' tensor missing from cond_input."  # Ensures conditioning information is supplied when needed
            cond_image = cond_input["image"]
            assert (
                cond_image.shape[0] == batch_size
            ), "Image conditioning batch size must match input."
            assert (
                cond_image.shape[1] == self.im_cond_input_ch
            ), "Image conditioning channels must match configured input channels."
            if cond_image.shape[-2:] != x.shape[-2:]:
                cond_image = F.interpolate(
                    cond_image,
                    size=x.shape[-2:],
                    mode="nearest",
                )  # Align conditioning resolution with latent size
            cond_image = self.cond_conv_in(
                cond_image
            )  # Projects conditioning image to the configured channel count
            out = self.conv_in(
                torch.cat([x, cond_image], dim=1)
            )  # Concatenates conditioning features with the input and applies initial convolution
        else:  # Handles standard case without image conditioning
            out = self.conv_in(
                x
            )  # Applies the initial convolution to obtain first-stage features

        if isinstance(
            t, torch.Tensor
        ):  # Supports both tensor and scalar timestep inputs
            if t.dim() == 0:  # Broadcasts scalar tensors to match batch size
                t_vec = t.expand(batch_size).to(
                    device=x.device, dtype=torch.float32
                )  # Broadcasts single timestep across the batch
            else:  # Handles per-sample timestep tensors
                assert (
                    t.shape[0] == batch_size
                ), "Per-sample timestep tensor must align with batch size."  # Validates that timestep tensor matches batch dimension
                t_vec = t.to(
                    device=x.device, dtype=torch.float32
                )  # Moves timesteps to the correct device and dtype
        else:  # Handles integer or float timesteps supplied as standard Python values
            t_vec = torch.full(
                (batch_size,), float(t), device=x.device, dtype=torch.float32
            )  # Creates a tensor filled with the scalar timestep

        time_embedding = time_embedding_fun(
            t_vec, self.time_emb_dim
        )  # Computes sinusoidal timestep embeddings
        time_embedding = self.time_proj(
            time_embedding
        )  # Projects timestep embeddings through the learned projection

        if self.class_cond:  # Applies class conditioning when configured
            validate_class_conditional_input(
                cond_input, x, self.num_classes
            )  # Ensures provided class conditioning tensor is valid
            class_probs = cond_input[
                "class"
            ].float()  # Retrieves class probabilities (or one-hot vectors) as float values
            class_embed = einsum(
                class_probs, self.class_emb.weight, "b n, n d -> b d"
            )  # Pools class embeddings weighted by the provided probabilities
            time_embedding = (
                time_embedding + class_embed
            )  # Adds class-conditioning signal to the timestep embedding
        
        ############# Text cond
        context_hidden_states = None
        if self.text_cond:
            assert (
                cond_input is not None and "text" in cond_input
            ), "Text conditioning requested but 'text' tensor missing from cond_input."
            context_hidden_states = cond_input["text"]
            assert (
                context_hidden_states.shape[0] == batch_size
            ), "Text conditioning batch size must match input."
            assert (
                context_hidden_states.shape[-1] == self.text_embed_dim
            ), "Text conditioning embed dim must match configured text_embed_dim."
        ############################

        down_outs = []  # Stores activations for skip connections during decoding

        for (
            down_block
        ) in (
            self.down_sampling
        ):  # Iterates through encoder blocks to progressively downsample features
            down_outs.append(out)  # Saves current features for later skip connection
            out = down_block(
                out, time_embedding, context_hidden_states
            )  # Processes features through the encoder block with timestep conditioning

        for (
            mid_block
        ) in self.bottleneck:  # Applies bottleneck processing at the lowest resolution
            out = mid_block(
                out, time_embedding, context_hidden_states
            )  # Refines features with additional residual layers and attention

        for (
            up_block
        ) in (
            self.up_sampling
        ):  # Iterates through decoder blocks to reconstruct spatial resolution
            skip = (
                down_outs.pop()
            )  # Retrieves the corresponding skip connection features
            out = up_block(
                out, time_embedding, skip, context_hidden_states
            )  # Merges current features with skip connection and applies decoder processing

        out = self.norm_out(out)  # Normalizes decoder output before activation
        out = self.activation(out)  # Applies non-linearity prior to final projection
        out = self.conv_out(
            out
        )  # Projects features back to the original number of channels
        return out  # Returns the denoised prediction matching the input tensor shape
