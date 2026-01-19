import torch
import torch.nn as nn
import torch.nn.functional as F

from ControlNet.models.UNetBlocks import DownSamplingBlock, BottleNeck, UpSamplingBlock


class VectorQuantizedVAE(nn.Module):
    """
    Vector Quantized Variational Autoencoder (VQ-VAE) implementation with configurable architecture.

    Key Components:
    - Encoder: Downsampling blocks followed by bottleneck layers
    - Quantizer: Discrete codebook mapping with straight-through estimation
    - Decoder: Upsampling blocks preceded by bottleneck layers

    Args:
        input_channels (int): Number of channels in input images
        model_config (dict): Configuration dictionary containing:
            - down_channels (list): Encoder channel progression
            - bottleneck_channels (list): Bottleneck layer channels
            - downsampling_steps (list): Booleans for downsampling at each level
            - num_down_layers (int): Number of layers per downsampling block
            - num_bottleneck_layers (int): Number of layers per bottleneck block
            - num_up_layers (int): Number of layers per upsampling block
            - use_attention_downsample (list): Booleans for attention in downsampling blocks
            - latent_channels (int): Number of channels in latent space
            - codebook_size (int): Number of entries in codebook
            - group_norm_channels (int): Number of groups for GroupNorm
            - num_attention_heads (int): Number of heads for attention modules
    """

    def __init__(self, input_channels: int, *args, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        assert hasattr(
            self, "VQVAE"
        ), "VectorQuantizedVAE expects a 'VQVAE' configuration dictionary."

        # Architecture configuration
        self.down_channels = self.VQVAE["down_channels"]
        self.bottleneck_channels = self.VQVAE["mid_channels"]
        self.downsampling_steps = self.VQVAE["down_sample"]
        self.num_down_layers = self.VQVAE["num_down_layers"]
        self.num_bottleneck_layers = self.VQVAE["num_mid_layers"]
        self.num_up_layers = self.VQVAE["num_up_layers"]
        self.use_attention_downsample = self.VQVAE["attn_down"]

        # Latent space configuration
        self.latent_channels = self.VQVAE["z_channels"]
        self.codebook_size = self.VQVAE["codebook_size"]
        self.group_norm_channels = self.VQVAE["norm_channels"]
        self.num_attention_heads = self.VQVAE["num_heads"]
        self.use_ema = self.VQVAE.get("use_ema", False)
        self.ema_decay = _as_float(self.VQVAE.get("ema_decay", 0.99), "ema_decay")
        self.ema_eps = _as_float(self.VQVAE.get("ema_eps", 1e-5), "ema_eps")
        self.activation = nn.SiLU()

        # Architecture validation
        self._validate_architecture()

        # Create decoder upsampling steps (reverse of encoder downsampling)
        self.upsampling_steps = list(reversed(self.downsampling_steps))

        ######################
        # Encoder Components #
        ######################
        self.encoder = nn.ModuleDict(
            {
                "input_conv": nn.Conv2d(
                    input_channels, self.down_channels[0], kernel_size=3, padding=1
                ),
                "down_blocks": nn.ModuleList(
                    [
                        DownSamplingBlock(
                            in_channels=self.down_channels[i],
                            out_channels=self.down_channels[i + 1],
                            time_emb_dim=0,
                            down_sample=self.downsampling_steps[i],
                            num_heads=self.num_attention_heads,
                            num_layers=self.num_down_layers,
                            use_attn=self.use_attention_downsample[i],
                            grp_norm_chanels=self.group_norm_channels,
                        )
                        for i in range(len(self.down_channels) - 1)
                    ]
                ),
                "bottleneck_blocks": nn.ModuleList(
                    [
                        BottleNeck(
                            in_channels=self.bottleneck_channels[i],
                            out_channels=self.bottleneck_channels[i + 1],
                            time_emb_dim=None,
                            num_heads=self.num_attention_heads,
                            num_layers=self.num_bottleneck_layers,
                            grp_norm_chanels=self.group_norm_channels,
                        )
                        for i in range(len(self.bottleneck_channels) - 1)
                    ]
                ),
                "output_norm": nn.GroupNorm(
                    self.group_norm_channels, self.down_channels[-1]
                ),
                "output_conv": nn.Conv2d(
                    self.down_channels[-1],
                    self.latent_channels,
                    kernel_size=3,
                    padding=1,
                ),
                "pre_quant_conv": nn.Conv2d(
                    self.latent_channels, self.latent_channels, kernel_size=1
                ),
            }
        )

        ######################
        # Codebook Component #
        ######################
        self.codebook = nn.Embedding(self.codebook_size, self.latent_channels)
        if self.use_ema:
            self.codebook.weight.requires_grad_(False)
            self.register_buffer(
                "ema_cluster_size", torch.zeros(self.codebook_size)
            )
            self.register_buffer(
                "ema_weight", torch.zeros(self.codebook_size, self.latent_channels)
            )
            self.ema_weight.copy_(self.codebook.weight.data)
        self.register_buffer("_codebook_weight_sq", torch.empty(0), persistent=False)
        self._codebook_weight_version = None

        ######################
        # Decoder Components #
        ######################
        self.decoder = nn.ModuleDict(
            {
                "post_quant_conv": nn.Conv2d(
                    self.latent_channels, self.latent_channels, kernel_size=1
                ),
                "input_conv": nn.Conv2d(
                    self.latent_channels,
                    self.bottleneck_channels[-1],
                    kernel_size=3,
                    padding=1,
                ),
                "bottleneck_blocks": nn.ModuleList(
                    [
                        BottleNeck(
                            in_channels=self.bottleneck_channels[i],
                            out_channels=self.bottleneck_channels[i - 1],
                            time_emb_dim=None,
                            num_heads=self.num_attention_heads,
                            num_layers=self.num_bottleneck_layers,
                            grp_norm_chanels=self.group_norm_channels,
                        )
                        for i in reversed(range(1, len(self.bottleneck_channels)))
                    ]
                ),
                "up_blocks": nn.ModuleList(
                    [
                        UpSamplingBlock(
                            in_channels=self.down_channels[i],
                            out_channels=self.down_channels[i - 1],
                            time_emb_dim=None,
                            up_sample=self.downsampling_steps[i - 1],
                            num_heads=self.num_attention_heads,
                            num_layers=self.num_up_layers,
                            use_attn=self.use_attention_downsample[i - 1],
                            grp_norm_chanels=self.group_norm_channels,
                        )
                        for i in reversed(range(1, len(self.down_channels)))
                    ]
                ),
                "output_norm": nn.GroupNorm(
                    self.group_norm_channels, self.down_channels[0]
                ),
                "output_conv": nn.Conv2d(
                    self.down_channels[0], input_channels, kernel_size=3, padding=1
                ),
            }
        )

    def _validate_architecture(self):
        """Validate architectural consistency of configuration"""
        assert (
            self.bottleneck_channels[0] == self.down_channels[-1]
        ), "First bottleneck channel must match last down channel"

        assert (
            self.bottleneck_channels[-1] == self.down_channels[-1]
        ), "Bottleneck output must match final down channel"

        assert len(self.downsampling_steps) == len(
            self.down_channels
        ) - 1, "Downsampling steps must match down channels - 1"

        assert len(self.use_attention_downsample) == len(
            self.down_channels
        ) - 1, "Attention flags must match down channels - 1"

    def _quantize_latents(self, latents: torch.Tensor) -> tuple:
        """
        Quantize latent vectors using codebook with straight-through estimation

        Args:
            latents (torch.Tensor): Continuous latent vectors [B, C, H, W]

        Returns:
            tuple: (quantized latents, quantization losses, code indices)
        """
        batch_size, channels, height, width = latents.shape

        # Reshape latents for codebook comparison
        flat_latents = latents.permute(0, 2, 3, 1).reshape(
            batch_size * height * width, channels
        )  # [B*H*W, C]

        # Compute squared distances: ||x||^2 + ||e||^2 - 2 x·e
        codebook_weight = self.codebook.weight  # [K, C]
        codebook_weight_sq = self._get_codebook_weight_sq(codebook_weight)  # [K]
        latents_sq = (flat_latents ** 2).sum(dim=1, keepdim=True)  # [B*H*W, 1]
        distances = (
            latents_sq
            + codebook_weight_sq.unsqueeze(0)
            - 2.0 * flat_latents @ codebook_weight.t()
        )  # [B*H*W, K]

        # Find nearest codebook indices
        encoding_indices = torch.argmin(distances, dim=-1)  # [B*H*W]
        quantized = self.codebook(encoding_indices)  # [B*H*W, C]

        if self.use_ema and self.training:
            self._ema_update(flat_latents, encoding_indices)

        # Calculate quantization losses
        commitment_loss = torch.mean((quantized.detach() - flat_latents) ** 2)
        codebook_loss = torch.mean((quantized - flat_latents.detach()) ** 2)

        # Straight-through estimation gradient trick
        quantized = flat_latents + (quantized - flat_latents).detach()

        # Reshape back to original dimensions
        quantized = quantized.view(batch_size, height, width, channels)
        quantized = quantized.permute(0, 3, 1, 2)  # [B, C, H, W]

        # Reshape indices for visualization
        encoding_indices = encoding_indices.view(batch_size, height, width)

        return (
            quantized,
            {"codebook": codebook_loss, "commitment": commitment_loss},
            encoding_indices,
        )

    def _get_codebook_weight_sq(self, codebook_weight: torch.Tensor) -> torch.Tensor:
        version = getattr(codebook_weight, "_version", None)
        if version is None:
            return (codebook_weight ** 2).sum(dim=1)
        if (
            self._codebook_weight_sq.numel() == 0
            or self._codebook_weight_sq.device != codebook_weight.device
            or self._codebook_weight_sq.dtype != codebook_weight.dtype
            or self._codebook_weight_version != version
        ):
            self._codebook_weight_sq = (codebook_weight ** 2).sum(dim=1)
            self._codebook_weight_version = version
        return self._codebook_weight_sq

    def _ema_update(
        self, flat_latents: torch.Tensor, encoding_indices: torch.Tensor
    ) -> None:
        with torch.no_grad():
            encodings = F.one_hot(
                encoding_indices, self.codebook_size
            ).type(flat_latents.dtype)
            cluster_size = encodings.sum(dim=0)  # [K]
            dw = encodings.t() @ flat_latents  # [K, C]

            self.ema_cluster_size.mul_(self.ema_decay).add_(
                cluster_size * (1.0 - self.ema_decay)
            )
            self.ema_weight.mul_(self.ema_decay).add_(dw * (1.0 - self.ema_decay))

            n = self.ema_cluster_size.sum()
            cluster_size = (self.ema_cluster_size + self.ema_eps) / (
                n + self.codebook_size * self.ema_eps
            ) * n
            embed_normalized = self.ema_weight / cluster_size.unsqueeze(1)
            self.codebook.weight.copy_(embed_normalized)
            self._codebook_weight_version = None
            
    def encode(self, x: torch.Tensor) -> tuple:
        """Encode input images to quantized latents"""
        # Encoder forward pass
        x = self.encoder["input_conv"](x)
        for down_block in self.encoder["down_blocks"]:
            # Blocks support optional time embeddings; not used here
            x = down_block(x)
        for bottleneck in self.encoder["bottleneck_blocks"]:
            x = bottleneck(x)

        # Final processing before quantization
        x = self.encoder["output_norm"](x)
        x = self.activation(x)
        x = self.encoder["output_conv"](x)
        x = self.encoder["pre_quant_conv"](x)

        # Quantize latent representation
        quantized, losses, _ = self._quantize_latents(x)
        return quantized, losses

    def encode_with_indices(self, x: torch.Tensor) -> tuple:
        """
        Encode input images to quantized latents and also return codebook indices.

        Returns:
            tuple: (quantized latents, quantization losses, encoding indices [B,H,W])
        """
        z = self.encoder["input_conv"](x)
        for down_block in self.encoder["down_blocks"]:
            z = down_block(z)
        for bottleneck in self.encoder["bottleneck_blocks"]:
            z = bottleneck(z)
        z = self.encoder["output_norm"](z)
        z = self.activation(z)
        z = self.encoder["output_conv"](z)
        z = self.encoder["pre_quant_conv"](z)
        quantized, losses, indices = self._quantize_latents(z)
        return quantized, losses, indices

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode quantized latents back to images"""
        # Decoder forward pass
        z = self.decoder["post_quant_conv"](z)
        z = self.decoder["input_conv"](z)

        for bottleneck in self.decoder["bottleneck_blocks"]:
            z = bottleneck(z)

        for up_block in self.decoder["up_blocks"]:
            z = up_block(z)

        # Final output processing
        z = self.decoder["output_norm"](z)
        z = self.activation(z)
        return self.decoder["output_conv"](z)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Full VQ-VAE forward pass

        Args:
            x (torch.Tensor): Input images [B, C, H, W]

        Returns:
            tuple: (reconstructed images, quantized latents, quantization losses)
        """
        quantized, losses = self.encode(x)
        reconstructed = self.decode(quantized)
        return reconstructed, quantized, losses


def _as_float(value, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a float, got {value!r}.") from exc
