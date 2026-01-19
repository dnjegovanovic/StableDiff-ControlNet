import torch
import torch.nn as nn


class Discriminator(nn.Module):
    r"""
    PatchGAN Discriminator that outputs a grid of predictions indicating
    real/fake probabilities for image patches.

    Architecture:
    - Series of convolutional layers with decreasing spatial dimensions
    - Batch normalization applied to intermediate layers
    - LeakyReLU activations except for the final output layer
    - Outputs a feature map rather than single probability

    Args:
        im_channels (int): Number of input image channels (default: 3)
        conv_channels (list): Output channels for intermediate conv layers (default: [64, 128, 256])
        kernels (list): Kernel sizes for each conv layer (default: [4, 4, 4, 4])
        strides (list): Strides for each conv layer (default: [2, 2, 2, 1])
        paddings (list): Padding for each conv layer (default: [1, 1, 1, 1])

    Input:
        Tensor of shape (batch_size, im_channels, height, width)

    Output:
        Tensor of shape (batch_size, 1, height//2^(n-1), width//2^(n-1)),
        where n is the number of strided conv layers
    """

    def __init__(
        self,
        im_channels: int = 3,
        conv_channels: list = [64, 128, 256],
        kernels: list = [4, 4, 4, 4],
        strides: list = [2, 2, 2, 1],
        paddings: list = [1, 1, 1, 1],
    ):
        super().__init__()

        # Store initialization parameters
        self.im_channels = im_channels
        self.conv_channels = conv_channels
        self.kernels = kernels
        self.strides = strides
        self.paddings = paddings

        # Validate layer configuration
        num_layers = len(conv_channels) + 1  # +1 for final output layer
        assert (
            len(kernels) == num_layers
        ), "Kernels list length must match number of conv layers"
        assert (
            len(strides) == num_layers
        ), "Strides list length must match number of conv layers"
        assert (
            len(paddings) == num_layers
        ), "Paddings list length must match number of conv layers"

        # Activation function for intermediate layers
        intermediate_activation = nn.LeakyReLU(0.2, inplace=True)

        # Construct full channel sequence: input -> conv_channels -> output
        channel_sequence = [self.im_channels] + self.conv_channels + [1]

        # Build convolutional blocks
        self.conv_blocks = nn.ModuleList()
        for layer_idx in range(len(channel_sequence) - 1):
            # Current layer parameters
            in_channels = channel_sequence[layer_idx]
            out_channels = channel_sequence[layer_idx + 1]
            kernel_size = self.kernels[layer_idx]
            stride = self.strides[layer_idx]
            padding = self.paddings[layer_idx]

            # Convolutional layer configuration
            conv_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                # Use bias only in first layer and final output layer
                bias=(layer_idx == 0 or layer_idx == len(channel_sequence) - 2),
            )

            # Batch normalization configuration
            # Skip batch norm in first and last layers
            use_batch_norm = layer_idx not in (0, len(channel_sequence) - 2)
            batch_norm = (
                nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
            )

            # Activation configuration
            # Skip activation in final output layer
            activation = (
                intermediate_activation
                if layer_idx != len(channel_sequence) - 2
                else nn.Identity()
            )

            # Assemble the sequential block
            block = nn.Sequential(conv_layer, batch_norm, activation)
            self.conv_blocks.append(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the discriminator network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, im_channels, height, width)

        Returns:
            torch.Tensor: Output feature map of shape (batch_size, 1, H', W')
        """
        # Sequentially apply all convolutional blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        return x
