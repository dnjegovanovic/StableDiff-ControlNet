import torch
from torch import Tensor

from typing import Dict


class LinearNoiseScheduler:
    """Implements the linear noise scheduling mechanism from the DDPM paper.

    This class handles the forward process of gradually adding noise to images
    and the reverse process parameters for sampling. The noise schedule follows
    a linear beta trajectory as proposed in the original paper.

    Attributes:
        num_timesteps: Number of diffusion timesteps.
        beta_start: Starting value of beta for noise scheduling.
        beta_end: Ending value of beta for noise scheduling.
        betas: Linearly spaced beta values from beta_start to beta_end.
        alphas: 1 - betas values used for noise scaling.
        alphas_cumprod: Cumulative product of alphas for each timestep.
        sqrt_alphas_cumprod: Square root of alphas_cumprod for efficient computation.
        sqrt_one_minus_alphas_cumprod: Square root of (1 - alphas_cumprod) for noise mixing.
    """

    ALLOWED_PARAMS = {"num_timesteps", "beta_start", "beta_end", "device"}

    # Built-in defaults if a key is not provided
    DEFAULTS: Dict[str, float] = {
        "num_timesteps": 1000,
        "beta_start": 1e-4,
        "beta_end": 2e-2,  # 0.02
    }

    def __init__(self, device, *args, **kwargs):
        """Initializes the noise scheduler with linear beta schedule.

        Args:
            num_timesteps: Number of timesteps in the diffusion process.
            beta_start: Starting value for beta schedule (small positive value).
            beta_end: Ending value for beta schedule.
        """
        self.device = device

        # 1) Start from defaults
        cfg: Dict = dict(self.DEFAULTS)

        # Try to read kwargs["DDPMParams"].items(); fall back to defaults if missing/invalid
        try:
            ddpm = kwargs["DDPMParams"]
            # accept dict-like objects (must have .items())
            items_iter = ddpm.items() if hasattr(ddpm, "items") else ()
        except KeyError:
            items_iter = ()  # key absent → just keep defaults

        # Overlay provided DDPMParams (only known keys)
        for k, v in items_iter:
            if k not in self.ALLOWED_PARAMS:
                raise AttributeError(f"Invalid parameter in DDPMParams: {k}")
            cfg[k] = v

        # (Optional) allow direct kwargs overrides too; comment this block out if not desired
        for k in self.ALLOWED_PARAMS:
            if k in kwargs:
                cfg[k] = kwargs[k]

        # Normalize types + validate
        self.num_timesteps = int(cfg["num_timesteps"])
        self.beta_start = float(cfg["beta_start"])
        self.beta_end = float(cfg["beta_end"])
        if self.num_timesteps <= 0:
            raise ValueError("num_timesteps must be > 0")
        if not (0.0 < self.beta_start < self.beta_end < 1.0):
            raise ValueError("Require 0 < beta_start < beta_end < 1")

        # Generate linearly spaced beta values (Eq. 2 in paper)
        self.betas = torch.linspace(
            self.beta_start, self.beta_end, self.num_timesteps
        ).to(self.device)

        # Calculate alpha parameters (Eq. 4 in paper)
        self.alphas = 1.0 - self.betas

        # Compute cumulative product of alphas (ᾱ_t in paper)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)

        # Precompute terms for efficient noise addition during forward process
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(
            self.device
        )

    def add_noise(self, x_start: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        """Applies forward process noise to input images at specific timesteps.

        Implements equation 4 from the DDPM paper:
        q(x_t | x_0) = N(x_t; sqrt(ᾱ_t) x_0, (1 - ᾱ_t)I)

        Args:
            x_start: Original clean images (batch, channels, height, width).
            noise: Gaussian noise tensor to add.
            t: Timestep indices for each image in the batch (batch_size,).

        Returns:
            Noised images at timestep t.
        """
        batch_size = x_start.shape[0]
        # Gather precomputed coefficients for given timesteps t
        sqrt_alpha = self.sqrt_alphas_cumprod[t].reshape(batch_size, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].reshape(
            batch_size, 1, 1, 1
        )

        # Combine images with noise using precomputed scaling factors
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def sample_prev_timestep(
        self, x_t: Tensor, noise_pred: Tensor, t: int
    ) -> tuple[Tensor, Tensor]:
        """Samples from the reverse process distribution to denoise images.

        Implements the reverse diffusion step from equation 11 in the paper.

        Args:
            x_t: Noisy images at current timestep t.
            noise_pred: Model's prediction of the noise component.
            t: Current timestep index (scalar).

        Returns:
            tuple: (denoised sample at t-1, predicted denoised image x0)
        """
        # Calculate predicted x0 from noise prediction (Eq. 11 first part)
        x0_coeff = 1.0 / self.sqrt_alphas_cumprod[t]
        x0 = x0_coeff * (x_t - noise_pred * self.sqrt_one_minus_alphas_cumprod[t])
        x0 = torch.clamp(x0, -1.0, 1.0)

        # Calculate mean of reverse process distribution (Eq. 11 second part)
        mean_coeff = 1.0 / torch.sqrt(self.alphas[t])
        mean = mean_coeff * (
            x_t - (self.betas[t] * noise_pred) / self.sqrt_one_minus_alphas_cumprod[t]
        )

        if t == 0:
            return mean, x0

        # Calculate variance for reverse process (Eq. 15)
        variance = (1.0 - self.alphas_cumprod[t - 1]) / (1.0 - self.alphas_cumprod[t])
        variance *= self.betas[t]
        sigma = variance**0.5

        # Add noise for stochastic sampling when t > 0
        z = torch.randn_like(x_t)
        return mean + sigma * z, x0


def to(self, device):
    self.device = device
    self.betas = self.betas.to(device)
    self.alphas = self.alphas.to(device)
    self.alphas_cumprod = self.alphas_cumprod.to(device)
    self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
    self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
    return self
