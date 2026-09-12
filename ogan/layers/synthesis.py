"""Noise injection, bias, and the layer that assembles a style block.

Provenance (see PROVENANCE.md):

- P2, §2: the noise inputs are "single-channel images consisting of uncorrelated
  Gaussian noise", with "a dedicated noise image to each layer of the synthesis
  network". StyleGAN scaled them per channel.
- P1 App. B, *Generator redesign*: StyleGAN2 simplifies "the noise broadcast
  operations to use a single shared scaling factor for all feature maps", and
  initialises "all biases and noise scaling factors to zero".
- P1 §2.1: StyleGAN applied bias and noise *inside* the style block, "causing
  their relative impact to be inversely proportional to the current style's
  magnitudes". StyleGAN2 moves both outside it, "where they operate on
  normalized data". The same paragraph removes bias, noise and normalisation
  from the constant input entirely.
- P2, config B: upsampling is followed by the lowpass filter, which is what
  `upsample2d` does in one step.

**Two orderings, and only one of them is a real question.** Noise and bias are
both additions, so their relative order changes nothing — it is not a decision.
The activation's position is a decision, and P1 does not state it in prose. It is
`derived`: a bias applied *after* the nonlinearity could only offset the output,
never shift where LeakyReLU bends, which is what a bias in a nonlinear network is
for. So both additions precede the activation.
"""

import torch
from torch import nn

from ogan.layers.equalized import EqualizedLinear, leaky_relu
from ogan.layers.modulated import ModulatedConv2d
from ogan.layers.resample import upsample2d


class NoiseInjection(nn.Module):
    """`B` in P1 Fig. 2 — one Gaussian map per layer, one learned scalar.

    The scalar initialises to zero, so a freshly built network ignores noise
    entirely and only learns to use it if that helps. The map is single-channel
    and broadcasts across features; it is drawn per sample, since the stochastic
    variation it exists for — P2 §3.2's placement of hairs, freckles, pores — is
    a property of an individual image.

    Pass `noise` explicitly to make a forward pass reproducible.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            n, _, h, w = x.shape
            noise = torch.randn(n, 1, h, w, device=x.device, dtype=x.dtype)
        return x + self.gain * noise

    def extra_repr(self) -> str:
        return f"gain={self.gain.item():.6g}"


class SynthesisLayer(nn.Module):
    """One style block, plus the operations P1 §2.1 moves outside it.

    In order: optional filtered upsampling, the affine transform producing the
    style, the modulated convolution, the noise, the bias, the activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        w_dim: int,
        *,
        kernel_size: int = 3,
        upsample: bool = False,
    ) -> None:
        super().__init__()
        self.upsample = upsample
        self.affine = EqualizedLinear(w_dim, in_channels, bias_init=1.0)
        self.conv = ModulatedConv2d(in_channels, out_channels, kernel_size)
        self.noise = NoiseInjection()
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(
        self, x: torch.Tensor, w: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.upsample:
            x = upsample2d(x)
        x = self.conv(x, self.affine(w))
        x = self.noise(x, noise)
        return leaky_relu(x + self.bias.reshape(1, -1, 1, 1))

    def extra_repr(self) -> str:
        return f"upsample={self.upsample}"
