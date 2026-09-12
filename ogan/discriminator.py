"""The discriminator: a residual network mirroring the synthesis capacity.

Provenance (see PROVENANCE.md):

- P3 Table 2, discriminator column: fRGB is a 1x1 convolution to the feature
  count of the input resolution, each resolution holds two 3x3 convolutions
  (the second widening to the next resolution's count) followed by a
  downsample, and the 4x4 tail is minibatch standard deviation, a 3x3
  convolution, a 4x4 convolution to 1x1, and a linear layer to one output.
- P3 §3: the minibatch statistic is the standard deviation per feature per
  spatial location over the minibatch, averaged over all features and locations
  "to arrive at a single value", replicated and concatenated as "one additional
  (constant) feature map". No learnable parameters and no hyperparameters.
- P3 §A.1: it goes at 4x4 resolution, toward the end of the discriminator.
- P1 §4.1 and Fig. 7c: configurations E and F use residual connections, where
  "Down" also includes "1x1 convolutions to adjust the number of feature maps".
- P1 Fig. 7 footnote 3: the residual junction causes a "doubling of signal
  variance, which we cancel by multiplying with 1/sqrt(2)". The footnote adds
  that this "is crucial for our networks, whereas in classification resnets the
  issue is typically hidden by batch normalization".
- P1 footnote 4: 24M parameters, rising to 29M in the larger configuration.

**Two orderings that look like decisions and are not.** On the skip path, the
1x1 convolution acts per pixel and the resampling filter acts per channel, so
they commute exactly — filtering before or after the projection gives the same
tensor. And P3's 4x4 convolution to 1x1 is the same map as a dense layer over
the flattened 4x4 block, with the same 512x512x16 parameters. The table's form is
used here because it is the one written down.
"""

import math

import torch
from torch import nn

from ogan.layers import EqualizedConv2d, EqualizedLinear, downsample2d, leaky_relu
from ogan.synthesis import CHANNEL_MAX, channels_at

RESIDUAL_SCALE = 2.0**-0.5  # P1 Fig. 7 footnote 3
STDDEV_EPS = 1e-8


def minibatch_stddev(x: torch.Tensor, eps: float = STDDEV_EPS) -> torch.Tensor:
    """Append P3 §3's diversity statistic as one constant feature map.

    `eps` is ours: P3 gives the operation without one, and the square root of a
    variance that is exactly zero — which happens whenever a batch collapses,
    the very case this layer exists to report — has no gradient.
    """
    deviation = x - x.mean(dim=0, keepdim=True)
    stddev = deviation.square().mean(dim=0).add(eps).sqrt()  # per feature, per location
    value = stddev.mean()  # "a single value"
    return torch.cat([x, value.expand(x.shape[0], 1, *x.shape[2:])], dim=1)


class DiscriminatorBlock(nn.Module):
    """One resolution: two convolutions and a downsample, with a residual skip."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv0 = EqualizedConv2d(in_channels, in_channels, 3, padding=1)
        self.conv1 = EqualizedConv2d(in_channels, out_channels, 3, padding=1)
        self.skip = EqualizedConv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = downsample2d(self.skip(x))
        x = leaky_relu(self.conv0(x))
        x = leaky_relu(downsample2d(self.conv1(x)))
        return (x + skip) * RESIDUAL_SCALE


class DiscriminatorEpilogue(nn.Module):
    """The 4x4 tail: the diversity statistic, then a score."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = EqualizedConv2d(channels + 1, channels, 3, padding=1)
        self.collapse = EqualizedConv2d(channels, channels, 4)
        self.out = EqualizedLinear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = leaky_relu(self.conv(minibatch_stddev(x)))
        x = leaky_relu(self.collapse(x))
        return self.out(x.flatten(1))


class Discriminator(nn.Module):
    """Mirrors the synthesis network's capacity, from `resolution` down to 4x4."""

    def __init__(
        self, resolution: int = 512, *, large: bool = False, channel_max: int = CHANNEL_MAX
    ) -> None:
        super().__init__()
        if resolution < 4 or resolution & (resolution - 1):
            raise ValueError(f"resolution must be a power of two and at least 4, got {resolution}")
        self.resolution = resolution

        def width(res: int) -> int:
            return channels_at(res, large=large, channel_max=channel_max)

        self.from_rgb = EqualizedConv2d(3, width(resolution), 1)
        self.blocks = nn.ModuleList(
            DiscriminatorBlock(width(res), width(res // 2))
            for res in (2**i for i in range(int(math.log2(resolution)), 2, -1))
        )
        self.epilogue = DiscriminatorEpilogue(width(4))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1:] != (3, self.resolution, self.resolution):
            raise ValueError(
                f"image must be [N, 3, {self.resolution}, {self.resolution}], "
                f"got {tuple(image.shape)}"
            )
        x = leaky_relu(self.from_rgb(image))
        for block in self.blocks:
            x = block(x)
        return self.epilogue(x)

    def extra_repr(self) -> str:
        return f"resolution={self.resolution}"
