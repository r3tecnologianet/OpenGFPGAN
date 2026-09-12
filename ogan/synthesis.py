"""The synthesis network g: W -> image, with output skips.

Provenance (see PROVENANCE.md):

- P3 Table 2 gives the feature map count per resolution, and it is a table of
  resolutions rather than of depths: 512 up to 32², then 256, 128, 64, 32, 16.
  A 512² network is the 1024² one without its last block.
- P1 footnote 4: the larger configuration doubles the feature maps "in
  resolutions 64²-1024² while keeping other parts of the networks unchanged",
  raising the generator from 25M parameters to 30M.
- P1 §4.1 and Fig. 7b: output skips, formed by "upsampling and summing the
  contributions of RGB outputs corresponding to different resolutions", with
  bilinear filtering in all up and downsampling operations.
- P1 App. B: the constant input is initialised `N(0, 1)`.
- P1 §2.1: "The application of bias, noise, and normalization to the constant
  input can also be safely removed without observable drawbacks" — so the
  constant is a bare parameter, not a layer with those operations disabled.

**The number of style inputs is ours.** P2 counts "18 layers — two for each
resolution" for a 1024² StyleGAN, where tRGB was a single unmodulated output
layer. StyleGAN2 modulates a tRGB at every resolution, so that count does not
carry over, and P1 does not restate it. Here every style input takes its own
entry in `w`: a plain, auditable rule, at the cost of a wider `w` than an
implementation that shares entries between a block's tRGB and its successor.
"""

import math

import torch
from torch import nn

from ogan.layers import SynthesisLayer, ToRGB, upsample2d
from ogan.mapping import W_DIM

#: P3 Table 2, indexed by resolution.
CHANNELS = {4: 512, 8: 512, 16: 512, 32: 512, 64: 256, 128: 128, 256: 64, 512: 32, 1024: 16}

#: P1 footnote 4: the larger configuration doubles feature maps from 64² up.
LARGE_FROM_RESOLUTION = 64


def channels_at(resolution: int, *, large: bool = False) -> int:
    if resolution not in CHANNELS:
        raise ValueError(f"no channel count for resolution {resolution}")
    count = CHANNELS[resolution]
    return count * 2 if large and resolution >= LARGE_FROM_RESOLUTION else count


class SynthesisBlock(nn.Module):
    """One resolution: its convolutions, and its contribution to the image."""

    def __init__(self, in_channels: int, out_channels: int, w_dim: int, resolution: int) -> None:
        super().__init__()
        self.resolution = resolution
        self.is_first = resolution == 4

        if self.is_first:
            # P1 §2.1: no bias, no noise, no normalisation here. A bare parameter.
            self.const = nn.Parameter(torch.randn(1, out_channels, 4, 4))
            self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim)
        else:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim, upsample=True)
            self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim)
        self.to_rgb = ToRGB(out_channels, w_dim)

    @property
    def num_styles(self) -> int:
        return 2 if self.is_first else 3

    def forward(
        self, x: torch.Tensor | None, image: torch.Tensor | None, ws: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_first:
            x = self.const.to(ws.dtype).expand(ws.shape[0], -1, -1, -1)
            x = self.conv1(x, ws[:, 0])
        else:
            x = self.conv0(x, ws[:, 0])
            x = self.conv1(x, ws[:, 1])

        contribution = self.to_rgb(x, ws[:, self.num_styles - 1])
        image = contribution if image is None else upsample2d(image) + contribution
        return x, image


class SynthesisNetwork(nn.Module):
    """Blocks from 4² up to `resolution`, summing RGB contributions as it goes."""

    def __init__(self, resolution: int = 512, w_dim: int = W_DIM, *, large: bool = False) -> None:
        super().__init__()
        if resolution < 4 or resolution & (resolution - 1):
            raise ValueError(f"resolution must be a power of two and at least 4, got {resolution}")
        self.resolution = resolution
        self.w_dim = w_dim

        resolutions = [2**i for i in range(2, int(math.log2(resolution)) + 1)]
        blocks, previous = [], 0
        for res in resolutions:
            out_channels = channels_at(res, large=large)
            blocks.append(SynthesisBlock(previous, out_channels, w_dim, res))
            previous = out_channels
        self.blocks = nn.ModuleList(blocks)

    @property
    def num_ws(self) -> int:
        return sum(block.num_styles for block in self.blocks)

    def forward(self, ws: torch.Tensor) -> torch.Tensor:
        if ws.shape[1:] != (self.num_ws, self.w_dim):
            raise ValueError(f"ws must be [N, {self.num_ws}, {self.w_dim}], got {tuple(ws.shape)}")

        x, image, index = None, None, 0
        for block in self.blocks:
            x, image = block(x, image, ws[:, index : index + block.num_styles])
            index += block.num_styles
        return image

    def extra_repr(self) -> str:
        return f"resolution={self.resolution}, w_dim={self.w_dim}, num_ws={self.num_ws}"
