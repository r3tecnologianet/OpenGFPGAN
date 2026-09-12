"""Weight modulation and demodulation — the operation StyleGAN2 is named for.

Provenance (see PROVENANCE.md), all of it P1:

- Eq. 1, modulation: `w'[i,j,k] = s[i] · w[i,j,k]`, scaling each input feature
  map of the convolution by its style.
- Eq. 2: under the assumption that input activations are i.i.d. with unit
  standard deviation, the output of the modulated convolution has standard
  deviation `σ[j] = sqrt(Σ_ik w'[i,j,k]²)` — the L2 norm of the corresponding
  weights.
- Eq. 3, demodulation: divide by that norm, so the output returns to unit scale.
  Baked into the weights rather than applied to the activations, which is what
  removes the droplet artifact: the normalisation is based on the expected
  statistics of the signal instead of its actual contents, so the generator can
  no longer smuggle signal strength past it with a localised spike (§2.2).
- App. B, *Weight demodulation*: the modulated weights differ per sample, which
  "rules out direct implementation using standard convolution primitives". The
  paper's own answer is grouped convolution — reshape so that "each convolution
  sees one sample with N groups, instead of N samples with one group", noting
  that the reshapes "do not actually modify the contents".
- App. B, *Generator redesign*: demodulation applies in every convolution
  **except** the output layers, where modulation alone lets the magnitude of the
  RGB signal follow the style.

No bias is added here. P1 §2 moves the addition of the bias and the noise
outside the active area of a style, so both belong to the block that wraps this
layer, not to the layer.
"""

import torch
import torch.nn.functional as F
from torch import nn


class ModulatedConv2d(nn.Module):
    """A convolution whose weights are modulated by a per-sample style.

    Takes the styles as an argument rather than owning the affine layer that
    produces them: P1 Fig. 2 draws `A` as a separate box, and keeping it separate
    means this layer can be tested against arbitrary styles, including the
    extreme ones that demodulation exists to absorb.

    `eps` guards the division in Eq. 3. P1 calls it "a small constant to avoid
    numerical issues" without giving a value, so the value is **ours**: 1e-8
    matches the magnitude the same authors use elsewhere (P3 §4.2 for pixel
    normalisation, P1 App. B for Adam).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        demodulate: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.eps = eps
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.scale = (in_channels * kernel_size * kernel_size) ** -0.5
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor, styles: torch.Tensor) -> torch.Tensor:
        """`x` is `[N, C_in, H, W]`, `styles` is `[N, C_in]`, output is `[N, C_out, H, W]`."""
        n, c_in, h, w_spatial = x.shape
        if styles.shape != (n, c_in):
            raise ValueError(f"styles must be [{n}, {c_in}], got {tuple(styles.shape)}")

        weight = self.weight * self.scale
        weight = weight.unsqueeze(0) * styles.reshape(n, 1, c_in, 1, 1)  # Eq. 1
        if self.demodulate:
            norm = weight.square().sum(dim=(2, 3, 4), keepdim=True)  # Eq. 2, squared
            weight = weight * (norm + self.eps).rsqrt()  # Eq. 3

        # App. B: one sample with N groups, instead of N samples with one group.
        x = x.reshape(1, n * c_in, h, w_spatial)
        weight = weight.reshape(n * self.out_channels, c_in, self.kernel_size, self.kernel_size)
        out = F.conv2d(x, weight, padding=self.padding, groups=n)
        return out.reshape(n, self.out_channels, h, w_spatial)

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels} -> {self.out_channels}, k={self.kernel_size}, "
            f"demodulate={self.demodulate}, scale={self.scale:.6g}"
        )
