"""Filtered 2× resampling — the operation P2 calls bilinear sampling.

Provenance (see PROVENANCE.md):

- P2, improved baseline (config B): "replace the nearest-neighbor up/downsampling
  in both networks with bilinear sampling, which we implement by lowpass
  filtering the activations with a separable 2nd order binomial filter after each
  upsampling layer and before each downsampling layer". Second order binomial
  coefficients are `[1, 2, 1]`.
- P1 App. B lists bilinear filtering in all up/downsampling layers among the
  details kept unchanged from StyleGAN, so P1 inherits rather than redefines it.

**The normalisation is derived, and it confirms the kernel.** Take upsampling as
insertion of zeros followed by the filter, and write the 1D kernel as `[a, b, a]`.
Output samples that land on an original position receive `b·x`; those between two
originals receive `a·(x_left + x_right)`. Bilinear interpolation wants the first
to be `x` and the second to be their mean, so `b = 1` and `a = 1/2`:

    k_up = [1, 2, 1] / 2

That is exactly the filter P2 names, producing exactly the interpolation P2 says
it is implementing, with no free parameter left over. Downsampling instead wants
unit DC gain, so it normalises to sum one:

    k_down = [1, 2, 1] / 4

In two dimensions each is the outer product of its 1D kernel with itself, giving
totals of 4 and 1 respectively.

`[1, 3, 3, 1]`, which every implementation uses, is third order and is not exact
bilinear. It appears in none of P1, P2 or P3.

**Upsampling has two gains, and only one of them is 1.** The normalisation above
fixes the *DC* gain: a constant field upsamples to the same constant. White noise
is a different matter. After zero insertion the four output parities see
different subsets of the kernel — in 2D, weights 1, 1/2, 1/2 and 1/4 — so their
variances are 1, 1/2, 1/2 and 1/4 of the input's, averaging 9/16. The white-noise
gain is therefore `sqrt(9/16) = 0.75`, measured 0.7522.

This is not a defect to correct. It is inherent to filtered upsampling, and
`[1, 3, 3, 1]` is worse at 0.625 by the same calculation. Real feature maps are
neither constant nor white, so a layer that upsamples lands between the two
figures. What matters is that anything claiming to preserve second moments must
say which input it means.
"""

import torch
import torch.nn.functional as F

#: P2: a separable 2nd order binomial filter.
BINOMIAL_2 = (1.0, 2.0, 1.0)


def _kernel2d(scale: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    k = torch.tensor(BINOMIAL_2, device=device, dtype=dtype) / scale
    return torch.outer(k, k)


def upsample2d(x: torch.Tensor) -> torch.Tensor:
    """Insert zeros to double the resolution, then lowpass. `[N,C,H,W] -> [N,C,2H,2W]`.

    Implemented as a transposed convolution with stride 2, which *is* zero
    insertion followed by convolution, done in one pass and with the gradient
    handled by autograd.
    """
    n, c, h, w = x.shape
    k = _kernel2d(2.0, x.device, x.dtype).expand(c, 1, 3, 3)
    return F.conv_transpose2d(
        x, k.transpose(0, 1).reshape(c, 1, 3, 3), stride=2, padding=1, output_padding=1, groups=c
    )


def downsample2d(x: torch.Tensor) -> torch.Tensor:
    """Lowpass, then keep every second sample. `[N,C,H,W] -> [N,C,H/2,W/2]`.

    The filter comes first: that is the whole point of P2's change, since
    decimating an unfiltered signal aliases everything above the new Nyquist
    limit back into the band.
    """
    c = x.shape[1]
    k = _kernel2d(4.0, x.device, x.dtype).expand(c, 1, 3, 3)
    return F.conv2d(x, k, stride=2, padding=1, groups=c)
