"""P4's colour transform category, as a composed 4x4 matrix.

Provenance (see PROVENANCE.md) — P4 Appendix B.1, Figure 22, verbatim:

    C <- I4
    apply brightness with probability p:  b ~ N(0, 0.2^2)
        C <- Translate3D(b, b, b) . C
    apply contrast with probability p:    c ~ Lognormal(0, (0.5 ln2)^2)
        C <- Scale3D(c, c, c) . C
    v <- [1, 1, 1, 0] / sqrt(3)                                    # the luma axis
    apply luma flip with probability p:   i ~ U{0, 1}
        C <- (I4 - 2 v^T v . i) . C                        # Householder reflection
    apply hue rotation with probability p: theta ~ U(-pi, +pi)
        C <- Rotate3D(v, theta) . C
    apply saturation with probability p:  s ~ Lognormal(0, (1 ln2)^2)
        C <- (v^T v + (I4 - v^T v) . s) . C
    for each pixel: [r,g,b,a] <- C . [r,g,b,1], keep (r,g,b)

**Every distribution here has the identity as its median**, which is what makes
the category safe to compose: `N(0, .)` is centred on zero for a translation, and
`Lognormal(0, .)` has median one for a scale. An augmentation drawn at the median
of every parameter leaves the image untouched, so raising `p` increases variance
rather than introducing a bias — and a bias is precisely what would leak into the
generator.

The luma axis is the grey diagonal of the colour cube. Hue rotation turns about
it, so it cannot change how bright a pixel is; saturation scales the distance
from it; the flip reflects through the plane perpendicular to it.
"""

import math

import torch

BRIGHTNESS_STD = 0.2
CONTRAST_LOG_STD = 0.5 * math.log(2.0)
SATURATION_LOG_STD = 1.0 * math.log(2.0)


def _luma_axis(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor([1.0, 1.0, 1.0, 0.0], device=device, dtype=dtype) / math.sqrt(3.0)


def _taken(n: int, p: float, device: torch.device) -> torch.Tensor:
    """Which images take this transform. Drawn afresh per transform, per §2."""
    return torch.rand(n, device=device) < p


def _or_default(value: torch.Tensor, taken: torch.Tensor, default: float) -> torch.Tensor:
    return torch.where(taken, value, torch.full_like(value, default))


def color_matrices(n: int, p: float, device=None, dtype=torch.float32) -> torch.Tensor:
    """Draw one 4x4 colour transform per image, in P4's order."""
    device = device or torch.device("cpu")
    eye = torch.eye(4, device=device, dtype=dtype)
    c = eye.expand(n, 4, 4).clone()

    # brightness: a translation along the grey diagonal
    brightness = torch.randn(n, device=device, dtype=dtype) * BRIGHTNESS_STD
    brightness = _or_default(brightness, _taken(n, p, device), 0.0)
    translate = eye.expand(n, 4, 4).clone()
    translate[:, :3, 3] = brightness.reshape(n, 1)
    c = translate @ c

    # contrast: a scale about the origin
    contrast = torch.exp(torch.randn(n, device=device, dtype=dtype) * CONTRAST_LOG_STD)
    contrast = _or_default(contrast, _taken(n, p, device), 1.0)
    scale = eye.expand(n, 4, 4).clone()
    scale[:, :3, :3] = torch.diag_embed(contrast.reshape(n, 1).expand(n, 3))
    c = scale @ c

    v = _luma_axis(device, dtype)
    outer = torch.outer(v, v)

    # luma flip: a Householder reflection through the plane normal to the axis
    flip = torch.randint(0, 2, (n,), device=device, dtype=dtype)
    flip = flip * _taken(n, p, device).to(dtype)
    c = (eye - 2.0 * outer * flip.reshape(n, 1, 1)) @ c

    # hue: a rotation about the axis (Rodrigues, in homogeneous form)
    theta = (torch.rand(n, device=device, dtype=dtype) * 2.0 - 1.0) * math.pi
    theta = theta * _taken(n, p, device).to(dtype)
    axis = v[:3]
    cross = torch.tensor(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        device=device,
        dtype=dtype,
    )
    eye3 = torch.eye(3, device=device, dtype=dtype)
    sin, cos = theta.reshape(n, 1, 1).sin(), theta.reshape(n, 1, 1).cos()
    rotation = eye.expand(n, 4, 4).clone()
    rotation[:, :3, :3] = eye3 + sin * cross + (1.0 - cos) * (cross @ cross)
    c = rotation @ c

    # saturation: scale the distance from the axis, leaving the axis itself alone
    saturation = torch.exp(torch.randn(n, device=device, dtype=dtype) * SATURATION_LOG_STD)
    saturation = _or_default(saturation, _taken(n, p, device), 1.0)
    c = (outer + (eye - outer) * saturation.reshape(n, 1, 1)) @ c

    return c


def apply_color(images: torch.Tensor, matrices: torch.Tensor) -> torch.Tensor:
    """`[r,g,b,a] <- C . [r,g,b,1]`, keeping the first three."""
    n, channels, height, width = images.shape
    if channels != 3:
        raise ValueError(f"colour transforms expect 3 channels, got {channels}")
    flat = images.reshape(n, 3, -1)
    transformed = matrices[:, :3, :3] @ flat + matrices[:, :3, 3:4]
    return transformed.reshape(n, 3, height, width)


def color_transform(images: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0.0:
        return images
    matrices = color_matrices(images.shape[0], p, images.device, images.dtype)
    return apply_color(images, matrices)
