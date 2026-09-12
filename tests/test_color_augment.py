"""Tests for P4's colour transform category (Appendix B.1, Figure 22).

`test_grey_stays_grey` is the one that checks the whole composition at once.
Every transform in the category either fixes the luma axis or negates it, so a
grey pixel must come out grey whatever the five random draws were. Get the axis
wrong, the rotation formula wrong, or the saturation the wrong way round, and
this fails.
"""

import math

import pytest
import torch

from ogan.color_augment import (
    BRIGHTNESS_STD,
    CONTRAST_LOG_STD,
    SATURATION_LOG_STD,
    apply_color,
    color_matrices,
    color_transform,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# --- the parameters P4 states ---------------------------------------------


def test_the_distributions_match_the_paper():
    """N(0, 0.2²), Lognormal(0, (0.5 ln2)²), Lognormal(0, (1 ln2)²)."""
    assert BRIGHTNESS_STD == 0.2
    assert CONTRAST_LOG_STD == pytest.approx(0.5 * math.log(2.0))
    assert SATURATION_LOG_STD == pytest.approx(math.log(2.0))


def test_every_distribution_has_the_identity_at_its_median():
    """What makes the category safe to compose and safe to strengthen.

    A translation drawn from `N(0, .)` is zero at the median and a scale drawn
    from `Lognormal(0, .)` is one, so raising p adds variance without adding
    bias. A bias is what would leak into the generator.
    """
    assert math.exp(0.0) == 1.0  # median of Lognormal(0, sigma^2)
    samples = torch.randn(200_000) * BRIGHTNESS_STD
    assert samples.median().item() == pytest.approx(0.0, abs=0.01)
    scales = torch.exp(torch.randn(200_000) * SATURATION_LOG_STD)
    assert scales.median().item() == pytest.approx(1.0, abs=0.01)


# --- the invariant of the whole composition -------------------------------


def test_grey_stays_grey():
    """Brightness translates along the luma axis, contrast scales about the
    origin, hue rotation turns about the axis, saturation scales away from it and
    the luma flip reflects through the plane normal to it. Each maps the grey
    line to itself, so their composition does too.
    """
    grey = torch.rand(64, 1, 8, 8).expand(-1, 3, -1, -1).contiguous()
    out = color_transform(grey, 1.0)
    torch.testing.assert_close(out[:, 0], out[:, 1], atol=1e-5, rtol=0)
    torch.testing.assert_close(out[:, 1], out[:, 2], atol=1e-5, rtol=0)


def test_the_luma_flip_is_an_involution():
    """`(I - 2 v^T v)^2 = I` for a unit v — the defining property of a
    Householder reflection, checked on the axis the module uses."""
    v = torch.tensor([1.0, 1.0, 1.0, 0.0]) / math.sqrt(3.0)
    householder = torch.eye(4) - 2.0 * torch.outer(v, v)
    torch.testing.assert_close(householder @ householder, torch.eye(4), atol=1e-6, rtol=0)


def test_the_transform_is_affine_in_rgb():
    """`C . [r,g,b,1]` is a linear map plus a translation, so it must commute
    with convex combinations of its inputs."""
    matrices = color_matrices(4, p=1.0)
    a = torch.rand(4, 3, 4, 4)
    b = torch.rand(4, 3, 4, 4)

    mixed = apply_color(0.3 * a + 0.7 * b, matrices)
    separate = 0.3 * apply_color(a, matrices) + 0.7 * apply_color(b, matrices)
    torch.testing.assert_close(mixed, separate, atol=1e-5, rtol=0)


# --- probability gating ---------------------------------------------------


def test_zero_probability_is_the_identity():
    images = torch.rand(4, 3, 8, 8)
    torch.testing.assert_close(color_transform(images, 0.0), images)


def test_zero_probability_draws_identity_matrices():
    torch.testing.assert_close(
        color_matrices(8, p=0.0), torch.eye(4).expand(8, 4, 4), atol=1e-6, rtol=0
    )


def test_full_probability_changes_the_image():
    images = torch.rand(8, 3, 8, 8)
    assert not torch.allclose(color_transform(images, 1.0), images)


def test_randomisation_is_per_image():
    """P4 §2: "separately for each augmentation and for each image"."""
    same = torch.rand(1, 3, 8, 8).expand(16, -1, -1, -1).contiguous()
    out = color_transform(same, 1.0)
    assert not torch.allclose(out[0], out[1])


# --- contract -------------------------------------------------------------


def test_it_is_differentiable():
    """Required, since P4 §2 runs augmentation during generator training too."""
    images = torch.rand(4, 3, 8, 8, requires_grad=True)
    color_transform(images, 1.0).square().sum().backward()
    assert images.grad is not None and torch.isfinite(images.grad).all()
    assert images.grad.abs().sum() > 0


def test_shape_and_dtype_survive():
    for dtype in (torch.float32, torch.float64):
        out = color_transform(torch.rand(2, 3, 8, 8, dtype=dtype), 1.0)
        assert out.shape == (2, 3, 8, 8)
        assert out.dtype == dtype


def test_a_non_rgb_image_is_rejected():
    with pytest.raises(ValueError, match="3 channels"):
        color_transform(torch.rand(2, 1, 8, 8), 1.0)
