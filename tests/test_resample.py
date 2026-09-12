"""Numerical tests for filtered 2× resampling (P2, config B).

Two tests carry the weight. `test_upsampling_is_exactly_bilinear` checks that the
kernel does what P2 says it is implementing, which is what closes the derivation
of the normalisation. `test_downsampling_nulls_the_nyquist_checkerboard` checks
the reason the filter is there at all.
"""

import pytest
import torch

from ogan.layers import BINOMIAL_2, EqualizedConv2d, downsample2d, leaky_relu, upsample2d

TOL = 1e-5


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# --- the kernel -----------------------------------------------------------


def test_kernel_is_the_second_order_binomial():
    """P2 names a separable 2nd order binomial filter: [1, 2, 1]."""
    assert BINOMIAL_2 == (1.0, 2.0, 1.0)


def test_kernel_is_not_the_third_order_one():
    """[1, 3, 3, 1] appears in no paper and is not exact bilinear."""
    assert len(BINOMIAL_2) == 3
    assert BINOMIAL_2 != (1.0, 3.0, 3.0, 1.0)


# --- shape ----------------------------------------------------------------


def test_upsample_doubles_and_downsample_halves():
    x = torch.randn(2, 3, 8, 12)
    assert upsample2d(x).shape == (2, 3, 16, 24)
    assert downsample2d(x).shape == (2, 3, 4, 6)


def test_resampling_is_depthwise():
    """Channels must not mix: the filter is separable *and* per channel."""
    x = torch.zeros(1, 3, 8, 8)
    x[:, 1] = 1.0
    up = upsample2d(x)
    assert up[:, 0].abs().max() == 0.0
    assert up[:, 2].abs().max() == 0.0
    assert up[:, 1].abs().max() > 0.0


# --- DC gain, which is what the two normalisations are chosen for ----------


def test_upsampling_preserves_a_constant():
    out = upsample2d(torch.full((1, 2, 8, 8), 3.0))
    assert out[..., 2:-2, 2:-2].mean().item() == pytest.approx(3.0, abs=TOL)


def test_downsampling_preserves_a_constant():
    out = downsample2d(torch.full((1, 2, 16, 16), 3.0))
    assert out[..., 1:-1, 1:-1].mean().item() == pytest.approx(3.0, abs=TOL)


def test_zero_padding_lowers_the_border():
    """Not a defect, and pinned so nobody later "fixes" it into a shifted image."""
    out = upsample2d(torch.ones(1, 1, 8, 8))
    assert out[0, 0, -1, -1].item() < 1.0
    assert out[..., 2:-2, 2:-2].min().item() == pytest.approx(1.0, abs=TOL)


# --- the derivation, checked ---------------------------------------------


def test_upsampling_is_exactly_bilinear():
    """Even outputs reproduce the input; odd outputs are the neighbour mean.

    This is what fixes the normalisation. Writing the 1D kernel as [a, b, a] over
    a zero-inserted signal, bilinear interpolation forces b = 1 and a = 1/2, so
    k_up = [1, 2, 1] / 2 with no freedom left. A kernel that failed this test
    would not be implementing what P2 says it implements.
    """
    ramp = torch.arange(8, dtype=torch.float32).reshape(1, 1, 1, 8).expand(1, 1, 8, 8)
    row = upsample2d(ramp.contiguous())[0, 0, 4]

    torch.testing.assert_close(row[0:14:2], ramp[0, 0, 0, 0:7], atol=TOL, rtol=0)
    midpoints = (ramp[0, 0, 0, 0:7] + ramp[0, 0, 0, 1:8]) / 2
    torch.testing.assert_close(row[1:14:2], midpoints, atol=TOL, rtol=0)


def test_downsampling_nulls_the_nyquist_checkerboard():
    """Why the filter exists: decimating without it aliases Nyquist onto DC.

    A ±1 checkerboard is the highest frequency the grid can carry. Taking every
    second sample of it lands on one parity and returns a constant ±1 — the
    signal has aliased onto DC. The 1D response of [1, 2, 1]/4 at frequency π is
    (1 - 2 + 1)/4 = 0, so the filtered path nulls it instead.
    """
    i = torch.arange(16).reshape(16, 1)
    j = torch.arange(16).reshape(1, 16)
    board = ((-1.0) ** (i + j)).reshape(1, 1, 16, 16)

    aliased = board[..., ::2, ::2]
    assert aliased.abs().mean().item() == pytest.approx(1.0, abs=TOL)

    filtered = downsample2d(board)
    assert filtered[..., 1:-1, 1:-1].abs().max().item() == pytest.approx(0.0, abs=TOL)


def test_downsampling_nulls_single_axis_nyquist_too():
    i = torch.arange(16).reshape(1, 1, 16, 1)
    stripes = ((-1.0) ** i).expand(1, 1, 16, 16).contiguous()
    assert downsample2d(stripes)[..., 1:-1, 1:-1].abs().max().item() == pytest.approx(0.0, abs=TOL)


# --- composition ----------------------------------------------------------


def test_up_then_down_returns_a_smooth_field():
    """Not an identity — the pair is a lowpass — but it must not drift in scale."""
    x = torch.full((1, 2, 16, 16), 2.0)
    out = downsample2d(upsample2d(x))
    assert out.shape == x.shape
    assert out[..., 2:-2, 2:-2].mean().item() == pytest.approx(2.0, abs=TOL)


# --- dtype, device and autograd ------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_dtype_is_preserved(dtype):
    x = torch.ones(1, 2, 8, 8, dtype=dtype)
    assert upsample2d(x).dtype == dtype
    assert downsample2d(x).dtype == dtype


def test_resampling_alone_has_no_second_derivative():
    """Because it is linear, and worth pinning so the next test is not misread.

    Both operations are fixed convolutions, so the Jacobian is constant and does
    not depend on the input. Autograd still builds the double-backward node — the
    gradient carries a ConvolutionBackwardBackward grad_fn — but x is not
    reachable through it, so differentiating again returns nothing. That is an
    identity, not a failure, and it is why the next test routes the second
    derivative through a parameter instead.
    """
    x = torch.randn(2, 3, 8, 8, requires_grad=True)
    grad = torch.autograd.grad(downsample2d(upsample2d(x)).sum(), x, create_graph=True)[0]
    assert grad.requires_grad
    assert grad.grad_fn is not None
    second = torch.autograd.grad(grad.square().sum(), x, allow_unused=True)[0]
    assert second is None


def test_second_derivatives_flow_through_resampling_to_parameters():
    """The requirement that actually matters.

    R1 differentiates the discriminator twice, and its path runs through
    downsampling; path length differentiates the generator twice, through
    upsampling. Neither needs resampling to be nonlinear — it needs the second
    derivative to reach the parameters on the other side of it.
    """
    conv = EqualizedConv2d(3, 3, 3, padding=1)
    x = torch.randn(2, 3, 8, 8, requires_grad=True)

    out = downsample2d(leaky_relu(conv(upsample2d(x))))
    grad = torch.autograd.grad(out.sum(), x, create_graph=True)[0]
    grad.square().sum().backward()

    assert conv.weight.grad is not None
    assert torch.isfinite(conv.weight.grad).all()
    assert conv.weight.grad.abs().sum() > 0
