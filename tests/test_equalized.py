"""Numerical tests for the equalized-learning-rate layers.

Each test names the invariant it checks and the source of the value. A test that
only asserts the code runs is not a test of a clean-room port.

Everything here runs on the CPU: the invariants are algebraic, not hardware
dependent, and keeping them CPU-only means they run anywhere, including CI.
"""

import math

import pytest
import torch

from ogan.layers import (
    ACTIVATION_GAIN,
    LRELU_SLOPE,
    EqualizedConv2d,
    EqualizedLinear,
    leaky_relu,
)

TOL = 0.01  # single layer, ~2M samples
STACK_DEPTH = 10  # layers in the unbiasedness check
STACK_SEEDS = 32  # seeds averaged over; standard error of the geometric mean ≈0.02


def rms(t: torch.Tensor) -> float:
    """Root mean square — the quantity the gain preserves, not the std."""
    return t.pow(2).mean().sqrt().item()


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# --- the constants ---------------------------------------------------------


def test_slope_is_the_papers_value():
    """P3 §A.1, restated in P1 App. B."""
    assert LRELU_SLOPE == 0.2


def test_gain_matches_its_derivation():
    """E[y²] = (1+a²)/2 for unit-normal input, so the gain is sqrt(2/(1+a²))."""
    assert ACTIVATION_GAIN == pytest.approx(math.sqrt(2.0 / 1.04), rel=1e-12)
    assert ACTIVATION_GAIN == pytest.approx(1.38675049, abs=1e-8)


def test_gain_is_not_the_relu_constant():
    """sqrt(2) is He's constant for plain ReLU. These networks are LeakyReLU."""
    assert ACTIVATION_GAIN != pytest.approx(math.sqrt(2.0), abs=1e-3)


# --- initialisation (P3 §4.1 and §A.1) -------------------------------------


@pytest.mark.parametrize(
    "layer",
    [EqualizedLinear(512, 512), EqualizedConv2d(64, 64, 3)],
    ids=["linear", "conv"],
)
def test_weights_are_unit_normal(layer):
    assert rms(layer.weight) == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize(
    "layer",
    [EqualizedLinear(8, 8), EqualizedConv2d(8, 8, 3)],
    ids=["linear", "conv"],
)
def test_biases_are_zero(layer):
    assert torch.all(layer.bias == 0)


def test_scale_is_inverse_sqrt_fan_in():
    assert EqualizedLinear(512, 64).scale == pytest.approx(512**-0.5)
    # fan_in for a convolution counts the kernel footprint: 64 * 3 * 3
    assert EqualizedConv2d(64, 32, 3).scale == pytest.approx((64 * 9) ** -0.5)


# --- the invariant each unit is responsible on its own ---------------------


def test_linear_preserves_second_moment():
    layer = EqualizedLinear(512, 512, bias=False)
    assert rms(layer(torch.randn(4096, 512))) == pytest.approx(1.0, abs=TOL)


def test_conv_preserves_second_moment():
    """Unpadded, so every output element sums over a full kernel footprint."""
    layer = EqualizedConv2d(64, 64, 3, padding=0, bias=False)
    assert rms(layer(torch.randn(32, 64, 34, 34))) == pytest.approx(1.0, abs=TOL)


def test_zero_padding_lowers_the_second_moment_at_the_border():
    """Not a defect, and worth pinning so nobody later "fixes" it.

    A padded output element on the border sums over zeros, so it carries less
    energy than an interior one. The invariant holds on the interior; the full
    tensor reads low in proportion to how much of it is border.
    """
    layer = EqualizedConv2d(64, 64, 3, padding=1, bias=False)
    y = layer(torch.randn(32, 64, 32, 32))
    assert rms(y[..., 1:-1, 1:-1]) == pytest.approx(1.0, abs=TOL)
    assert rms(y) < rms(y[..., 1:-1, 1:-1])


def test_activation_preserves_second_moment():
    assert rms(leaky_relu(torch.randn(2048, 1024))) == pytest.approx(1.0, abs=TOL)


def test_activation_does_not_preserve_the_standard_deviation():
    """The distinction that makes this suite correct rather than plausible.

    LeakyReLU leaves a positive mean, so the gain that fixes E[y²] leaves the
    standard deviation at ≈0.897. Analytically, E[y]·gain = 0.3989·(1-a)·gain,
    and std = sqrt(1 - E[y]²). A suite asserting std ∈ [0.99, 1.01] here would
    fail while the implementation was correct.
    """
    y = leaky_relu(torch.randn(2048, 1024))
    expected = math.sqrt(1.0 - (0.3989423 * (1 - LRELU_SLOPE) * ACTIVATION_GAIN) ** 2)
    assert y.std().item() == pytest.approx(expected, abs=TOL)
    assert y.std().item() < 0.95


# --- the composition, and the bug it guards against -----------------------


def test_layer_then_activation_preserves_second_moment():
    """The gain belongs to exactly one of the two. This fails if it is in both."""
    layer = EqualizedLinear(512, 512, bias=False)
    assert rms(leaky_relu(layer(torch.randn(4096, 512)))) == pytest.approx(1.0, abs=TOL)


def test_deep_stack_is_unbiased_across_seeds():
    """The invariant holds *per layer in expectation*, not exactly through a stack.

    A single finite-width random layer's empirical gain deviates from one by a
    few percent — the sampling noise of projecting a particular input onto a
    particular weight matrix, which is O(1/sqrt(units)) ≈ 4% at 512 units. Those
    deviations compound multiplicatively, so one ten-layer stack lands anywhere
    in roughly [0.8, 1.3]. That is a random walk, not a bias.

    So the testable claim is unbiasedness: the geometric mean over many seeds
    returns to one. A systematic error does not average out, which is what gives
    this test teeth — see the negative control below.
    """
    assert _geomean_stack_rms(double_gain=False) == pytest.approx(1.0, abs=0.1)


def test_applying_the_gain_twice_is_caught_by_that_test():
    """Negative control: the bug the layer/activation split exists to prevent.

    Folding the gain into both the weight scale and the activation multiplies by
    ≈1.387 per layer over the correct value. Ten layers inflate by ≈26×, so the
    unbiasedness test above separates right from wrong by a factor of 26 — not by
    a margin that a tolerance could hide.
    """
    inflated = _geomean_stack_rms(double_gain=True)
    assert inflated > 10.0
    assert inflated == pytest.approx(ACTIVATION_GAIN**STACK_DEPTH, rel=0.2)


def _geomean_stack_rms(*, double_gain: bool) -> float:
    logs = []
    for seed in range(STACK_SEEDS):
        torch.manual_seed(seed)
        x = torch.randn(1024, 512)
        for _ in range(STACK_DEPTH):
            x = leaky_relu(EqualizedLinear(512, 512, bias=False)(x))
            if double_gain:
                x = x * ACTIVATION_GAIN
        logs.append(math.log(rms(x)))
    return math.exp(sum(logs) / len(logs))
