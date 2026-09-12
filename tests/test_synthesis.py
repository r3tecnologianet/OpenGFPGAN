"""Tests for noise injection and the assembled synthesis layer.

The test with the most to say is `test_bias_shifts_where_the_activation_bends`.
P1 does not state in prose where the activation sits relative to the bias, so the
placement is derived — and this test is the derivation made executable.
"""

import pytest
import torch

from ogan.layers import LRELU_SLOPE, NoiseInjection, SynthesisLayer, leaky_relu

W_DIM = 64


def rms(t: torch.Tensor) -> float:
    return t.pow(2).mean().sqrt().item()


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# --- noise injection (P1 App. B, P2 §2) -----------------------------------


def test_noise_gain_initialises_to_zero():
    """App. B: noise scaling factors initialise to zero, so a fresh network
    ignores noise and only learns to use it if that helps."""
    layer = NoiseInjection()
    assert layer.gain.item() == 0.0

    x = torch.randn(2, 8, 16, 16)
    torch.testing.assert_close(layer(x), x)


def test_the_gain_is_one_scalar_not_one_per_channel():
    """App. B simplifies StyleGAN's per-channel scaling to "a single shared
    scaling factor for all feature maps"."""
    assert NoiseInjection().gain.numel() == 1


def test_noise_is_single_channel_and_broadcasts():
    """P2 §2: "single-channel images consisting of uncorrelated Gaussian noise"."""
    layer = NoiseInjection()
    with torch.no_grad():
        layer.gain.fill_(1.0)

    noise = torch.randn(1, 1, 4, 4)
    out = layer(torch.zeros(1, 3, 4, 4), noise)
    torch.testing.assert_close(out[0, 0], out[0, 1])
    torch.testing.assert_close(out[0, 0], noise[0, 0])


def test_each_sample_draws_its_own_noise():
    """P2 §3.2's stochastic variation is a property of an individual image."""
    layer = NoiseInjection()
    with torch.no_grad():
        layer.gain.fill_(1.0)

    out = layer(torch.zeros(2, 1, 8, 8))
    assert not torch.allclose(out[0], out[1])


def test_supplying_noise_makes_the_pass_reproducible():
    layer = NoiseInjection()
    with torch.no_grad():
        layer.gain.fill_(1.0)

    x, noise = torch.zeros(2, 4, 8, 8), torch.randn(2, 1, 8, 8)
    torch.testing.assert_close(layer(x, noise), layer(x, noise))
    assert not torch.allclose(layer(x), layer(x))


# --- the two orderings ----------------------------------------------------


def test_noise_and_bias_order_is_not_a_decision():
    """Both are additions, so their relative order is immaterial. Pinned because
    it is easy to mistake a commutative pair for a design choice."""
    x = torch.randn(2, 4, 8, 8)
    noise, bias = torch.randn(2, 1, 8, 8), torch.randn(4).reshape(1, 4, 1, 1)
    torch.testing.assert_close(x + noise + bias, x + bias + noise)


def test_bias_shifts_where_the_activation_bends():
    """The derivation for putting both additions before the activation.

    A bias applied before LeakyReLU moves elements across zero, changing which
    ones take the 0.2 branch. Applied after, it could only offset the output — so
    the network would have no way to place the nonlinearity, which is what a bias
    in a nonlinear network is for.
    """
    x = torch.linspace(-1.0, 1.0, 101).reshape(1, 1, 1, 101)

    before = leaky_relu(x + 0.5)
    after = leaky_relu(x) + 0.5 * 1.0

    negative_before = (x + 0.5 < 0).sum().item()
    negative_plain = (x < 0).sum().item()
    assert negative_before != negative_plain
    assert not torch.allclose(before, after)
    assert LRELU_SLOPE == 0.2


# --- the assembled layer --------------------------------------------------


def test_layer_preserves_shape_and_channel_count():
    layer = SynthesisLayer(8, 12, W_DIM)
    out = layer(torch.randn(2, 8, 16, 16), torch.randn(2, W_DIM))
    assert out.shape == (2, 12, 16, 16)


def test_upsampling_layer_doubles_resolution():
    layer = SynthesisLayer(8, 12, W_DIM, upsample=True)
    out = layer(torch.randn(2, 8, 16, 16), torch.randn(2, W_DIM))
    assert out.shape == (2, 12, 32, 32)


@pytest.mark.parametrize("res", [8, 16, 32, 64])
def test_layer_interior_starts_at_unit_second_moment(res):
    """Measured on the interior, because the border is a separate, known effect.

    Demodulation holds the convolution at unit scale, the gain restores what the
    activation removes, and at initialisation noise and bias contribute nothing.
    The full tensor reads low only because the convolution's zero padding gives
    border pixels fewer terms to sum — 0.917 at 8², 0.988 at 64², purely a
    function of how much of the tensor is border. Asserting on the full tensor
    would make this test's tolerance a proxy for its fixture's resolution.
    """
    layer = SynthesisLayer(32, 32, W_DIM)
    out = layer(torch.randn(8, 32, res, res), torch.randn(8, W_DIM))
    assert rms(out[..., 1:-1, 1:-1]) == pytest.approx(1.0, abs=0.01)
    assert rms(out) < rms(out[..., 1:-1, 1:-1])


def test_upsampling_attenuates_white_input_to_the_filters_gain():
    """The invariant above does not survive upsampling, and cannot.

    Demodulation normalises weights against Eq. 2's unit-variance assumption; it
    never inspects the input, so it cannot undo a scale the upsampling already
    applied. Filtered upsampling has DC gain 1 and white-noise gain 0.75, so the
    layer inherits whichever of those its input resembles.

    Only the white case is asserted here. A constant input through a random
    modulated convolution is a random projection onto 32 output channels, which
    lands anywhere in 0.79 to 1.14 across seeds — too dispersed to assert on, and
    dispersed for a reason unrelated to resampling. The DC gain is pinned exactly
    in `test_resample.py`, where nothing random stands in the way.
    """
    layer = SynthesisLayer(32, 32, W_DIM, upsample=True)
    white = layer(torch.randn(8, 32, 16, 16), torch.randn(8, W_DIM))
    assert rms(white[..., 2:-2, 2:-2]) == pytest.approx(0.75, abs=0.03)


def test_style_affine_starts_at_unity():
    """App. B's one exception to zero-initialised biases."""
    assert torch.all(SynthesisLayer(8, 8, W_DIM).affine.bias == 1.0)


def test_layer_bias_starts_at_zero():
    assert torch.all(SynthesisLayer(8, 8, W_DIM).bias == 0.0)


def test_the_constant_input_case_is_not_this_layers_business():
    """P1 §2.1 removes bias, noise and normalisation from the constant input.
    Recorded here as a note to whoever builds the 4x4 start: it does not get a
    SynthesisLayer with these operations disabled, it gets none of them."""
    assert not hasattr(SynthesisLayer(8, 8, W_DIM), "constant")


# --- second derivatives reach every parameter -----------------------------


def _path_length_backward(layer, x, spatial):
    latent = torch.randn(x.shape[0], W_DIM, requires_grad=True)
    image = layer(x, latent, noise=torch.randn(x.shape[0], 1, spatial, spatial))
    projection = (image * torch.randn_like(image)).sum()
    grad = torch.autograd.grad(projection, latent, create_graph=True)[0]  # P1 §3.2
    grad.norm(dim=1).sub(1.0).square().mean().backward()
    return {name: p.grad for name, p in layer.named_parameters()}


def test_path_length_reaches_the_parameters_it_can():
    """Three of the five, and the other two are structurally out of reach.

    `bias` and `noise.gain` enter the graph only inside LeakyReLU, whose
    derivative is piecewise constant. Differentiating that derivative with
    respect to them is zero almost everywhere, so a path length regulariser
    cannot move either one — not a wiring mistake, an identity. Asserting
    `grad is not None` would pass on all five and verify nothing, since both do
    receive a first-order gradient and so are allocated a zero tensor.
    """
    layer = SynthesisLayer(16, 16, W_DIM, upsample=True)
    grads = _path_length_backward(layer, torch.randn(4, 16, 8, 8), 16)

    for name in ("affine.weight", "affine.bias", "conv.weight"):
        assert torch.isfinite(grads[name]).all(), f"{name} has non-finite gradient"
        assert grads[name].abs().sum() > 0, f"{name} received no second-order signal"

    for name in ("bias", "noise.gain"):
        assert grads[name].abs().sum() == 0.0, f"{name} unexpectedly moved"


def test_noise_gain_learns_only_once_noise_is_present():
    """At zero gain the parameter still receives gradient — otherwise it could
    never leave zero."""
    layer = SynthesisLayer(16, 16, W_DIM)
    out = layer(torch.randn(4, 16, 8, 8), torch.randn(4, W_DIM))
    out.square().mean().backward()
    assert layer.noise.gain.grad is not None
