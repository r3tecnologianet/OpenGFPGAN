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


def test_layer_output_starts_near_unit_second_moment():
    """Demodulation holds the convolution at unit scale, the gain restores what
    the activation removes, and at initialisation noise and bias contribute
    nothing. So a fresh layer neither amplifies nor attenuates."""
    layer = SynthesisLayer(32, 32, W_DIM)
    out = layer(torch.randn(8, 32, 16, 16), torch.randn(8, W_DIM))
    assert rms(out) == pytest.approx(1.0, abs=0.05)


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


def test_path_length_shaped_double_backward_reaches_all_parameters():
    layer = SynthesisLayer(16, 16, W_DIM, upsample=True)
    x = torch.randn(4, 16, 8, 8)
    latent = torch.randn(4, W_DIM, requires_grad=True)

    image = layer(x, latent, noise=torch.randn(4, 1, 16, 16))
    projection = (image * torch.randn_like(image)).sum()
    grad = torch.autograd.grad(projection, latent, create_graph=True)[0]  # P1 §3.2
    grad.norm(dim=1).sub(1.0).square().mean().backward()

    for name, p in layer.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"


def test_noise_gain_learns_only_once_noise_is_present():
    """At zero gain the parameter still receives gradient — otherwise it could
    never leave zero."""
    layer = SynthesisLayer(16, 16, W_DIM)
    out = layer(torch.randn(4, 16, 8, 8), torch.randn(4, W_DIM))
    out.square().mean().backward()
    assert layer.noise.gain.grad is not None
