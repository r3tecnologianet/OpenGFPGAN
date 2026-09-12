"""Numerical tests for weight modulation and demodulation (P1 Eq. 1-3, App. B).

The load-bearing test here is `test_grouped_convolution_equals_a_per_sample_loop`.
P1 App. B claims the grouped reshape is a way of *implementing* Equations 1 and 3,
not an approximation of them. That claim is checkable against a naive loop, and
the loop is written from the equations, so the check needs no reference code.
"""

import pytest
import torch
import torch.nn.functional as F

from ogan.layers import EqualizedLinear, ModulatedConv2d


def rms(t: torch.Tensor) -> float:
    return t.pow(2).mean().sqrt().item()


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# --- the reshape does what App. B says it does ----------------------------


def naive_modulated_conv(layer: ModulatedConv2d, x: torch.Tensor, styles: torch.Tensor):
    """Equations 1 and 3 applied one sample at a time, with no grouping trick."""
    outputs = []
    for i in range(x.shape[0]):
        w = layer.weight * layer.scale
        w = w * styles[i].reshape(1, -1, 1, 1)  # Eq. 1
        if layer.demodulate:
            norm = w.square().sum(dim=(1, 2, 3), keepdim=True)  # Eq. 2, squared
            w = w * (norm + layer.eps).rsqrt()  # Eq. 3
        outputs.append(F.conv2d(x[i : i + 1], w, padding=layer.padding))
    return torch.cat(outputs, dim=0)


@pytest.mark.parametrize("demodulate", [True, False], ids=["demodulated", "modulated only"])
def test_grouped_convolution_equals_a_per_sample_loop(demodulate):
    layer = ModulatedConv2d(16, 24, 3, demodulate=demodulate)
    x = torch.randn(5, 16, 12, 12)
    styles = torch.randn(5, 16)
    torch.testing.assert_close(layer(x, styles), naive_modulated_conv(layer, x, styles))


def test_samples_do_not_leak_into_each_other():
    """The failure mode a grouped convolution invites: groups crossing samples."""
    layer = ModulatedConv2d(16, 16, 3)
    x = torch.randn(4, 16, 8, 8)
    styles = torch.randn(4, 16)
    before = layer(x, styles)

    changed = styles.clone()
    changed[0] += 5.0
    after = layer(x, changed)

    assert not torch.allclose(before[0], after[0])
    torch.testing.assert_close(before[1:], after[1:])


# --- what demodulation is for (P1 §2.2) -----------------------------------


@pytest.mark.parametrize("style_scale", [0.01, 0.5, 1.0, 10.0, 100.0])
def test_demodulation_absorbs_any_style_magnitude(style_scale):
    """§2.2: modulation "may amplify certain feature maps by an order of
    magnitude or more", and demodulation must counteract it per sample."""
    layer = ModulatedConv2d(32, 32, 3)
    x = torch.randn(8, 32, 16, 16)
    styles = torch.full((8, 32), style_scale)
    assert rms(layer(x, styles)) == pytest.approx(1.0, abs=0.05)


def test_without_demodulation_the_output_follows_the_style():
    """Negative control, and the property the output layers rely on (App. B)."""
    layer = ModulatedConv2d(32, 32, 3, demodulate=False)
    x = torch.randn(8, 32, 16, 16)
    unit = rms(layer(x, torch.ones(8, 32)))
    tenfold = rms(layer(x, torch.full((8, 32), 10.0)))
    assert tenfold == pytest.approx(10.0 * unit, rel=0.01)


def test_equation_2_predicts_the_pre_demodulation_scale():
    """Eq. 2 states σ[j] is the L2 norm of the modulated weights. Check it."""
    layer = ModulatedConv2d(32, 8, 3, demodulate=False)
    x = torch.randn(64, 32, 24, 24)
    styles = torch.randn(64, 32)
    out = layer(x, styles)

    w = (layer.weight * layer.scale).unsqueeze(0) * styles.reshape(64, 1, 32, 1, 1)
    predicted = w.square().sum(dim=(2, 3, 4)).sqrt()  # [N, C_out]
    measured = out.square().mean(dim=(2, 3)).sqrt()  # [N, C_out]
    assert (measured / predicted).mean().item() == pytest.approx(1.0, abs=0.05)


# --- a property of demodulation that is easy to get wrong later -----------


def test_the_equalized_scale_cancels_under_demodulation():
    """Demodulation is homogeneous of degree zero in the weight scale.

    Eq. 3 divides by the norm of the very weights Eq. 1 produced, so any constant
    factor in them — including `1/sqrt(fan_in)` — cancels exactly. The equalized
    scale is therefore a no-op in a demodulated convolution, and load-bearing
    only in an output layer, where demodulation is omitted.

    Pinned because it is the kind of thing someone later either removes as dead
    weight, or reintroduces twice on the assumption that it matters.
    """
    layer = ModulatedConv2d(16, 16, 3)
    x = torch.randn(4, 16, 8, 8)
    styles = torch.randn(4, 16)
    baseline = layer(x, styles)

    layer.scale *= 1000.0
    torch.testing.assert_close(layer(x, styles), baseline)


def test_the_equalized_scale_matters_without_demodulation():
    layer = ModulatedConv2d(16, 16, 3, demodulate=False)
    x = torch.randn(4, 16, 8, 8)
    styles = torch.randn(4, 16)
    baseline = layer(x, styles)

    layer.scale *= 2.0
    torch.testing.assert_close(layer(x, styles), baseline * 2.0)


# --- the style affine layer's one documented exception --------------------


def test_style_affine_biases_initialise_to_one():
    """App. B: all biases zero, "except for the biases of the affine
    transformation layers, which we initialize to one"."""
    affine = EqualizedLinear(512, 32, bias_init=1.0)
    assert torch.all(affine.bias == 1.0)
    assert torch.all(EqualizedLinear(512, 32).bias == 0.0)


# --- the premise the whole project rests on, pinned as a regression -------


def test_r1_shaped_double_backward():
    layer = ModulatedConv2d(16, 16, 3)
    x = torch.randn(4, 16, 8, 8, requires_grad=True)
    styles = torch.randn(4, 16)
    grad = torch.autograd.grad(layer(x, styles).sum(), x, create_graph=True)[0]
    grad.square().sum().backward()
    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()


def test_path_length_shaped_double_backward():
    layer = ModulatedConv2d(16, 16, 3)
    affine = EqualizedLinear(64, 16, bias_init=1.0)
    x = torch.randn(4, 16, 8, 8)
    latent = torch.randn(4, 64, requires_grad=True)

    image = layer(x, affine(latent))
    projection = (image * torch.randn_like(image)).sum()
    grad = torch.autograd.grad(projection, latent, create_graph=True)[0]  # P1 §3.2
    grad.norm(dim=1).sub(1.0).square().mean().backward()

    assert torch.isfinite(layer.weight.grad).all()
    assert torch.isfinite(affine.weight.grad).all()


# --- shape and contract ---------------------------------------------------


def test_output_shape_preserves_spatial_size():
    out = ModulatedConv2d(8, 12, 3)(torch.randn(2, 8, 17, 23), torch.randn(2, 8))
    assert out.shape == (2, 12, 17, 23)


def test_wrong_style_shape_is_rejected():
    layer = ModulatedConv2d(8, 8, 3)
    with pytest.raises(ValueError, match=r"styles must be"):
        layer(torch.randn(2, 8, 8, 8), torch.randn(2, 16))
