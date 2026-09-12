"""Tests for the mapping network (P1 App. B, P2 Fig. 1, P3 §4.2).

`test_the_mapping_network_learns_a_hundred_times_more_slowly` is the one that
matters. P1 states an effect — "100x lower learning rate" — and not a mechanism,
so the mechanism is ours and the test measures the effect instead of inspecting
the construction that produces it.
"""

import math

import pytest
import torch

from ogan.layers import EqualizedLinear
from ogan.mapping import (
    MAPPING_LAYERS,
    MAPPING_LR_MULTIPLIER,
    W_DIM,
    Z_DIM,
    MappingNetwork,
    normalize_latent,
)

TOL = 1e-4


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# --- the constants P1 App. B keeps from StyleGAN --------------------------


def test_architecture_matches_the_paper():
    net = MappingNetwork()
    assert (Z_DIM, W_DIM) == (512, 512)
    assert MAPPING_LAYERS == 8
    assert len(net.layers) == 8
    assert MAPPING_LR_MULTIPLIER == 0.01


# --- the Normalize block of P2 Fig. 1 -------------------------------------


def test_normalisation_puts_the_latent_on_the_hypersphere():
    """P3 §4.2's formula on a d-vector gives norm sqrt(d) (P3 §A.1)."""
    z = torch.randn(16, 512)
    assert normalize_latent(z).norm(dim=-1).allclose(torch.full((16,), math.sqrt(512)), atol=1e-3)


@pytest.mark.parametrize("scale", [0.1, 1.0, 10.0, 1e3])
def test_normalisation_discards_latent_magnitude(scale):
    """Only the direction survives, which is what makes Z a sphere."""
    z = torch.randn(4, 512)
    torch.testing.assert_close(normalize_latent(z * scale), normalize_latent(z), atol=1e-3, rtol=0)


def test_epsilon_breaks_scale_invariance_for_vanishing_latents():
    """Where the guarantee stops, stated rather than left to be discovered.

    The invariance above is exact only while `mean(a²)` dwarfs eps. Shrink the
    latent until the two are comparable and the normalisation starts shortening
    the output. At the scale latents actually arrive — z ~ N(0, I), so
    mean(a²) ≈ 1 against eps = 1e-8 — the effect is eight orders of magnitude
    away and cannot matter.
    """
    z = torch.randn(4, 512)
    unit = normalize_latent(z).norm(dim=-1)
    assert normalize_latent(z * 1e-3).norm(dim=-1).mean() < unit.mean()
    assert normalize_latent(z * 1e-3).norm(dim=-1).mean() > 0.99 * unit.mean()


def test_normalisation_survives_a_zero_latent():
    """eps is there for exactly this, and a NaN here would poison a whole batch."""
    assert torch.isfinite(normalize_latent(torch.zeros(2, 512))).all()


# --- the effect P1 states, measured --------------------------------------


def _effective_weight(layer: EqualizedLinear) -> torch.Tensor:
    return layer.weight.detach() * layer.scale


def _step_once(lr_multiplier: float, adam_eps: float) -> tuple[float, float]:
    """One Adam step; returns how far the effective weight moved, and |grad|."""
    torch.manual_seed(1)
    layer = EqualizedLinear(512, 512, lr_multiplier=lr_multiplier)
    before = _effective_weight(layer).clone()
    opt = torch.optim.Adam(layer.parameters(), lr=1e-3, betas=(0.0, 0.99), eps=adam_eps)
    opt.zero_grad()
    layer(torch.randn(64, 512)).square().mean().backward()
    gradient = layer.weight.grad.abs().mean().item()
    opt.step()
    return (_effective_weight(layer) - before).abs().mean().item(), gradient


def test_the_mechanism_scales_the_stored_gradient_by_the_multiplier():
    """The construction, checked without an optimiser in the way."""
    _, slow = _step_once(MAPPING_LR_MULTIPLIER, 1e-8)
    _, fast = _step_once(1.0, 1e-8)
    assert slow / fast == pytest.approx(MAPPING_LR_MULTIPLIER, rel=1e-3)


def test_the_mapping_network_learns_a_hundred_times_more_slowly():
    """P1's stated effect, measured on the effective weight after one Adam step.

    Adam's step size does not depend on gradient magnitude — with beta1 = 0 the
    first step is lr times the sign of the gradient — so the stored parameter
    moves the same amount either way and the effective weight moves by the
    runtime scale times that. Their ratio is the multiplier.
    """
    slow, _ = _step_once(MAPPING_LR_MULTIPLIER, 1e-12)
    fast, _ = _step_once(1.0, 1e-12)
    assert slow / fast == pytest.approx(MAPPING_LR_MULTIPLIER, rel=0.01)


def test_adams_epsilon_damps_the_mapping_network_further():
    """And at P1's own epsilon it is not 100x, it is about 112x.

    Scaling the stored gradient down by 100 lands the mapping network's
    gradients near 1.9e-7, where Adam's eps of 1e-8 (P1 App. B) is 5.4% of the
    denominator rather than a rounding term. The step is damped accordingly, so
    the layer learns slightly slower still.

    Faithful rather than wrong — the mechanism and the epsilon are both the
    paper's — but worth pinning, because "the mapping network barely moves" is
    otherwise a puzzle to be rediscovered during a training run.
    """
    slow, _ = _step_once(MAPPING_LR_MULTIPLIER, 1e-8)
    fast, _ = _step_once(1.0, 1e-8)
    ratio = slow / fast
    assert ratio == pytest.approx(0.0089, rel=0.05)
    assert ratio < MAPPING_LR_MULTIPLIER


def test_the_multiplier_does_not_change_the_layer_at_initialisation():
    """It changes how fast the layer learns, not what it computes on step zero."""
    x = torch.randn(8, 512)

    torch.manual_seed(2)
    slow = EqualizedLinear(512, 64, lr_multiplier=0.01)
    torch.manual_seed(2)
    fast = EqualizedLinear(512, 64)

    torch.testing.assert_close(slow(x), fast(x), atol=1e-4, rtol=0)


def test_the_multiplier_preserves_bias_init():
    """The stored bias is scaled up, so the effective one still starts where asked."""
    layer = EqualizedLinear(512, 8, bias_init=1.0, lr_multiplier=0.01)
    torch.testing.assert_close(layer.bias * layer.lr_multiplier, torch.ones(8))


def test_the_multiplier_still_preserves_the_second_moment():
    layer = EqualizedLinear(512, 512, bias=False, lr_multiplier=0.01)
    out = layer(torch.randn(2048, 512))
    assert out.pow(2).mean().sqrt().item() == pytest.approx(1.0, abs=0.01)


# --- the network ----------------------------------------------------------


def test_forward_shape():
    assert MappingNetwork()(torch.randn(4, 512)).shape == (4, 512)


def test_latent_magnitude_cannot_reach_w():
    """The Normalize block makes the network a function of direction alone."""
    net = MappingNetwork()
    z = torch.randn(4, 512)
    torch.testing.assert_close(net(z * 50.0), net(z), atol=1e-3, rtol=0)


def test_w_does_not_collapse_across_latents():
    """Eight layers at unit second moment should not funnel everything together."""
    w = MappingNetwork()(torch.randn(64, 512))
    assert w.std(dim=0).mean().item() > 0.1


def test_output_stays_at_a_sane_scale_through_eight_layers():
    w = MappingNetwork()(torch.randn(64, 512))
    assert 0.5 < w.pow(2).mean().sqrt().item() < 2.0


def test_wrong_latent_width_is_rejected():
    with pytest.raises(ValueError, match="512 components"):
        MappingNetwork()(torch.randn(4, 256))


def test_gradients_reach_every_layer():
    net = MappingNetwork()
    net(torch.randn(4, 512)).square().mean().backward()
    for name, p in net.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, f"{name} got no gradient"


def test_double_backward_through_the_mapping_network():
    """Path length differentiates through f as well as g."""
    net = MappingNetwork()
    z = torch.randn(4, 512, requires_grad=True)
    w = net(z)
    grad = torch.autograd.grad((w * torch.randn_like(w)).sum(), z, create_graph=True)[0]
    grad.square().sum().backward()
    assert torch.isfinite(net.layers[0].weight.grad).all()
