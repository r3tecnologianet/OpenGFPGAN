"""Tests for the discriminator (P3 Table 2 and §3; P1 §4.1 and footnotes 3-4)."""

import math

import pytest
import torch

from ogan.discriminator import (
    RESIDUAL_SCALE,
    STDDEV_EPS,
    Discriminator,
    DiscriminatorBlock,
    minibatch_stddev,
)
from ogan.layers import EqualizedConv2d, downsample2d, leaky_relu
from ogan.synthesis import channels_at


def count(module) -> int:
    return sum(p.numel() for p in module.parameters())


def rms(t: torch.Tensor) -> float:
    return t.pow(2).mean().sqrt().item()


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# --- the paper's own size, which checks the whole structure at once -------


def test_parameter_counts_match_the_paper():
    """P1 footnote 4: 24M parameters, rising to 29M, up 21%.

    ProGAN's discriminator at 1024² is 23.1M (P3 Table 2). The residual design
    adds the 1x1 skip convolutions and lands on 24M, so this figure checks the
    channel mirror, the two convolutions per resolution, the skip projections
    and the 4x4 tail together.
    """
    small, large = count(Discriminator(1024)), count(Discriminator(1024, large=True))
    assert small / 1e6 == pytest.approx(24.0, abs=0.3)
    assert large / 1e6 == pytest.approx(29.0, abs=0.3)
    assert (large / small - 1) * 100 == pytest.approx(21.0, abs=0.5)


def test_capacity_mirrors_the_generator():
    net = Discriminator(64)
    assert net.from_rgb.out_channels == channels_at(64)
    assert [b.conv1.out_channels for b in net.blocks] == [channels_at(r) for r in (32, 16, 8, 4)]


# --- minibatch standard deviation (P3 §3, §A.1) ---------------------------


def test_it_appends_exactly_one_feature_map():
    out = minibatch_stddev(torch.randn(4, 8, 4, 4))
    assert out.shape == (4, 9, 4, 4)


def test_the_appended_map_is_one_constant_value():
    """P3 §3: averaged "to arrive at a single value", replicated "to all spatial
    locations and over the minibatch"."""
    appended = minibatch_stddev(torch.randn(4, 8, 4, 4))[:, -1]
    assert appended.min().item() == pytest.approx(appended.max().item(), abs=1e-7)


def test_an_identical_batch_reports_no_diversity():
    """The signal the layer exists to send. Mode collapse drives it to zero, and
    the floor is sqrt(eps) rather than zero because of the guard."""
    same = torch.randn(1, 8, 4, 4).expand(6, -1, -1, -1)
    value = minibatch_stddev(same.contiguous())[:, -1].mean().item()
    assert value == pytest.approx(math.sqrt(STDDEV_EPS), abs=1e-6)
    assert value < 1e-3


def test_a_diverse_batch_reports_diversity():
    value = minibatch_stddev(torch.randn(16, 8, 4, 4))[:, -1].mean().item()
    assert value == pytest.approx(1.0, abs=0.1)


def test_it_has_no_parameters_and_no_hyperparameters():
    """P3 §3: "neither learnable parameters nor new hyperparameters"."""
    net = Discriminator(16)
    assert not any("stddev" in name for name, _ in net.named_parameters())


def test_a_batch_of_one_survives():
    """Zero variance by construction, and a NaN here would stop a run."""
    assert torch.isfinite(minibatch_stddev(torch.randn(1, 8, 4, 4))).all()


def test_the_statistic_sits_at_4x4():
    """P3 §A.1: at 4x4 resolution, toward the end. The epilogue's first
    convolution takes the extra channel, which is where that shows up."""
    net = Discriminator(64)
    assert net.epilogue.conv.in_channels == channels_at(4) + 1


# --- the residual junction (P1 Fig. 7 footnote 3) -------------------------


def test_the_residual_scale_is_the_papers_constant():
    assert RESIDUAL_SCALE == pytest.approx(2.0**-0.5)


def test_the_scale_keeps_the_junction_from_inflating():
    """Footnote 3: the junction doubles signal variance and the factor cancels it.

    Measured against the same block with the factor removed, which is how the
    footnote frames the problem. It is not an absolute claim about the output:
    the two branches are not independent, and both pass through a downsample
    whose gain on white input is 0.375, so a block fed white noise emerges near
    0.42 whatever the junction does. That is the filter, not the junction.
    """
    block = DiscriminatorBlock(32, 32)
    scaled = rms(block(torch.randn(8, 32, 16, 16)))

    assert (scaled / RESIDUAL_SCALE) / scaled == pytest.approx(math.sqrt(2.0), rel=1e-5)
    assert scaled == pytest.approx(0.42, abs=0.1)


def test_the_skip_projection_and_the_filter_commute():
    """So their order on the skip path is not a decision, and neither reading of
    Fig. 7c's "Down ... includes 1x1 convolutions" changes the result."""
    conv = EqualizedConv2d(16, 24, 1, bias=False)
    x = torch.randn(4, 16, 16, 16)
    torch.testing.assert_close(downsample2d(conv(x)), conv(downsample2d(x)), atol=1e-5, rtol=0)


def test_a_block_halves_resolution_and_changes_width():
    out = DiscriminatorBlock(32, 64)(torch.randn(2, 32, 16, 16))
    assert out.shape == (2, 64, 8, 8)


# --- the network ----------------------------------------------------------


@pytest.mark.parametrize("resolution", [8, 16, 64])
def test_output_is_one_score_per_image(resolution):
    net = Discriminator(resolution)
    assert net(torch.randn(3, 3, resolution, resolution)).shape == (3, 1)


def test_a_bad_resolution_is_rejected():
    with pytest.raises(ValueError, match="power of two"):
        Discriminator(48)


def test_a_mismatched_image_is_rejected():
    with pytest.raises(ValueError, match=r"image must be"):
        Discriminator(16)(torch.randn(2, 3, 32, 32))


# --- gradients ------------------------------------------------------------


def test_every_parameter_receives_a_gradient():
    net = Discriminator(32)
    net(torch.randn(4, 3, 32, 32)).sum().backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"


def _r1_gradients(net, resolution):
    """P1 App. B evaluates R1 on real samples: the gradient of the score with
    respect to the input, squared, differentiated again."""
    real = torch.randn(4, 3, resolution, resolution, requires_grad=True)
    grad = torch.autograd.grad(net(real).sum(), real, create_graph=True)[0]
    grad.square().sum().backward()
    return {name: p.grad for name, p in net.named_parameters()}


def test_r1_reaches_the_weights_and_finds_no_infinities():
    net = Discriminator(32)
    grads = _r1_gradients(net, 32)
    for name, grad in grads.items():
        if grad is not None:
            assert torch.isfinite(grad).all(), f"{name} has non-finite gradient"
    assert grads["from_rgb.weight"].abs().sum() > 0


def test_r1_does_not_reach_the_biases_after_the_diversity_statistic():
    """Three parameters get nothing, and the pattern says why.

    Convolutions, resampling and the linear head are linear; LeakyReLU is
    piecewise linear. So the score's gradient with respect to its input is
    piecewise constant, and differentiating it again with respect to a bias gives
    zero — the bias only moves where the pieces meet, a set of measure zero. The
    final linear bias is not even reachable: it shifts the score by a constant,
    so it never enters the input gradient at all.
    """
    grads = _r1_gradients(Discriminator(32), 32)
    assert grads["epilogue.out.bias"] is None
    assert grads["epilogue.conv.bias"].abs().sum() == 0.0
    assert grads["epilogue.collapse.bias"].abs().sum() == 0.0


def test_the_diversity_statistic_is_the_only_source_of_curvature():
    """And therefore the only reason R1 reaches any bias at all.

    Measured on the real discriminator: 7 of 10 biases receive second-order
    signal with the statistic in place and 0 of 10 without it. Isolated here on
    a two-layer stand-in, because the claim is about the operation rather than
    about this network — the statistic takes a square root of a mean of squares
    over the batch, which is the one genuinely curved thing in the path.
    """

    def second_order_bias_gradient(with_statistic):
        torch.manual_seed(3)
        conv = EqualizedConv2d(3, 4, 3, padding=1)
        head = torch.nn.Linear(5 if with_statistic else 4, 1)
        x = torch.randn(4, 3, 8, 8, requires_grad=True)

        h = leaky_relu(conv(x))
        if with_statistic:
            h = minibatch_stddev(h)
        score = head(h.mean(dim=(2, 3))).sum()
        grad = torch.autograd.grad(score, x, create_graph=True)[0]
        grad.square().sum().backward()
        return conv.bias.grad.abs().sum().item()

    assert second_order_bias_gradient(with_statistic=False) == 0.0
    assert second_order_bias_gradient(with_statistic=True) > 0.0
