"""Tests for the synthesis network (P1 §4.1, App. B; P3 Table 2).

`test_parameter_counts_match_the_paper` is the strongest test in the suite. P1
footnote 4 states the generator's size in both configurations and the percentage
between them, so the channel table, the block layout, the output skips, the tRGB
layers and the mapping network are all checked at once, against a number nobody
could tune towards without getting the architecture right.
"""

import pytest
import torch

from ogan.mapping import MappingNetwork
from ogan.synthesis import CHANNELS, SynthesisNetwork, channels_at

W_DIM = 512


def count(module) -> int:
    return sum(p.numel() for p in module.parameters())


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# --- capacity (P3 Table 2, P1 footnote 4) ---------------------------------


def test_channels_match_progan_table_2():
    assert CHANNELS == {
        4: 512,
        8: 512,
        16: 512,
        32: 512,
        64: 256,
        128: 128,
        256: 64,
        512: 32,
        1024: 16,
    }


def test_the_large_configuration_doubles_from_64_upwards():
    """P1 footnote 4: "in resolutions 64²-1024², while keeping other parts of the
    networks unchanged"."""
    for res in (4, 8, 16, 32):
        assert channels_at(res, large=True) == channels_at(res)
    for res in (64, 128, 256, 512, 1024):
        assert channels_at(res, large=True) == 2 * channels_at(res)


def test_parameter_counts_match_the_paper():
    """P1 footnote 4: the generator goes from 25M parameters to 30M, up 22%.

    Everything upstream has to be right for this to land: the per-resolution
    channel counts, two convolutions per block and one at 4², a modulated tRGB at
    every resolution, and the mapping network — whose inclusion this test also
    settles, since the synthesis network alone is 22.8M.
    """
    mapping = count(MappingNetwork())
    small = count(SynthesisNetwork(1024)) + mapping
    large = count(SynthesisNetwork(1024, large=True)) + mapping

    assert small / 1e6 == pytest.approx(25.0, abs=0.3)
    assert large / 1e6 == pytest.approx(30.0, abs=0.5)
    assert (large / small - 1) * 100 == pytest.approx(22.0, abs=0.5)


# --- the constant input (P1 App. B, §2.1) ---------------------------------


def test_the_constant_is_a_bare_parameter():
    """§2.1 removes bias, noise and normalisation from the constant input, so it
    is not a layer with those switched off — there is nothing to switch."""
    first = SynthesisNetwork(64).blocks[0]
    assert first.const.shape == (1, 512, 4, 4)
    assert not hasattr(first, "conv0")


def test_the_constant_initialises_unit_normal():
    """P1 App. B: "we initialize components of the constant input c1 using N(0,1)"."""
    const = SynthesisNetwork(64).blocks[0].const
    assert const.pow(2).mean().sqrt().item() == pytest.approx(1.0, abs=0.1)


def test_the_constant_is_shared_across_the_batch():
    net = SynthesisNetwork(16)
    ws = torch.randn(4, net.num_ws, W_DIM)
    assert net.blocks[0].const.shape[0] == 1
    assert net(ws).shape[0] == 4


# --- output skips (P1 §4.1, Fig. 7b) --------------------------------------


def test_output_shape():
    for res in (8, 16, 64):
        net = SynthesisNetwork(res)
        assert net(torch.randn(2, net.num_ws, W_DIM)).shape == (2, 3, res, res)


def test_every_resolution_contributes_to_the_image():
    """§4.1 forms the image by "upsampling and summing the contributions of RGB
    outputs corresponding to different resolutions" — so silencing any one of
    them must change the result, including the lowest."""
    net = SynthesisNetwork(32)
    ws = torch.randn(2, net.num_ws, W_DIM)
    baseline = net(ws)

    for index, block in enumerate(net.blocks):
        with torch.no_grad():
            saved = block.to_rgb.conv.weight.clone()
            block.to_rgb.conv.weight.zero_()
        assert not torch.allclose(net(ws), baseline), f"block {index} contributed nothing"
        with torch.no_grad():
            block.to_rgb.conv.weight.copy_(saved)


def test_the_image_is_not_merely_the_last_contribution():
    net = SynthesisNetwork(32)
    ws = torch.randn(2, net.num_ws, W_DIM)
    full = net(ws)

    with torch.no_grad():
        for block in net.blocks[:-1]:
            block.to_rgb.conv.weight.zero_()
            block.to_rgb.bias.zero_()
    assert not torch.allclose(net(ws), full)


def test_to_rgb_is_not_demodulated():
    """P1 App. B: demodulation everywhere "except for the output layers (tRGB)"."""
    for block in SynthesisNetwork(32).blocks:
        assert block.to_rgb.conv.demodulate is False


# --- the w tensor ---------------------------------------------------------


def test_num_ws_counts_every_style_input():
    """Two at 4² (one convolution, one tRGB) and three at every doubling."""
    net = SynthesisNetwork(512)
    assert net.num_ws == 2 + 3 * 7
    assert SynthesisNetwork(4).num_ws == 2


def test_wrong_ws_shape_is_rejected():
    net = SynthesisNetwork(16)
    with pytest.raises(ValueError, match=r"ws must be"):
        net(torch.randn(2, net.num_ws + 1, W_DIM))


def test_style_mixing_changes_the_image():
    """P2 §3.1 switches latents at a crossover point in the synthesis network,
    which only means anything if the later entries of w are actually used."""
    net = SynthesisNetwork(32)
    first = torch.randn(2, 1, W_DIM).expand(-1, net.num_ws, -1).clone()
    mixed = first.clone()
    mixed[:, 4:] = torch.randn(2, 1, W_DIM).expand(-1, net.num_ws - 4, -1)
    assert not torch.allclose(net(first), net(mixed))


def test_a_bad_resolution_is_rejected():
    with pytest.raises(ValueError, match="power of two"):
        SynthesisNetwork(48)


# --- gradients ------------------------------------------------------------


def test_every_parameter_receives_a_gradient():
    net = SynthesisNetwork(32)
    net(torch.randn(2, net.num_ws, W_DIM)).square().mean().backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"


def test_path_length_shaped_double_backward():
    """P1 §3.2's identity, taken through the whole synthesis network."""
    net = SynthesisNetwork(16)
    ws = torch.randn(2, net.num_ws, W_DIM, requires_grad=True)

    image = net(ws)
    grad = torch.autograd.grad((image * torch.randn_like(image)).sum(), ws, create_graph=True)[0]
    grad.square().sum(dim=(1, 2)).sqrt().sub(1.0).square().mean().backward()

    const_grad = net.blocks[0].const.grad
    assert const_grad is not None and torch.isfinite(const_grad).all()
