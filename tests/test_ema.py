"""Tests for the averaged generator (P3 §A.1, P4).

`test_the_half_life_is_a_half_life` is the one that matters: it checks the
conversion against the definition of the word, by running the average over
exactly one half-life and measuring how far it travelled.
"""

import copy

import pytest
import torch
from torch import nn

from ogan.ema import (
    EMA_DECAY_PROGAN,
    EMA_HALF_LIFE_IMAGES,
    AveragedGenerator,
    decay_from_half_life,
    half_life_from_decay,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def tiny() -> nn.Module:
    return nn.Linear(4, 4)


# --- the two notations ----------------------------------------------------


def test_the_papers_constants():
    assert EMA_HALF_LIFE_IMAGES == 20_000
    assert EMA_DECAY_PROGAN == 0.999


def test_the_two_papers_agree_at_the_minibatch_they_share():
    """P4 states a 20k half-life, P3 a per-step decay of 0.999. At P1's minibatch
    of 32 the first is a decay of 0.99889 and the second a half-life of 22.2k —
    the same mechanism in two notations, 10% apart."""
    assert decay_from_half_life(20_000, 32) == pytest.approx(0.998892, abs=1e-6)
    assert half_life_from_decay(0.999, 32) == pytest.approx(22_170, rel=0.01)


def test_the_conversions_invert_each_other():
    for batch in (8, 32, 64):
        decay = decay_from_half_life(20_000, batch)
        assert half_life_from_decay(decay, batch) == pytest.approx(20_000, rel=1e-9)


def test_a_larger_batch_means_a_faster_per_step_decay():
    """The reason the half-life form is the one used: it says what it means when
    the batch size changes, and a fixed per-step decay silently does not."""
    decays = [decay_from_half_life(20_000, b) for b in (8, 16, 32, 64)]
    assert decays == sorted(decays, reverse=True)


def test_the_half_life_is_a_half_life():
    """Run the average over exactly one half-life and it must cover half the gap.

    This is the definition of the word, checked rather than assumed, and it is
    what the conversion has to get right.
    """
    half_life, batch = 2_000, 25
    steps = half_life // batch

    source = tiny()
    ema = AveragedGenerator(source, half_life_images=half_life, batch_size=batch)
    with torch.no_grad():
        for p in source.parameters():
            p.fill_(1.0)
        for p in ema.module.parameters():
            p.fill_(0.0)

    for _ in range(steps):
        ema.update(source)

    for p in ema.module.parameters():
        assert p.mean().item() == pytest.approx(0.5, abs=0.01)


# --- the wrapper ----------------------------------------------------------


def test_it_starts_as_a_copy():
    source = tiny()
    ema = AveragedGenerator(source)
    for averaged, live in zip(ema.module.parameters(), source.parameters(), strict=True):
        torch.testing.assert_close(averaged, live)


def test_the_average_carries_no_gradient():
    """It is an artifact, not a participant: nothing optimises it."""
    ema = AveragedGenerator(tiny())
    assert all(not p.requires_grad for p in ema.module.parameters())


def test_it_starts_in_eval_mode():
    assert not AveragedGenerator(tiny()).module.training


def test_updating_leaves_the_source_untouched():
    source = tiny()
    before = copy.deepcopy(source.state_dict())
    AveragedGenerator(source).update(source)
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_it_converges_to_a_stationary_source():
    source = tiny()
    ema = AveragedGenerator(source, half_life_images=100, batch_size=50)
    with torch.no_grad():
        for p in source.parameters():
            p.fill_(3.0)

    for _ in range(200):
        ema.update(source)
    for p in ema.module.parameters():
        assert p.mean().item() == pytest.approx(3.0, abs=1e-3)


def test_it_lags_a_moving_source():
    """The property the average exists for: it is smoother than what it tracks."""
    source = tiny()
    ema = AveragedGenerator(source, half_life_images=10_000, batch_size=32)
    with torch.no_grad():
        for p in source.parameters():
            p.fill_(0.0)

    for step in range(50):
        with torch.no_grad():
            for p in source.parameters():
                p.fill_(float(step))
        ema.update(source)

    averaged = next(iter(ema.module.parameters())).mean().item()
    assert averaged < 49.0
    assert averaged > 0.0


def test_buffers_are_copied_rather_than_averaged():
    """A buffer holds counted or measured state; interpolating one has no meaning."""
    source = nn.BatchNorm1d(4)
    ema = AveragedGenerator(source)
    with torch.no_grad():
        source.running_mean.fill_(7.0)
    ema.update(source)
    torch.testing.assert_close(ema.module.running_mean, source.running_mean)


def test_the_average_round_trips_through_a_state_dict():
    """It belongs in every checkpoint: restoring the live generator and
    recomputing the average from it would silently restart the average."""
    source = tiny()
    ema = AveragedGenerator(source)
    for _ in range(5):
        with torch.no_grad():
            for p in source.parameters():
                p.add_(0.1)
        ema.update(source)

    restored = AveragedGenerator(tiny())
    restored.load_state_dict(ema.state_dict())
    for a, b in zip(restored.module.parameters(), ema.module.parameters(), strict=True):
        torch.testing.assert_close(a, b)


def test_it_can_be_called_like_the_generator_it_wraps():
    ema = AveragedGenerator(tiny())
    assert ema(torch.randn(2, 4)).shape == (2, 4)
