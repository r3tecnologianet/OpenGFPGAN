"""Tests for one training iteration and for resuming one (P1 App. B).

`test_a_resumed_run_matches_an_uninterrupted_one` is the reason this file exists.
Cloud training is preemptible, so a checkpoint that restores *almost* everything
does not fail — it continues into a run that is quietly not the one interrupted.
The only way to know is to interrupt a run and compare.
"""

import pytest
import torch

from ogan.loss import PL_INTERVAL, R1_INTERVAL
from ogan.training import Trainer, TrainingConfig

RESOLUTION, BATCH, CHANNEL_MAX, LATENT = 16, 4, 24, 32


def small(**overrides) -> TrainingConfig:
    """A real network in shape, narrowed so the suite stays in seconds.

    At 16 pixels the paper's table still asks for 512 channels at every
    resolution, and the mapping network is eight 512x512 layers whatever the
    image size — together a 14M-parameter generator. Narrowing both changes
    capacity, not architecture, and every structural property under test here is
    independent of it.
    """
    return TrainingConfig(
        resolution=RESOLUTION,
        batch_size=BATCH,
        channel_max=CHANNEL_MAX,
        z_dim=LATENT,
        w_dim=LATENT,
        **overrides,
    )


def batch() -> torch.Tensor:
    return torch.randn(BATCH, 3, RESOLUTION, RESOLUTION)


def fingerprint(trainer: Trainer) -> list[torch.Tensor]:
    return [p.detach().clone() for p in trainer.averaged.module.parameters()]


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# --- the lazy schedule ----------------------------------------------------


def test_regularizers_fire_on_their_own_intervals():
    """R1 every 16 iterations and path length every 8, per App. B."""
    trainer = Trainer(small())
    r1_steps, pl_steps = [], []
    for index in range(17):
        metrics = trainer.step(batch())
        if "r1" in metrics:
            r1_steps.append(index)
        if "path_length" in metrics:
            pl_steps.append(index)

    assert r1_steps == [0, R1_INTERVAL]
    assert pl_steps == [0, PL_INTERVAL, 2 * PL_INTERVAL]


def test_the_regularizer_takes_its_own_optimizer_step():
    """App. B runs k+1 optimizer steps where a naive reading runs k.

    "the optimizer first sees gradients from the main loss for k iterations,
    followed by gradients from the regularization terms for one iteration" — so
    folding `k * R1` into the discriminator loss would be a different algorithm,
    and the hyperparameter compensation would then be correcting for something
    that never happens.
    """
    trainer = Trainer(small())
    steps = 0
    original = trainer.opt_d.step

    def counted(*args, **kwargs):
        nonlocal steps
        steps += 1
        return original(*args, **kwargs)

    trainer.opt_d.step = counted
    for _ in range(R1_INTERVAL):
        trainer.step(batch())

    assert steps == R1_INTERVAL + 1


# --- one iteration --------------------------------------------------------


def test_a_step_moves_both_networks():
    trainer = Trainer(small())
    before_g = [p.detach().clone() for p in trainer.generator.parameters()]
    before_d = [p.detach().clone() for p in trainer.discriminator.parameters()]
    trainer.step(batch())

    assert any(
        not torch.equal(a, b) for a, b in zip(before_g, trainer.generator.parameters(), strict=True)
    )
    assert any(
        not torch.equal(a, b)
        for a, b in zip(before_d, trainer.discriminator.parameters(), strict=True)
    )


def test_the_average_lags_the_live_generator():
    trainer = Trainer(small())
    for _ in range(5):
        trainer.step(batch())
    assert any(
        not torch.equal(a, b)
        for a, b in zip(
            trainer.averaged.module.parameters(), trainer.generator.parameters(), strict=True
        )
    )


def test_the_gamma_comes_from_the_scaling_law():
    assert Trainer(small()).gamma == pytest.approx(0.0002 * RESOLUTION**2 / BATCH)
    assert Trainer(small(gamma=7.0)).gamma == 7.0


def test_augmentation_can_be_switched_off():
    trainer = Trainer(small(augment=False))
    assert trainer.augment is None
    assert "augment_p" not in trainer.step(batch())


# --- style mixing ---------------------------------------------------------


def test_mixing_off_gives_one_style_for_every_layer():
    generator = Trainer(small()).generator
    ws = generator.styles(torch.randn(4, generator.z_dim), style_mixing_prob=0.0)
    for layer in range(1, ws.shape[1]):
        torch.testing.assert_close(ws[:, layer], ws[:, 0])


def test_mixing_on_switches_latents_partway_through():
    """P2 §3.1: two latents, switching "at a randomly selected point"."""
    generator = Trainer(small()).generator
    ws = generator.styles(torch.randn(64, generator.z_dim), style_mixing_prob=1.0)
    differing = [not torch.equal(ws[i, 0], ws[i, -1]) for i in range(ws.shape[0])]
    assert sum(differing) > 50  # a crossover exists for nearly every image


# --- resumption -----------------------------------------------------------

RESUME_STEPS = PL_INTERVAL + 2  # a path length step falls on each side of the break


def test_a_resumed_run_matches_an_uninterrupted_one():
    """The property a preemptible run depends on, checked end to end.

    Ten steps straight through, against five, a checkpoint, a brand new trainer,
    and five more. The weights must agree exactly — not closely.

    The length is chosen, not arbitrary. It has to straddle a path length step,
    or the target is written before the break and never read after it, and the
    test passes while proving nothing about it. Dropping each checkpoint entry in
    turn was measured against this window: every one of rng, both optimizers, the
    average, the path length target and the augmentation state moves the result,
    so `atol=0` fails on each.
    """
    torch.manual_seed(7)
    straight = Trainer(small())
    for _ in range(RESUME_STEPS):
        straight.step(batch())
    expected = fingerprint(straight)

    torch.manual_seed(7)
    interrupted = Trainer(small())
    for _ in range(RESUME_STEPS // 2):
        interrupted.step(batch())
    checkpoint = interrupted.state_dict()

    resumed = Trainer(small())
    resumed.load_state_dict(checkpoint)
    for _ in range(RESUME_STEPS - RESUME_STEPS // 2):
        resumed.step(batch())

    for a, b in zip(expected, fingerprint(resumed), strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_the_checkpoint_carries_the_state_that_cannot_be_recomputed():
    """Each of these restarts something if it is missing, silently."""
    trainer = Trainer(small())
    for _ in range(PL_INTERVAL + 1):
        trainer.step(batch())

    state = trainer.state_dict()
    for key in (
        "step_index",
        "generator",
        "discriminator",
        "averaged",
        "opt_g",
        "opt_d",
        "path_length_target",
        "augment",
        "rng",
    ):
        assert key in state, f"{key} would be lost"


def test_the_optimizer_moments_survive_the_round_trip():
    """Adam's second moment is the state App. B's compensation is about."""
    trainer = Trainer(small())
    for _ in range(3):
        trainer.step(batch())

    restored = Trainer(small())
    restored.load_state_dict(trainer.state_dict())

    original = trainer.opt_d.state_dict()["state"]
    copy = restored.opt_d.state_dict()["state"]
    assert set(original) == set(copy)
    for key in original:
        torch.testing.assert_close(original[key]["exp_avg_sq"], copy[key]["exp_avg_sq"])


def test_the_path_length_target_survives():
    trainer = Trainer(small())
    for _ in range(PL_INTERVAL + 1):
        trainer.step(batch())
    assert trainer.path_length.target > 0.0

    restored = Trainer(small())
    restored.load_state_dict(trainer.state_dict())
    assert restored.path_length.target == trainer.path_length.target


def test_the_augmentation_probability_survives():
    trainer = Trainer(small())
    trainer.augment.p = 0.37
    restored = Trainer(small())
    restored.load_state_dict(trainer.state_dict())
    assert restored.augment.p == 0.37
