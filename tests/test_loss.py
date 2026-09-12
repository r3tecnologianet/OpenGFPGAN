"""Tests for the losses and regularizers.

Two carry real weight. `test_the_non_saturating_form_keeps_its_gradient` measures
the property the loss is named for, rather than trusting the name. And
`test_the_jacobian_vector_identity_holds` checks P1 §3.2's identity against an
explicitly assembled Jacobian, which is what makes the cheap path trustworthy.
"""

import math

import pytest
import torch

from ogan.loss import (
    PL_DECAY,
    PL_INTERVAL,
    R1_GAMMA_FFHQ_1024,
    R1_INTERVAL,
    PathLengthPenalty,
    discriminator_loss,
    generator_loss,
    lazy_adam_hyperparameters,
    path_length_weight,
    r1_penalty,
    saturating_generator_loss,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# --- the logistic losses, derived from their names ------------------------


def test_the_non_saturating_form_keeps_its_gradient():
    """The whole reason for the name, and the reason StyleGAN2 uses this form.

    When the discriminator confidently rejects a sample — exactly when the
    generator most needs a signal — the saturating objective's gradient collapses
    towards zero, while the non-saturating one stays near one.
    """
    confident_rejection = torch.tensor([-10.0], requires_grad=True)

    non_saturating = (
        torch.autograd.grad(generator_loss(confident_rejection), confident_rejection)[0]
        .abs()
        .item()
    )
    saturating = (
        torch.autograd.grad(saturating_generator_loss(confident_rejection), confident_rejection)[0]
        .abs()
        .item()
    )

    assert non_saturating == pytest.approx(1.0, abs=1e-3)
    assert saturating < 1e-4
    assert non_saturating / saturating > 1000


def test_the_two_forms_agree_when_the_discriminator_is_undecided():
    """At a logit of zero both are 1/2, which is the sanity check on the algebra."""
    undecided = torch.zeros(1, requires_grad=True)
    for loss in (generator_loss, saturating_generator_loss):
        grad = torch.autograd.grad(loss(undecided), undecided)[0].abs().item()
        assert grad == pytest.approx(0.5, abs=1e-6)


def test_the_discriminator_loss_rewards_separation():
    separated = discriminator_loss(torch.full((8,), 5.0), torch.full((8,), -5.0))
    confused = discriminator_loss(torch.zeros(8), torch.zeros(8))
    inverted = discriminator_loss(torch.full((8,), -5.0), torch.full((8,), 5.0))
    assert separated < confused < inverted


def test_the_discriminator_loss_at_chance_is_two_log_two():
    """Both terms are softplus(0) = ln 2 when the discriminator cannot tell."""
    at_chance = discriminator_loss(torch.zeros(64), torch.zeros(64)).item()
    assert at_chance == pytest.approx(2.0 * math.log(2.0), abs=1e-6)


# --- R1 -------------------------------------------------------------------


def test_r1_is_zero_for_a_discriminator_with_no_slope():
    images = torch.randn(4, 3, 8, 8, requires_grad=True)
    scores = (images * 0.0).sum(dim=(1, 2, 3))
    assert r1_penalty(scores, images, gamma=10.0).item() == pytest.approx(0.0, abs=1e-9)


def test_r1_matches_its_definition():
    """gamma/2 times the mean squared gradient norm, checked on a linear score."""
    images = torch.randn(4, 3, 8, 8, requires_grad=True)
    direction = torch.randn(3, 8, 8)
    scores = (images * direction).sum(dim=(1, 2, 3))

    expected = 0.5 * 10.0 * direction.square().sum().item()
    assert r1_penalty(scores, images, gamma=10.0).item() == pytest.approx(expected, rel=1e-5)


def test_r1_scales_linearly_with_gamma():
    images = torch.randn(4, 3, 8, 8, requires_grad=True)
    scores = (images * torch.randn(3, 8, 8)).sum(dim=(1, 2, 3))
    one = r1_penalty(scores, images, gamma=1.0).item()
    ten = r1_penalty(scores, images, gamma=10.0).item()
    assert ten / one == pytest.approx(10.0, rel=1e-5)


def test_the_papers_gamma():
    assert R1_GAMMA_FFHQ_1024 == 10.0


# --- path length (P1 Eq. 4, Eq. 5, §3.2) ---------------------------------


def test_the_jacobian_vector_identity_holds():
    """P1 §3.2: `J_w^T y = grad_w (g(w) . y)`, avoiding the Jacobian itself.

    Checked against a Jacobian assembled explicitly, which is the only way to
    know the cheap path computes what Eq. 4 asks for.
    """
    matrix = torch.randn(12, 5)

    def g(w):
        return torch.tanh(matrix @ w)

    w = torch.randn(5, requires_grad=True)
    y = torch.randn(12)

    cheap = torch.autograd.grad((g(w) * y).sum(), w)[0]
    jacobian = torch.autograd.functional.jacobian(g, w)
    explicit = jacobian.T @ y

    torch.testing.assert_close(cheap, explicit, atol=1e-6, rtol=0)


def test_the_weight_matches_equation_5():
    """And it is 1.06e-7 at 1024, not the 5.37e-7 the audited document tabulates."""
    assert path_length_weight(1024) == pytest.approx(1.0596e-7, rel=1e-3)
    assert path_length_weight(512) == pytest.approx(4.7684e-7, rel=1e-3)
    assert path_length_weight(1024) != pytest.approx(5.37e-7, rel=0.01)


def test_the_target_starts_at_zero_and_tracks_the_lengths():
    """App. B initialises the target to zero and moves it by an EMA, so the
    regularizer asks for consistency rather than for a chosen scale."""
    penalty = PathLengthPenalty()
    assert penalty.target == 0.0

    ws = torch.randn(8, 4, 16, requires_grad=True)
    images = (ws.sum(dim=1) @ torch.randn(16, 3 * 64)).reshape(8, 3, 8, 8)
    for _ in range(200):
        penalty(images, ws)
    assert penalty.target > 0.0

    settled = penalty.target
    penalty(images, ws)
    assert penalty.target == pytest.approx(settled, rel=0.05)


def test_the_penalty_falls_as_the_target_settles():
    penalty = PathLengthPenalty()
    ws = torch.randn(8, 4, 16, requires_grad=True)
    images = (ws.sum(dim=1) @ torch.randn(16, 3 * 64)).reshape(8, 3, 8, 8)

    first = penalty(images, ws).item()
    for _ in range(300):
        last = penalty(images, ws).item()
    assert last < first


def test_the_decay_is_the_papers():
    assert PL_DECAY == 0.99


# --- lazy regularization (P1 App. B) -------------------------------------


def test_the_intervals_are_the_papers():
    """ "We use k = 16 for the discriminator and k = 8 for the generator"."""
    assert (R1_INTERVAL, PL_INTERVAL) == (16, 8)


def test_the_optimizer_compensation_matches_the_paper():
    """`c = k/(k+1)`, `lr' = c*lr`, `beta' = beta^c`."""
    lr, betas = lazy_adam_hyperparameters(2e-3, (0.0, 0.99), R1_INTERVAL)
    assert lr == pytest.approx(16 / 17 * 2e-3)
    assert betas[1] == pytest.approx(0.99 ** (16 / 17))

    lr, betas = lazy_adam_hyperparameters(2e-3, (0.0, 0.99), PL_INTERVAL)
    assert lr == pytest.approx(8 / 9 * 2e-3)
    assert betas[1] == pytest.approx(0.99 ** (8 / 9))


def test_a_zero_first_moment_survives_the_compensation():
    """beta1 is 0 in P1 App. B, and 0**c is 0 for any positive c — worth pinning,
    since a naive compensation that raised 1-beta instead would not be."""
    _, betas = lazy_adam_hyperparameters(2e-3, (0.0, 0.99), R1_INTERVAL)
    assert betas[0] == 0.0


def test_a_longer_interval_compensates_less():
    for interval in (1, 4, 16, 64):
        lr, _ = lazy_adam_hyperparameters(1.0, (0.0, 0.99), interval)
        assert lr == pytest.approx(interval / (interval + 1))
