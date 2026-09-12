"""Tests for adaptive discriminator augmentation (P4 §2, §3).

Two carry weight. `test_blitting_is_a_lossless_permutation` checks the property
that separates this category from the geometric one and makes it safe at high
probability. And `test_the_ramp_reaches_one_in_the_papers_image_count` checks the
step size against the sentence that specifies it, rather than against itself.
"""

import pytest
import torch

from ogan.augment import (
    ADA_INTERVAL,
    ADA_RAMP_IMAGES,
    ADA_TARGET,
    AdaptiveAugment,
    pixel_blit,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# --- the constants P4 states ----------------------------------------------


def test_the_papers_constants():
    assert ADA_TARGET == 0.6
    assert ADA_INTERVAL == 4
    assert ADA_RAMP_IMAGES == 500_000


# --- pixel blitting (P4 §2) -----------------------------------------------


def test_zero_probability_is_the_identity():
    images = torch.randn(4, 3, 8, 8)
    torch.testing.assert_close(pixel_blit(images, 0.0), images)


def test_blitting_is_a_lossless_permutation():
    """What makes this category different from the geometric one.

    x-flips, 90° rotations and integer translation are all permutations of the
    pixel grid, so every pixel survives exactly once. No interpolation, no energy
    lost at an edge — which is why it can be applied at high probability without
    the augmentation itself becoming the thing the discriminator learns.
    """
    images = torch.randn(8, 3, 16, 16)
    blitted = pixel_blit(images, 1.0)
    for original, transformed in zip(images, blitted, strict=True):
        torch.testing.assert_close(
            original.flatten().sort().values, transformed.flatten().sort().values
        )


def test_randomisation_is_per_image():
    """P4 §2: "separately for each augmentation and for each image in a minibatch"."""
    same = torch.randn(1, 3, 16, 16).expand(16, -1, -1, -1).contiguous()
    blitted = pixel_blit(same, 1.0)
    distinct = {tuple(image.flatten()[:8].tolist()) for image in blitted}
    assert len(distinct) > 1


def test_it_is_differentiable():
    """P4 §2: augmentations run when training the generator too, "which requires
    the augmentations to be differentiable"."""
    images = torch.randn(4, 3, 8, 8, requires_grad=True)
    pixel_blit(images, 1.0).square().sum().backward()
    assert images.grad is not None
    assert torch.isfinite(images.grad).all()
    assert images.grad.abs().sum() > 0


def test_the_gradient_is_a_permutation_too():
    """A gather moves gradient the way it moved pixels: every input gets exactly
    the gradient of the one output it became."""
    images = torch.randn(4, 3, 8, 8, requires_grad=True)
    (pixel_blit(images, 1.0) * 1.0).sum().backward()
    torch.testing.assert_close(images.grad, torch.ones_like(images))


def test_shape_and_dtype_survive():
    for dtype in (torch.float32, torch.bfloat16):
        out = pixel_blit(torch.randn(2, 3, 8, 8, dtype=dtype), 1.0)
        assert out.shape == (2, 3, 8, 8)
        assert out.dtype == dtype


def test_non_square_images_are_rejected():
    with pytest.raises(ValueError, match="square"):
        pixel_blit(torch.randn(2, 3, 8, 16), 1.0)


# --- the controller (P4 §3) -----------------------------------------------


def test_p_starts_at_zero():
    assert AdaptiveAugment().p == 0.0


def test_the_heuristic_is_the_mean_sign():
    """Eq. 1: `r_t = E[sign(D_train)]` — the portion scored positively."""
    control = AdaptiveAugment()
    control.observe(torch.tensor([1.0, 1.0, 1.0, -1.0]))
    assert control.overfitting == pytest.approx(0.5)

    control = AdaptiveAugment()
    control.observe(torch.tensor([5.0, 2.0, 0.1]))
    assert control.overfitting == pytest.approx(1.0)


def test_overfitting_raises_p_and_the_reverse_lowers_it():
    control = AdaptiveAugment()
    control.observe(torch.ones(8))  # r_t = 1.0, above the target
    assert control.adjust() > 0.0

    control.observe(torch.ones(8))
    control.adjust()
    raised = control.p
    control.observe(-torch.ones(8))  # r_t = -1.0, below the target
    assert control.adjust() < raised


def test_p_never_goes_below_zero():
    """P4 §3: "After every step we clamp p from below to 0"."""
    control = AdaptiveAugment()
    for _ in range(50):
        control.observe(-torch.ones(8))
        control.adjust()
    assert control.p == 0.0


def test_p_never_goes_above_one():
    """From the definition of p as a probability in [0, 1]."""
    control = AdaptiveAugment(ramp_images=1000, batch_size=32)
    for _ in range(200):
        control.observe(torch.ones(8))
        control.adjust()
    assert control.p == 1.0


def test_the_ramp_reaches_one_in_the_papers_image_count():
    """P4 §3 sizes the step so p "can rise from 0 to 1 ... in 500k images".

    Checked against the sentence, not against the formula: run the controller
    under permanent overfitting and count the images it consumed getting there.
    """
    batch = 32
    control = AdaptiveAugment(batch_size=batch)

    images_seen = 0
    while control.p < 1.0:
        for _ in range(control.interval):
            control.observe(torch.ones(batch))
            images_seen += batch
        control.adjust()

    assert images_seen == pytest.approx(ADA_RAMP_IMAGES, rel=0.01)


def test_the_window_is_four_minibatches():
    """P4 §3 averages over N = 4 consecutive minibatches before adjusting."""
    control = AdaptiveAugment()
    for _ in range(ADA_INTERVAL - 1):
        control.observe(torch.ones(4))
        assert not control.ready
    control.observe(torch.ones(4))
    assert control.ready


def test_adjusting_clears_the_window():
    control = AdaptiveAugment()
    control.observe(torch.ones(4))
    control.adjust()
    assert control.overfitting == 0.0


def test_it_augments_with_its_own_p():
    control = AdaptiveAugment()
    images = torch.randn(4, 3, 8, 8)
    torch.testing.assert_close(control(images), images)  # p is still zero

    control.p = 1.0
    assert not torch.equal(control(images), images)


def test_the_pipeline_applies_both_implemented_categories():
    """Blitting permutes pixels and colour maps values, so a pipeline running
    both changes the sorted pixel values — which blitting alone cannot do."""
    control = AdaptiveAugment()
    control.p = 1.0
    images = torch.rand(8, 3, 8, 8)
    out = control(images)

    assert not torch.allclose(out, images)
    assert not torch.allclose(
        out.flatten(1).sort(dim=1).values, images.flatten(1).sort(dim=1).values
    )


def test_the_controller_round_trips_through_a_checkpoint():
    """`p` is training state: resuming without it restarts the schedule."""
    control = AdaptiveAugment()
    control.observe(torch.ones(8))
    control.adjust()
    control.observe(torch.ones(8))

    restored = AdaptiveAugment()
    restored.load_state_dict(control.state_dict())
    assert restored.p == control.p
    assert restored.overfitting == control.overfitting
