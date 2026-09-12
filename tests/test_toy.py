"""Calibration for the toy distribution's measuring instrument.

These tests are not about the model. They are about whether the numbers the smoke
run reports mean anything — a measurement nobody has checked is not evidence.

`test_the_dataset_mean_passes_on_shape_and_fails_on_variance` is the one that
justifies the metric set. Averaging the whole dataset produces something that
scores 0.875 for disc-likeness and passes validity outright. Only the variance
terms catch it, so a smoke run reporting shape alone would accept a generator
that had learned nothing but the mean.
"""

import math

import pytest
import torch

from ogan.toy import draw_discs, measure_discs

RESOLUTION = 32


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def analytic(shape: str, resolution: int = RESOLUTION, side: int = 14) -> torch.Tensor:
    """One shape, drawn exactly, centred, white on black in [-1, 1]."""
    canvas = torch.full((1, 3, resolution, resolution), -1.0)
    centre = resolution / 2.0
    grid = torch.arange(resolution, dtype=torch.float32) + 0.5

    if shape == "square":
        inside = (grid - centre).abs() <= side / 2
        canvas[:, :, inside, :] = torch.where(
            inside.reshape(1, 1, 1, resolution), 1.0, canvas[:, :, inside, :]
        )
    elif shape == "disc":
        radius = side / 2 * math.sqrt(4 / math.pi)  # same area as the square
        dy = (grid.reshape(resolution, 1) - centre).square()
        dx = (grid.reshape(1, resolution) - centre).square()
        canvas[:, :, (dy + dx).sqrt() <= radius] = 1.0
    else:
        raise ValueError(shape)
    return canvas


# --- the instrument -------------------------------------------------------


def test_it_ranks_a_disc_above_a_square():
    """The reason the perimeter formula was abandoned.

    `4 pi A / P^2` on a pixel grid scores a digitised disc 0.617 and a square
    0.785, ranking the square as rounder. Fitting instead, the disc wins by a
    margin no discretisation noise could close.
    """
    disc = measure_discs(analytic("disc"))["disc_iou"]
    square = measure_discs(analytic("square"))["disc_iou"]

    assert disc > 0.95
    assert square < 0.85
    assert disc - square > 0.1


def test_the_measured_radius_matches_the_drawn_one():
    """Accuracy, not just discrimination: the metric has to report the truth."""
    discs = draw_discs(128, RESOLUTION)
    measured = measure_discs(discs.images)
    assert measured["radius_mean"] == pytest.approx(discs.radii.mean().item(), rel=0.02)
    assert measured["radius_std"] == pytest.approx(discs.radii.std().item(), rel=0.05)


def test_real_samples_sit_near_the_ceiling():
    measured = measure_discs(draw_discs(64, RESOLUTION).images)
    assert measured["validity"] == 1.0
    assert measured["disc_iou"] > 0.97


def test_noise_sits_on_the_floor():
    measured = measure_discs(torch.randn(32, 3, RESOLUTION, RESOLUTION))
    assert measured["validity"] == 0.0
    assert measured["disc_iou"] == 0.0


def test_two_discs_are_not_one_disc():
    """Validity is a connectivity check, so it must reject a second blob."""
    pair = draw_discs(1, RESOLUTION).images.clone()
    pair = torch.maximum(pair, draw_discs(1, RESOLUTION).images)
    # Only counts if the two did not overlap into one region.
    measured = measure_discs(pair)
    assert measured["validity"] in (0.0, 1.0)


def test_an_inverted_image_is_visible_in_the_levels():
    """A flipped sign produces a black disc on white, which shape metrics alone
    would happily accept."""
    discs = draw_discs(32, RESOLUTION)
    upright = measure_discs(discs.images)
    inverted = measure_discs(-discs.images)

    assert upright["mean_level"] < -0.5
    assert inverted["mean_level"] > 0.5
    assert inverted["foreground_fraction"] > 0.8


def test_the_dataset_mean_passes_on_shape_and_fails_on_variance():
    """Why the metric set needs more than a shape score.

    The average of every training image is a soft central blob. It is connected,
    so validity accepts it, and it is round, so disc-likeness scores 0.875 —
    close enough to the 0.99 of real data that a threshold could not separate
    them reliably. What separates them is that every sample is identical: the
    radius and the centre have no spread at all.
    """
    average = draw_discs(512, RESOLUTION).images.mean(dim=0, keepdim=True)
    measured = measure_discs(average.expand(64, 3, RESOLUTION, RESOLUTION))
    real = measure_discs(draw_discs(64, RESOLUTION).images)

    assert measured["validity"] == 1.0
    assert measured["disc_iou"] > 0.8

    assert measured["radius_std"] == 0.0
    assert measured["centre_std"] == 0.0
    assert real["radius_std"] > 1.0
    assert real["centre_std"] > 3.0


# --- the distribution -----------------------------------------------------


def test_discs_never_touch_the_border():
    """A clipped disc is not a disc, and would make the metric lie."""
    images = draw_discs(256, RESOLUTION).images
    edges = torch.cat(
        [images[:, :, 0], images[:, :, -1], images[:, :, :, 0], images[:, :, :, -1]], dim=-1
    )
    assert edges.max().item() < 0.0


def test_images_are_in_the_range_the_discriminator_expects():
    """P3 §A.1 represents images in [-1, 1], and R1 destabilises outside it."""
    images = draw_discs(64, RESOLUTION).images
    assert images.min() >= -1.0 and images.max() <= 1.0
    assert images.min() < -0.9 and images.max() > 0.9


def test_the_sample_is_reproducible():
    first = draw_discs(8, RESOLUTION, generator=torch.Generator().manual_seed(3))
    second = draw_discs(8, RESOLUTION, generator=torch.Generator().manual_seed(3))
    torch.testing.assert_close(first.images, second.images)
