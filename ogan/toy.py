"""A distribution whose ground truth we know, for checking that training works.

The unit tests in this project check that each piece matches the paper it came
from. They cannot check that the assembly learns: a flipped sign, an inverted
loss or a gradient that never arrives would pass every one of them. This module
exists to answer that separately, and without a corpus.

The distribution is a white disc of random radius at a random position on black.
Three continuous degrees of freedom, all recoverable from a generated image —
which matters because each failure worth catching leaves its own signature:

| Failure                  | Signature                                        |
|--------------------------|--------------------------------------------------|
| gradient never arrives   | noise: zero components, or hundreds              |
| learned the dataset mean | a central smudge: low disc IoU, no radius spread |
| mode collapse            | centres clustered, radius std near zero          |
| inverted loss or sign    | a black disc on white                            |

**Shape is measured by fitting, not by perimeter.** The obvious metric,
`4 pi A / P^2`, is wrong on a pixel grid: a digitised disc has a crack perimeter
near `8r` rather than `2 pi r`, which scores it 0.617, while a square scores
`pi/4 = 0.785`. The metric would rank a square as rounder than a circle. So the
area and centroid are used to fit the disc the mask *should* be, and the score is
the overlap with it. `tests/test_toy.py` calibrates that on shapes drawn
analytically, because a measurement no one has checked is not evidence.

Images are in [-1, 1], per P3 §A.1.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

RADIUS_RANGE = (0.12, 0.30)  # as a fraction of the side
MARGIN = 0.04  # keep the disc clear of the border, so it is never clipped


@dataclass
class Discs:
    """Images and the parameters they were drawn from."""

    images: torch.Tensor  # [N, 3, R, R] in [-1, 1]
    radii: torch.Tensor  # [N], in pixels
    centres: torch.Tensor  # [N, 2], in pixels, as (y, x)


def draw_discs(n: int, resolution: int, *, device=None, generator=None) -> Discs:
    """Sample `n` discs, white on black, with antialiased edges."""
    device = device or torch.device("cpu")
    rand = lambda *shape: torch.rand(*shape, device=device, generator=generator)  # noqa: E731

    low, high = (r * resolution for r in RADIUS_RANGE)
    radii = rand(n) * (high - low) + low
    limit = resolution * MARGIN + radii
    centres = rand(n, 2) * (resolution - 2 * limit.reshape(n, 1)) + limit.reshape(n, 1)

    grid = torch.arange(resolution, device=device, dtype=torch.float32) + 0.5
    dy = grid.reshape(1, resolution, 1) - centres[:, 0].reshape(n, 1, 1)
    dx = grid.reshape(1, 1, resolution) - centres[:, 1].reshape(n, 1, 1)
    distance = (dy.square() + dx.square()).sqrt()

    # One pixel of antialiasing, so the edge carries gradient rather than a step.
    mask = (radii.reshape(n, 1, 1) - distance).clamp(-1.0, 1.0)
    return Discs(mask.unsqueeze(1).expand(n, 3, resolution, resolution), radii, centres)


def _largest_component(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Label 8-connected regions by propagating the maximum index.

    Returns the mask of the largest region and the number of regions. Converges
    in as many passes as the widest region is across, which at these resolutions
    is cheaper than leaving the device to call out to a labelling library.
    """
    n, _, height, width = mask.shape
    labels = torch.arange(1, height * width + 1, device=mask.device, dtype=torch.float32)
    labels = labels.reshape(1, 1, height, width).expand(n, 1, height, width) * mask

    for _ in range(height + width):
        spread = F.max_pool2d(labels, 3, stride=1, padding=1) * mask
        if torch.equal(spread, labels):
            break
        labels = spread

    counts = torch.tensor(
        [torch.unique(image[image > 0]).numel() for image in labels], device=mask.device
    )
    flat = labels.reshape(n, -1)
    dominant = torch.stack(
        [row[row > 0].mode().values if (row > 0).any() else row.max() for row in flat]
    )
    largest = (labels == dominant.reshape(n, 1, 1, 1)) & (labels > 0)
    return largest.to(mask.dtype), counts


def measure_discs(images: torch.Tensor) -> dict[str, float]:
    """Score a batch of images against the distribution `draw_discs` produces."""
    n, _, height, width = images.shape
    grey = images.mean(dim=1, keepdim=True)
    mask = (grey > 0.0).to(torch.float32)  # midpoint of [-1, 1]

    largest, components = _largest_component(mask)
    area = largest.sum(dim=(1, 2, 3))
    valid = (components == 1) & (area > 4)

    grid = torch.arange(height, device=images.device, dtype=torch.float32) + 0.5
    safe_area = area.clamp(min=1.0)
    centre_y = (largest.squeeze(1) * grid.reshape(1, height, 1)).sum(dim=(1, 2)) / safe_area
    centre_x = (largest.squeeze(1) * grid.reshape(1, 1, width)).sum(dim=(1, 2)) / safe_area
    radius = (safe_area / torch.pi).sqrt()

    # The disc the mask should have been, if it is one.
    dy = grid.reshape(1, height, 1) - centre_y.reshape(n, 1, 1)
    dx = grid.reshape(1, 1, width) - centre_x.reshape(n, 1, 1)
    ideal = ((dy.square() + dx.square()).sqrt() <= radius.reshape(n, 1, 1)).to(torch.float32)
    flat_largest = largest.squeeze(1)
    intersection = (flat_largest * ideal).sum(dim=(1, 2))
    union = ((flat_largest + ideal) > 0).to(torch.float32).sum(dim=(1, 2))
    iou = intersection / union.clamp(min=1.0)

    def over_valid(values: torch.Tensor, default: float = 0.0) -> float:
        return values[valid].mean().item() if valid.any() else default

    return {
        "validity": valid.to(torch.float32).mean().item(),
        "disc_iou": over_valid(iou),
        "radius_mean": over_valid(radius),
        "radius_std": radius[valid].std().item() if valid.sum() > 1 else 0.0,
        "centre_std": (
            torch.stack([centre_y[valid], centre_x[valid]]).std(dim=1).mean().item()
            if valid.sum() > 1
            else 0.0
        ),
        "foreground_fraction": mask.mean().item(),
        "mean_level": grey.mean().item(),
    }
