"""Adaptive discriminator augmentation.

Provenance (see PROVENANCE.md), all of it P4:

- §2: a pipeline of 18 transformations in 6 categories, of which the authors
  "choose to use only pixel blitting, geometric, and color transforms for the
  rest of our tests" — on a 2k set "the vast majority of the benefit came from
  pixel blitting and geometric transforms", while filtering, noise and cutout
  "were not particularly useful". Pixel blitting is x-flips, 90° rotations and
  integer translation.
- §2: each transformation is applied "with probability p or skipped with
  probability 1 - p", with "the same value of p for all transformations",
  randomised "separately for each augmentation and for each image in a
  minibatch", in a fixed order.
- §2: "we execute augmentations also when training the generator, which requires
  the augmentations to be differentiable".
- §3 Eq. 1: `r_t = E[sign(D_train)]`, the portion of the training set receiving
  positive discriminator outputs, averaged over N = 4 consecutive minibatches.
- §3: p starts at zero, is adjusted every four minibatches by a fixed step sized
  so it "can rise from 0 to 1 sufficiently quickly, e.g., in 500k images", and is
  clamped from below to zero after every step. The target is 0.6.

**Pixel blitting is implemented as a single gather, and that is not an
optimisation.** All three transforms are permutations of the pixel grid, so
composing them gives one index map per image and applying it moves every pixel
exactly once. No interpolation, no resampling, no energy lost at the edges —
which is what separates this category from the geometric one, and what makes it
safe to apply at high probability. It is differentiable because a gather is.
"""

import torch

ADA_TARGET = 0.6  # P4 §3
ADA_INTERVAL = 4  # P4 §3: adjust once every four minibatches
ADA_RAMP_IMAGES = 500_000  # P4 §3: 0 -> 1 in this many images


def _blit_indices(n: int, size: int, p: float, device: torch.device) -> torch.Tensor:
    """Compose x-flip, 90° rotation and integer translation into one index map.

    Returns `[n, size, size]` flat indices into the spatial plane. Each transform
    is drawn independently per image and taken with probability `p`, per §2.
    """
    rows = torch.arange(size, device=device).reshape(1, size, 1).expand(n, size, size)
    cols = torch.arange(size, device=device).reshape(1, 1, size).expand(n, size, size)
    rows, cols = rows.clone(), cols.clone()

    def taken() -> torch.Tensor:
        return (torch.rand(n, 1, 1, device=device) < p).expand(n, size, size)

    # x-flip
    rows, cols = rows, torch.where(taken(), size - 1 - cols, cols)

    # 90 degree rotations: (r, c) -> (c, size-1-r), applied k times
    k = torch.where(
        (torch.rand(n, 1, 1, device=device) < p),
        torch.randint(1, 4, (n, 1, 1), device=device),
        torch.zeros(n, 1, 1, dtype=torch.long, device=device),
    ).expand(n, size, size)
    for turn in range(1, 4):
        rotate = k == turn
        turned_rows, turned_cols = rows, cols
        for _ in range(turn):
            turned_rows, turned_cols = turned_cols, size - 1 - turned_rows
        rows = torch.where(rotate, turned_rows, rows)
        cols = torch.where(rotate, turned_cols, cols)

    # integer translation, wrapping
    for axis in (0, 1):
        shift = torch.where(
            torch.rand(n, 1, 1, device=device) < p,
            torch.randint(0, size, (n, 1, 1), device=device),
            torch.zeros(n, 1, 1, dtype=torch.long, device=device),
        ).expand(n, size, size)
        if axis == 0:
            rows = (rows + shift) % size
        else:
            cols = (cols + shift) % size

    return rows * size + cols


def pixel_blit(images: torch.Tensor, p: float) -> torch.Tensor:
    """P4's pixel blitting category, as one gather per minibatch."""
    if p <= 0.0:
        return images
    n, c, height, width = images.shape
    if height != width:
        raise ValueError(f"pixel blitting expects square images, got {height}x{width}")

    flat = _blit_indices(n, height, p, images.device).reshape(n, 1, height * width)
    gathered = images.reshape(n, c, -1).gather(2, flat.expand(n, c, height * width))
    return gathered.reshape(images.shape)


class AdaptiveAugment:
    """P4 §3's controller: measure overfitting, move `p` against it.

    `r_t` is the fraction of real images the discriminator scores positively, so
    0.5 is a discriminator that cannot tell and 1.0 one that has memorised the
    training set. The target of 0.6 leaves it slightly ahead, which is where a
    GAN's feedback is still informative.

    Held separately from the augmentation itself because it is the part with
    state: the augmentation is a pure function of `p`.
    """

    def __init__(
        self,
        *,
        target: float = ADA_TARGET,
        interval: int = ADA_INTERVAL,
        ramp_images: int = ADA_RAMP_IMAGES,
        batch_size: int = 32,
    ) -> None:
        self.target = target
        self.interval = interval
        self.batch_size = batch_size
        self.p = 0.0
        self.step = interval * batch_size / ramp_images
        self._signs: list[float] = []

    def observe(self, real_scores: torch.Tensor) -> None:
        """Accumulate one minibatch of discriminator outputs on real images."""
        self._signs.append(real_scores.detach().sign().mean().item())

    @property
    def ready(self) -> bool:
        return len(self._signs) >= self.interval

    @property
    def overfitting(self) -> float:
        """`r_t` over the accumulated window, or 0 before any observation."""
        return sum(self._signs) / len(self._signs) if self._signs else 0.0

    def adjust(self) -> float:
        """Move `p` one step against the measured overfitting, and reset the window."""
        if not self._signs:
            return self.p
        direction = 1.0 if self.overfitting > self.target else -1.0
        self.p = min(1.0, max(0.0, self.p + direction * self.step))
        self._signs.clear()
        return self.p

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return pixel_blit(images, self.p)

    def state_dict(self) -> dict:
        return {"p": self.p, "signs": list(self._signs)}

    def load_state_dict(self, state: dict) -> None:
        self.p = state["p"]
        self._signs = list(state["signs"])
