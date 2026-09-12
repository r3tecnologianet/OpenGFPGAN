"""Adaptive discriminator augmentation.

Provenance (see PROVENANCE.md), all of it P4:

- §2: a pipeline of 18 transformations in 6 categories, of which the authors
  "choose to use only pixel blitting, geometric, and color transforms for the
  rest of our tests" — on a 2k set "the vast majority of the benefit came from
  pixel blitting and geometric transforms", while filtering, noise and cutout
  "were not particularly useful". Pixel blitting is x-flips, 90° rotations and
  integer translation; the colour transforms live in `ogan.color_augment`.
  Geometric transforms are **not implemented yet**: P4 executes them through a
  reflect-pad, Symlet-6 wavelet upsample, inverse-mapped bilinear lookup and
  downsample, which is a larger piece than the other two categories together.
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
optimisation.** All three transforms land every output pixel exactly on some
input pixel, so composing them gives one integer index map per image and applying
it needs no interpolation. That is what separates this category from the
geometric one, where P4 has to upsample through a wavelet filter to resample
safely. It is differentiable because a gather is.

Flips and rotations are permutations. **Translation is not**: P4 pads by
reflection and crops, so a shifted image duplicates some pixels near one edge and
discards some at the other. Exact, but not measure-preserving.
"""

import torch

from ogan.color_augment import color_transform

ADA_TARGET = 0.6  # P4 §3
ADA_INTERVAL = 4  # P4 §3: adjust once every four minibatches
ADA_RAMP_IMAGES = 500_000  # P4 §3: 0 -> 1 in this many images


BLIT_TRANSLATE_FRACTION = 0.125  # P4 App. B: t ~ U(-0.125, +0.125)


def _reflect(index: torch.Tensor, size: int) -> torch.Tensor:
    """Fold an out-of-range index back inside, without repeating the edge pixel.

    P4 pads with reflection before applying the transform and crops afterwards,
    which for an integer translation is the same as reading a reflected index.
    """
    period = 2 * (size - 1) if size > 1 else 1
    folded = index.abs() % period
    return torch.minimum(folded, period - folded)


def _blit_indices(n: int, size: int, p: float, device: torch.device) -> torch.Tensor:
    """Compose P4's three blitting transforms into one index map per image.

    Returns `[n, size, size]` flat indices into the spatial plane. Each transform
    is drawn independently per image and taken with probability `p`, per §2 — and
    each then draws its own parameter, which is why the effective rate differs
    per transform: an x-flip taken with `i ~ U{0,1}` mirrors half the time it is
    taken, and a rotation taken with `i ~ U{0,3}` turns three quarters of the
    time.
    """
    rows = torch.arange(size, device=device).reshape(1, size, 1).expand(n, size, size).clone()
    cols = torch.arange(size, device=device).reshape(1, 1, size).expand(n, size, size).clone()

    def taken() -> torch.Tensor:
        return torch.rand(n, 1, 1, device=device) < p

    # x-flip: i ~ U{0,1}, so mirrored on half the images that take it
    mirror = taken() & (torch.rand(n, 1, 1, device=device) < 0.5)
    cols = torch.where(mirror.expand(n, size, size), size - 1 - cols, cols)

    # 90 degree rotations: i ~ U{0,3}, which includes no rotation at all
    turns = torch.where(
        taken(),
        torch.randint(0, 4, (n, 1, 1), device=device),
        torch.zeros(n, 1, 1, dtype=torch.long, device=device),
    ).expand(n, size, size)
    for turn in range(1, 4):
        turned_rows, turned_cols = rows, cols
        for _ in range(turn):
            turned_rows, turned_cols = turned_cols, size - 1 - turned_rows
        selected = turns == turn
        rows = torch.where(selected, turned_rows, rows)
        cols = torch.where(selected, turned_cols, cols)

    # integer translation: t ~ U(-0.125, +0.125) of the side, rounded, reflected
    limit = BLIT_TRANSLATE_FRACTION * size
    for axis in range(2):
        offset = (torch.rand(n, 1, 1, device=device) * 2.0 - 1.0) * limit
        shift = torch.where(taken(), offset.round().long(), torch.zeros_like(offset).long())
        shift = shift.expand(n, size, size)
        if axis == 0:
            rows = _reflect(rows - shift, size)
        else:
            cols = _reflect(cols - shift, size)

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
        """The pipeline, in P4 §2's fixed order.

        Geometric transforms belong between these two and are not implemented
        yet; when they arrive they go there, not at the end, because the order
        is part of what was measured.
        """
        images = pixel_blit(images, self.p)
        return color_transform(images, self.p)

    def state_dict(self) -> dict:
        return {"p": self.p, "signs": list(self._signs)}

    def load_state_dict(self, state: dict) -> None:
        self.p = state["p"]
        self._signs = list(state["signs"])
