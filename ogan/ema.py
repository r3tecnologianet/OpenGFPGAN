"""The averaged generator — the model that actually gets published.

Provenance (see PROVENANCE.md):

- P3 §A.1: "for visualizing generator output at any given point during the
  training, we use an exponential running average for the weights of the
  generator with decay 0.999". P1 App. B keeps it, citing P3.
- P4 states the same mechanism as "exponential moving average half-life of 20k
  images for generator weights".

**The two forms agree, and converting between them is where the batch size
enters.** A half-life measured in images is a statement about how much data the
average forgets, independent of how that data is batched; a per-step decay is
not. With `B` images per step and a half-life of `H` images, the average must
halve over `H/B` steps, so `decay = 0.5 ** (B/H)`. At P1's minibatch of 32,
P4's 20k half-life gives 0.99889 — against P3's 0.999, which is itself a half
life of 22.2k images at that batch size. Two papers, two notations, a 10%
difference. `tests/test_ema.py` pins the agreement.

The half-life form is used here, because it says what it means when the batch
size changes and the decay form silently does not.
"""

import copy

import torch
from torch import nn

EMA_HALF_LIFE_IMAGES = 20_000  # P4
EMA_DECAY_PROGAN = 0.999  # P3 §A.1, at P1's minibatch of 32


def decay_from_half_life(half_life_images: float, batch_size: int) -> float:
    """`0.5 ** (batch / half_life)` — halve the average over `half_life` images."""
    return 0.5 ** (batch_size / half_life_images)


def half_life_from_decay(decay: float, batch_size: int) -> float:
    """The inverse, for reading a paper that states a per-step decay."""
    import math

    return batch_size * math.log(0.5) / math.log(decay)


class AveragedGenerator(nn.Module):
    """A frozen copy of a generator, tracking it by exponential moving average.

    This is the artifact, not a diagnostic: P3 introduces the average for
    visualising output during training, and what a published checkpoint contains
    is the averaged weights rather than the live ones. It follows that the
    average belongs in every checkpoint — restoring the live generator and
    recomputing the average from it would silently restart the average.

    Buffers are copied rather than averaged. A buffer holds state that is
    counted or measured rather than learned, and interpolating one has no
    meaning.
    """

    def __init__(
        self,
        source: nn.Module,
        *,
        half_life_images: float = EMA_HALF_LIFE_IMAGES,
        batch_size: int = 32,
    ) -> None:
        super().__init__()
        self.half_life_images = half_life_images
        self.batch_size = batch_size
        self.decay = decay_from_half_life(half_life_images, batch_size)
        self.module = copy.deepcopy(source).requires_grad_(False).eval()

    @torch.no_grad()
    def update(self, source: nn.Module) -> None:
        for averaged, live in zip(self.module.parameters(), source.parameters(), strict=True):
            averaged.lerp_(live.detach(), 1.0 - self.decay)
        for averaged, live in zip(self.module.buffers(), source.buffers(), strict=True):
            averaged.copy_(live)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def extra_repr(self) -> str:
        return (
            f"half_life_images={self.half_life_images:g}, batch_size={self.batch_size}, "
            f"decay={self.decay:.6f}"
        )
