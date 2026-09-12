"""Mapping and synthesis, composed, with style mixing.

Provenance (see PROVENANCE.md):

- P2 §3.1 and Table 2: mixing regularization runs two latents through the mapping
  network and switches from one to the other "at a randomly selected point in the
  synthesis network", on 90% of training images. Table 2 puts 90% ahead of both
  50% and 100%, so it is a chosen value rather than a ceiling.
- P1 App. B keeps it unchanged.

Mixing is off by default. It is a training regularizer, and sampling from a
finished model should not silently do it.
"""

import torch
from torch import nn

from ogan.mapping import W_DIM, Z_DIM, MappingNetwork
from ogan.synthesis import CHANNEL_MAX, SynthesisNetwork

STYLE_MIXING_PROB = 0.9  # P2 Table 2


class Generator(nn.Module):
    def __init__(
        self,
        resolution: int = 512,
        z_dim: int = Z_DIM,
        w_dim: int = W_DIM,
        *,
        large: bool = False,
        channel_max: int = CHANNEL_MAX,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.mapping = MappingNetwork(z_dim, w_dim)
        self.synthesis = SynthesisNetwork(resolution, w_dim, large=large, channel_max=channel_max)

    @property
    def num_ws(self) -> int:
        return self.synthesis.num_ws

    def styles(self, z: torch.Tensor, *, style_mixing_prob: float = 0.0) -> torch.Tensor:
        """`[N, z_dim]` latents to `[N, num_ws, w_dim]` styles."""
        w = self.mapping(z).unsqueeze(1).repeat(1, self.num_ws, 1)
        if style_mixing_prob <= 0.0:
            return w

        other = self.mapping(torch.randn_like(z)).unsqueeze(1)
        crossover = torch.randint(1, self.num_ws, (z.shape[0], 1, 1), device=z.device)
        mixing = torch.rand(z.shape[0], 1, 1, device=z.device) < style_mixing_prob
        layer = torch.arange(self.num_ws, device=z.device).reshape(1, -1, 1)
        return torch.where(mixing & (layer >= crossover), other, w)

    def forward(self, z: torch.Tensor, *, style_mixing_prob: float = 0.0) -> torch.Tensor:
        return self.synthesis(self.styles(z, style_mixing_prob=style_mixing_prob))
