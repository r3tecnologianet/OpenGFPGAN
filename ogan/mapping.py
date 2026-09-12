"""The mapping network, f: Z -> W.

Provenance (see PROVENANCE.md):

- P1 App. B: kept unchanged from StyleGAN — "the dimensionality of Z and W (512),
  mapping network architecture (8 fully connected layers, 100× lower learning
  rate)", with LeakyReLU α = 0.2 throughout.
- P2 Fig. 1: a `Normalize` block sits between the latent and the first fully
  connected layer of the mapping network.
- P3 §4.2 gives the formula, `b = a / sqrt(mean(a²) + ε)` with ε = 1e-8, and
  P3 §A.1 puts latents on "a 512-dimensional hypersphere", which is what applying
  it to a 512-vector produces.

**One thing in the sources is genuinely ambiguous and is not resolved here.** P2's
appendix states "We do not use batch normalization, spectral normalization,
attention mechanisms, dropout, or pixelwise feature vector normalization in our
networks" — in a paragraph otherwise about the classifier networks trained for
the separability metric. Read as a claim about the generator it would contradict
Fig. 1, which draws the `Normalize` block.

The reading taken is that the two are different operations: normalising the input
latent once, which Fig. 1 shows, versus applying P3's normalisation after every
convolution inside the generator, which is what ProGAN did and what StyleGAN
replaced with AdaIN. Only the first is implemented here, and nothing in this
package normalises inside the synthesis path. If that reading is wrong, the
consequence is confined to this file.
"""

import torch
from torch import nn

from ogan.layers import EqualizedLinear, leaky_relu

Z_DIM = 512
W_DIM = 512
MAPPING_LAYERS = 8
MAPPING_LR_MULTIPLIER = 0.01  # P1 App. B: "100x lower learning rate"
NORMALIZE_EPS = 1e-8  # P3 §4.2


def normalize_latent(z: torch.Tensor, eps: float = NORMALIZE_EPS) -> torch.Tensor:
    """Project a latent onto the hypersphere of radius sqrt(d) (P3 §4.2)."""
    return z * z.square().mean(dim=-1, keepdim=True).add(eps).rsqrt()


class MappingNetwork(nn.Module):
    """Eight fully connected layers that learn 100x more slowly than the rest."""

    def __init__(
        self,
        z_dim: int = Z_DIM,
        w_dim: int = W_DIM,
        *,
        num_layers: int = MAPPING_LAYERS,
        lr_multiplier: float = MAPPING_LR_MULTIPLIER,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        widths = [z_dim] + [w_dim] * num_layers
        self.layers = nn.ModuleList(
            EqualizedLinear(widths[i], widths[i + 1], lr_multiplier=lr_multiplier)
            for i in range(num_layers)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.shape[-1] != self.z_dim:
            raise ValueError(f"latent must have {self.z_dim} components, got {z.shape[-1]}")
        x = normalize_latent(z)
        for layer in self.layers:
            x = leaky_relu(layer(x))
        return x

    def extra_repr(self) -> str:
        return f"{self.z_dim} -> {self.w_dim}, layers={len(self.layers)}"
