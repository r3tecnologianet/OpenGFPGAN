"""Equalized learning rate, and the activation gain that pairs with it.

Provenance (see PROVENANCE.md):

- P3 §4.1 — the mechanism: initialise weights `N(0, 1)` and scale them at runtime
  by a per-layer constant, rather than scaling at initialisation. The point is
  that Adam normalises each update by its own gradient statistics, so a layer
  whose weights have an unusual dynamic range effectively gets an unusual
  learning rate.
- P3 §A.1 — biases initialise to zero, LeakyReLU has slope 0.2.
- The scale itself is `derived`. P3 §4.1 writes it as `ŵ = w/c` with `c` "the
  per-layer normalization constant from He's initializer", which read literally
  amplifies wide-fan-in layers — the opposite of the goal stated in the same
  paragraph. So we derive it from the intent instead, below.

**The split between layer and activation is deliberate.** The layer applies
`1/sqrt(fan_in)`, which is what keeps a linear map's output second moment at one.
The activation applies `ACTIVATION_GAIN`, which is what restores the second
moment that LeakyReLU removes. These are two different jobs, and folding both
into the weight scale — as is commonly done, since every convolution here is
followed by the same activation — makes it easy to apply the gain twice by
accident. Twice squares it to ≈1.92 per layer, and training diverges.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

#: P3 §A.1, restated in P1 App. B.
LRELU_SLOPE = 0.2

#: Derived. For `x ~ N(0, 1)` and `y = leaky_relu(x, a)`, splitting the integral
#: at zero gives `E[y²] = (1 + a²)/2`, so multiplying by `sqrt(2/(1+a²))` returns
#: the second moment to one. For a = 0.2 this is ≈1.38675049.
#:
#: Note `sqrt(2) ≈ 1.41421` is the same constant for plain ReLU. These networks
#: are LeakyReLU throughout, so the slope-aware value is the correct one.
ACTIVATION_GAIN = math.sqrt(2.0 / (1.0 + LRELU_SLOPE**2))


def leaky_relu(x: torch.Tensor) -> torch.Tensor:
    """LeakyReLU, gained so that the output's second moment stays at one.

    It is the *second moment* that is preserved, not the standard deviation.
    LeakyReLU leaves a positive mean, so for unit-normal input the output has
    RMS ≈ 1.0 and standard deviation ≈ 0.897. The second moment is the quantity
    that matters: for a following layer with zero-mean weights,
    `Var(Σ w·y) = Σ Var(w)·E[y²]`, so `E[y²]` is what propagates.
    """
    return F.leaky_relu(x, LRELU_SLOPE) * ACTIVATION_GAIN


class EqualizedLinear(nn.Module):
    """A linear layer with the runtime weight scale of P3 §4.1.

    Weights are stored as `N(0, 1)` and scaled by `1/sqrt(fan_in)` on every
    forward pass, so unit-second-moment input gives unit-second-moment output.
    No activation is applied; pair it with `leaky_relu`.
    """

    def __init__(self, in_features: int, out_features: int, *, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.scale = in_features**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.scale, self.bias)

    def extra_repr(self) -> str:
        return (
            f"{self.in_features} -> {self.out_features}, "
            f"bias={self.bias is not None}, scale={self.scale:.6g}"
        )


class EqualizedConv2d(nn.Module):
    """A 2D convolution with the runtime weight scale of P3 §4.1.

    `fan_in` is `in_channels * kernel_size²`, the number of inputs each output
    element sums over. Same contract as `EqualizedLinear`: no activation here.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.scale = (in_channels * kernel_size * kernel_size) ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x, self.weight * self.scale, self.bias, stride=self.stride, padding=self.padding
        )

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels} -> {self.out_channels}, k={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}, "
            f"bias={self.bias is not None}, scale={self.scale:.6g}"
        )
