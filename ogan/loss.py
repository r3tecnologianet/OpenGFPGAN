"""Losses and regularizers.

Provenance (see PROVENANCE.md):

- P1 App. B lists the objective as "non-saturating logistic loss [16] with R1
  regularization [30]" and **does not write either formula**, nor does P2. Both
  are therefore `derived` here, from the names plus the fact that the
  discriminator emits a logit. The papers' own citations are Goodfellow et al.
  2014 (arXiv 1406.2661) for the first and Mescheder et al. 2018 (arXiv
  1801.04406) for the second; neither has been read, and neither needs to be for
  a derivation this short, but they are where to go if it is ever disputed.
- P1 Eq. 4 and §3.2 give the path length objective and the Jacobian-vector
  identity outright.
- P1 Eq. 5 gives its weight; App. B gives the EMA decay, the zero initialisation
  of the target, and the averaging over synthesis layers.
- P1 App. B, *Lazy regularization*: intervals of 16 and 8, the regularization
  term multiplied by k, and the optimizer compensation.
- P4 appendix gives the R1 weight's scaling law, `gamma_0 = 0.0002 * N/M`, with
  a recommended sweep over `[gamma_0/5, gamma_0*5]`. P1 states only a value it
  tuned per dataset, which is not a number this project can inherit: it was
  fitted to a distribution this project will not train on.

**The logistic losses, derived.** The discriminator emits a logit `s`, so its
probability is `sigmoid(s)`, and `-log sigmoid(s) = softplus(-s)` while
`-log(1 - sigmoid(s)) = softplus(s)`. The discriminator minimises
`softplus(-s_real) + softplus(s_fake)`. The generator's *saturating* objective
would be to minimise `log(1 - sigmoid(s_fake))`, whose gradient vanishes exactly
when the discriminator is confident and the generator most needs one; the
non-saturating form instead maximises `log sigmoid(s_fake)`, giving
`softplus(-s_fake)`. `tests/test_loss.py` measures that difference rather than
asserting the name.
"""

import math

import torch
import torch.nn.functional as F

R1_INTERVAL = 16  # P1 App. B: k for the discriminator
PL_INTERVAL = 8  # P1 App. B: k for the generator
PL_DECAY = 0.99  # P1 App. B: beta_pl
R1_GAMMA_COEFFICIENT = 0.0002  # P4 appendix
R1_GAMMA_SWEEP = (0.2, 5.0)  # P4 appendix: multiply gamma_0 by these to bracket a sweep


def discriminator_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    return F.softplus(-real_scores).mean() + F.softplus(fake_scores).mean()


def generator_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """The non-saturating form: maximise `log sigmoid(s)`, not minimise `log(1-sigmoid(s))`."""
    return F.softplus(-fake_scores).mean()


def saturating_generator_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """The form StyleGAN2 does *not* use. Here so the difference can be measured."""
    return -F.softplus(fake_scores).mean()


def r1_penalty(real_scores: torch.Tensor, real_images: torch.Tensor, gamma: float) -> torch.Tensor:
    """`gamma/2 * E[||grad_x D(x)||^2]`, on real samples only (P1 App. B).

    Real samples only is the definition, not an optimisation: R1 penalises the
    discriminator for having a steep gradient where the data actually is.
    """
    (gradient,) = torch.autograd.grad(real_scores.sum(), real_images, create_graph=True)
    return 0.5 * gamma * gradient.square().sum(dim=(1, 2, 3)).mean()


def r1_gamma(resolution: int, batch_size: int) -> float:
    """P4's first guess: `0.0002 * pixels / minibatch`.

    A starting point and not a setting. P4 found the optimum "vary wildly, from
    0.01 to 10" across datasets and recommends sweeping `[gamma_0/5, gamma_0*5]`,
    so the value this returns is where a sweep begins.

    P1's own choice of 10 is not used here. It was tuned for one dataset at
    1024², and a hyperparameter fitted to a distribution is fitted to *that*
    distribution — a different consideration from licensing, and one that
    outlives it. The two do agree where they overlap: this law returns 6.55 at
    1024² with a minibatch of 32, against P1's 10.
    """
    return R1_GAMMA_COEFFICIENT * resolution * resolution / batch_size


def r1_gamma_sweep(resolution: int, batch_size: int) -> tuple[float, float]:
    """The bracket P4 recommends searching, around `r1_gamma`."""
    centre = r1_gamma(resolution, batch_size)
    return (centre * R1_GAMMA_SWEEP[0], centre * R1_GAMMA_SWEEP[1])


def path_length_weight(resolution: int) -> float:
    """P1 Eq. 5: `ln2 / (r^2 (ln r - ln 2))`."""
    return math.log(2.0) / (resolution**2 * (math.log(resolution) - math.log(2.0)))


class PathLengthPenalty:
    """P1 Eq. 4, with the running target of App. B.

    The target `a` starts at zero and tracks the exponential moving average of
    the observed lengths, so the regularizer asks for *consistency* rather than
    for any particular scale — P1 §3.2: it "allow[s] the optimization to find a
    suitable global scale by itself".

    Two details are ours, because Eq. 4 does not fix them:

    - `y` is scaled by `1/sqrt(H*W)`. Eq. 4 says only that `y` holds "normally
      distributed pixel intensities", which would make the projection grow with
      resolution and change what the weight of Eq. 5 has to absorb.
    - App. B computes the penalty "as an average of all individual layers of the
      synthesis network", which can be read as the mean of the per-layer lengths
      or as the length implied by their mean square. The second is taken, so the
      quantity stays a length in the sense of Eq. 4. The readings agree whenever
      the layers carry comparable gradient, and differ by Jensen's inequality
      otherwise.
    """

    def __init__(self, decay: float = PL_DECAY) -> None:
        self.decay = decay
        self.target = 0.0

    def __call__(self, images: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
        height, width = images.shape[2], images.shape[3]
        noise = torch.randn_like(images) / math.sqrt(height * width)
        (gradient,) = torch.autograd.grad(
            (images * noise).sum(), ws, create_graph=True
        )  # P1 §3.2: J_w^T y = grad_w (g(w) . y)

        lengths = gradient.square().sum(dim=2).mean(dim=1).sqrt()
        penalty = (lengths - self.target).square().mean()
        self.target += (1.0 - self.decay) * (lengths.mean().item() - self.target)
        return penalty


def lazy_adam_hyperparameters(
    lr: float, betas: tuple[float, float], interval: int
) -> tuple[float, tuple[float, float]]:
    """P1 App. B: with `c = k/(k+1)`, use `lr*c` and `beta**c`.

    Regularizing every k-th step means k+1 optimizer steps where there were k,
    so the hyperparameters are compensated to keep the effective schedule. The
    regularization term itself is multiplied by k, which is the caller's job.
    """
    c = interval / (interval + 1)
    return lr * c, (betas[0] ** c, betas[1] ** c)
