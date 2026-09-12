"""One training iteration, with lazy regularization done the way P1 describes it.

Provenance (see PROVENANCE.md), P1 App. B, *Lazy regularization*:

    "the optimizer first sees gradients from the main loss for k iterations,
    followed by gradients from the regularization terms for one iteration. We
    share the internal state of the Adam optimizer between the main loss and the
    regularization terms. To compensate for the fact that we now perform k+1
    training iterations instead of k, we adjust the optimizer hyperparameters
    ... We also multiply the regularization term by k."

**The regularizer is its own optimizer step, not a term added to the loss.** That
is the sentence, and it is the detail most easily implemented as something else:
folding `k * R1` into the discriminator loss would run k steps where the paper
runs k+1, and the hyperparameter compensation that follows would then be
correcting for something that is not happening. The two readings differ in the
Adam state each update sees, which is exactly what the shared-state sentence is
about.

Learning rate and betas come from P1 App. B for configurations E and F: lr 2e-3,
beta1 0, beta2 0.99, eps 1e-8, at a minibatch of 32.
"""

from dataclasses import dataclass, field

import torch

from ogan.augment import AdaptiveAugment
from ogan.discriminator import Discriminator
from ogan.ema import AveragedGenerator
from ogan.generator import STYLE_MIXING_PROB, Generator
from ogan.loss import (
    PL_INTERVAL,
    R1_INTERVAL,
    PathLengthPenalty,
    discriminator_loss,
    generator_loss,
    lazy_adam_hyperparameters,
    path_length_weight,
    r1_gamma,
    r1_penalty,
)
from ogan.mapping import W_DIM, Z_DIM


@dataclass
class TrainingConfig:
    resolution: int = 512
    batch_size: int = 32
    z_dim: int = Z_DIM  # P1 App. B
    w_dim: int = W_DIM  # P1 App. B
    learning_rate: float = 2e-3  # P1 App. B, configs E-F
    betas: tuple[float, float] = (0.0, 0.99)  # P1 App. B
    adam_eps: float = 1e-8  # P1 App. B
    style_mixing_prob: float = STYLE_MIXING_PROB
    large: bool = False
    channel_max: int = 512
    augment: bool = True
    gamma: float | None = None  # P4's law when left unset
    metrics: dict = field(default_factory=dict)


class Trainer:
    """Holds everything a resumed run needs, and nothing it does not."""

    def __init__(self, config: TrainingConfig, device: torch.device | str = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        self.step_index = 0

        self.generator = Generator(
            config.resolution,
            config.z_dim,
            config.w_dim,
            large=config.large,
            channel_max=config.channel_max,
        ).to(self.device)
        self.discriminator = Discriminator(
            config.resolution, large=config.large, channel_max=config.channel_max
        ).to(self.device)
        self.averaged = AveragedGenerator(self.generator, batch_size=config.batch_size)

        # Each optimizer's hyperparameters are compensated for its own interval,
        # since the discriminator regularizes every 16 steps and the generator
        # every 8 (P1 App. B).
        d_lr, d_betas = lazy_adam_hyperparameters(config.learning_rate, config.betas, R1_INTERVAL)
        g_lr, g_betas = lazy_adam_hyperparameters(config.learning_rate, config.betas, PL_INTERVAL)
        self.opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=d_lr, betas=d_betas, eps=config.adam_eps
        )
        self.opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=g_lr, betas=g_betas, eps=config.adam_eps
        )

        self.gamma = config.gamma or r1_gamma(config.resolution, config.batch_size)
        self.path_length = PathLengthPenalty()
        self.pl_weight = path_length_weight(config.resolution)
        self.augment = AdaptiveAugment(batch_size=config.batch_size) if config.augment else None

    # --- the pieces -------------------------------------------------------

    def _augmented(self, images: torch.Tensor) -> torch.Tensor:
        return self.augment(images) if self.augment is not None else images

    def _sample(self, batch: int) -> torch.Tensor:
        z = torch.randn(batch, self.generator.z_dim, device=self.device)
        return self.generator(z, style_mixing_prob=self.config.style_mixing_prob)

    def _discriminator_step(self, real: torch.Tensor) -> dict[str, float]:
        with torch.no_grad():
            fake = self._sample(real.shape[0])

        real_scores = self.discriminator(self._augmented(real))
        fake_scores = self.discriminator(self._augmented(fake))
        loss = discriminator_loss(real_scores, fake_scores)

        self.opt_d.zero_grad(set_to_none=True)
        loss.backward()
        self.opt_d.step()

        if self.augment is not None:
            self.augment.observe(real_scores)
            if self.augment.ready:
                self.augment.adjust()
        return {"loss_d": loss.item(), "scores_real": real_scores.mean().item()}

    def _r1_step(self, real: torch.Tensor) -> dict[str, float]:
        """A separate pass and a separate optimizer step, as App. B describes."""
        real = real.detach().requires_grad_(True)
        scores = self.discriminator(self._augmented(real))
        penalty = r1_penalty(scores, real, self.gamma) * R1_INTERVAL

        self.opt_d.zero_grad(set_to_none=True)
        penalty.backward()
        self.opt_d.step()
        return {"r1": penalty.item() / R1_INTERVAL}

    def _generator_step(self, batch: int) -> dict[str, float]:
        loss = generator_loss(self.discriminator(self._augmented(self._sample(batch))))

        self.opt_g.zero_grad(set_to_none=True)
        loss.backward()
        self.opt_g.step()
        return {"loss_g": loss.item()}

    def _path_length_step(self, batch: int) -> dict[str, float]:
        z = torch.randn(batch, self.generator.z_dim, device=self.device)
        ws = self.generator.styles(z, style_mixing_prob=self.config.style_mixing_prob)
        ws = ws.detach().requires_grad_(True)
        images = self.generator.synthesis(ws)
        penalty = self.path_length(images, ws) * self.pl_weight * PL_INTERVAL

        self.opt_g.zero_grad(set_to_none=True)
        penalty.backward()
        self.opt_g.step()
        return {"path_length": penalty.item() / PL_INTERVAL, "pl_target": self.path_length.target}

    # --- one iteration ----------------------------------------------------

    def step(self, real: torch.Tensor) -> dict[str, float]:
        metrics = self._discriminator_step(real)
        if self.step_index % R1_INTERVAL == 0:
            metrics |= self._r1_step(real)

        metrics |= self._generator_step(real.shape[0])
        if self.step_index % PL_INTERVAL == 0:
            metrics |= self._path_length_step(real.shape[0])

        self.averaged.update(self.generator)
        self.step_index += 1
        if self.augment is not None:
            metrics["augment_p"] = self.augment.p
        return metrics

    # --- resumption -------------------------------------------------------

    def state_dict(self) -> dict:
        """Everything a resumed run needs. A missing entry here does not fail
        loudly — it resumes into a run that is subtly not the one interrupted."""
        return {
            "step_index": self.step_index,
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "averaged": self.averaged.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "path_length_target": self.path_length.target,
            "augment": self.augment.state_dict() if self.augment is not None else None,
            "rng": torch.get_rng_state(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.step_index = state["step_index"]
        self.generator.load_state_dict(state["generator"])
        self.discriminator.load_state_dict(state["discriminator"])
        self.averaged.load_state_dict(state["averaged"])
        self.opt_g.load_state_dict(state["opt_g"])
        self.opt_d.load_state_dict(state["opt_d"])
        self.path_length.target = state["path_length_target"]
        if self.augment is not None and state["augment"] is not None:
            self.augment.load_state_dict(state["augment"])
        torch.set_rng_state(state["rng"])
