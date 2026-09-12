# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A clean-room reimplementation of StyleGAN2 from its published papers, on the way
to a blind face restoration model with provenance that survives an audit. The
provenance record, not the code, is the deliverable — see `PROVENANCE.md`.

## Commands

The virtualenv is managed by `uv`; there is no Makefile and no CI.

```sh
uv run --extra dev pytest                      # full suite, 221 tests, ~22s
uv run --extra dev pytest tests/test_loss.py   # one file
uv run --extra dev pytest -k demodulation      # one test by name
uv run --extra dev ruff check .                # lint (E, F, I, UP, B; line length 100)
uv run python scripts/smoke_run.py --steps 4000   # end-to-end convergence check
./papers/fetch.sh                              # download the five primary sources
uv run python docs/spikes/device_viability.py mps  # or cuda / cpu
```

`smoke_run.py` trains the whole assembly on a toy disc distribution and reports
metrics against a ground truth the toy generator knows. **Treat its numbers as a
regression baseline**: the unit suite is structurally unable to see a broken
gradient path, and a change can leave all 221 tests green while preventing
convergence. The recorded result is in `docs/spikes/2026-09-12-convergence-smoke-run.md`.

## Non-negotiable rules

These are the reason the project exists. Breaking one silently destroys the
repository's only claim to value.

1. **Never open a reference implementation.** `PROVENANCE.md` names them:
   `NVlabs/stylegan2*`, `rosinality/stylegan2-pytorch`, `XPixelGroup/BasicSR`
   (`stylegan2_arch.py`), `TencentARC/GFPGAN`, annotated ports. Work from the
   papers in `papers/`. This includes issue threads that quote source.
2. **`docs/reference/` is gitignored and is not a clean-room source.** The
   architecture document there was written by reading the forbidden repos. It may
   never be cited, and values may not be taken from it.
3. **No FFHQ, in any form** — not the aligned release, not a subset, not a
   derivative, not a crop from one. Never a substitute of it under another name.
4. **Every constant carries a source**, tagged with one of three categories:
   `paper` (cite ref + equation/table/section), `derived` (show the derivation),
   `ours` (record the measurement that produced the value). A value read from
   code may never be recorded as `paper` — that substitution is the failure mode
   `PROVENANCE.md` exists to prevent.
5. **English** for code, comments, commit messages and all documentation.
6. Papers are fetched, never committed (`papers/*.pdf` is gitignored).

When a paper and an implementation disagree, follow the paper. The divergences
are the evidence: `[1,2,1]` not `[1,3,3,1]`, path-length interval 8 not 4,
activation gain 1.38675 not √2, 23 style inputs at 512² not 16, γ_R1 from P4's
scaling law not P1's FFHQ-tuned 10.

## Architecture

Bottom-up; each layer depends only on those above it.

- `ogan/layers/equalized.py` — He-scaled `EqualizedLinear` / `EqualizedConv2d`,
  plus `leaky_relu` carrying `ACTIVATION_GAIN`. The layer/activation split is
  deliberate: applying the gain in both squares it to ≈1.92 per layer.
- `ogan/layers/modulated.py` — P1 Eq. 1–3. Modulate, demodulate, then App. B's
  reshape to `[1, N*C_in, H, W]` with `groups=N` so a batch of per-sample weights
  runs as one convolution. No bias; P1 §2 moves `b` and `B` outside the block.
- `ogan/layers/resample.py` — `[1,2,1]` binomial up/downsampling. Over zero
  insertion this is exactly bilinear, which fixes the normalisation constants.
- `ogan/layers/synthesis.py` — `NoiseInjection`, `SynthesisLayer`, `ToRGB`.
- `ogan/mapping.py`, `ogan/synthesis.py`, `ogan/generator.py` — mapping network
  (8 layers, lr multiplier 0.01), synthesis network (P3 Table 2 channel table,
  output skips, `num_ws = 2 + 3*(blocks-1)`), and the generator that joins them
  with style mixing.
- `ogan/discriminator.py` — residual blocks scaled by `2**-0.5`, minibatch stddev
  in the epilogue. Note stddev is the discriminator's **only** curved operation:
  remove it and biases stop receiving second-order signal, which silently breaks
  R1.
- `ogan/loss.py` — logistic losses, `r1_penalty`, `PathLengthPenalty` (EMA target,
  decay 0.99), the lazy intervals (k=16 for D, k=8 for G), and
  `lazy_adam_hyperparameters` implementing P1's `c = k/(k+1)` compensation.
- `ogan/ema.py` — `AveragedGenerator`, parameterised by half-life in images
  (20k), not by a per-step decay. Buffers are copied, not averaged.
- `ogan/augment.py`, `ogan/color_augment.py` — P4's ADA controller plus pixel
  blitting and colour transforms. **Geometric transforms are the missing third
  category**, and `AdaptiveAugment.__call__` leaves the gap between blitting and
  colour on purpose — the order is part of what P4 measured.
- `ogan/training.py` — `TrainingConfig` + `Trainer`. The regularizer is its **own
  optimizer step**, not a term folded into the loss; folding it in would run k
  steps where the paper runs k+1 and make the hyperparameter compensation correct
  for something that is not happening.
- `ogan/toy.py` — the disc distribution and its metrics, for `smoke_run.py`.

Tests mirror modules one-to-one under `tests/`. A numerical component ships with a
numerical test: a stated tolerance and the paper equation being verified. A test
that only checks the code runs does not test a clean-room port.

## Documentation conventions

- `PROVENANCE.md` — per-component source record. Any new constant goes here in the
  same commit that introduces it. Corrections are written as corrections, not
  quietly amended; the file already records claims that were withdrawn.
- `CHANGELOG.md` is organised by what was *established* (Added / Verified / Found
  / Measured / Corrected), not by version.
- `docs/spikes/YYYY-MM-DD-<topic>.md` — one measurement, its command, its result.
- Commit subjects are lowercase `area: what was established`, stating the finding
  rather than the diff (e.g. `ogan: equalized learning rate, and the second moment
  is what it preserves`).
