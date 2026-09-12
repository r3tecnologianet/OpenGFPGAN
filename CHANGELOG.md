# Changelog

Nothing has been released. There is no model and there are no weights; what
exists is the generator, the discriminator, the objective and a training step,
each value traceable to a paper or marked as our own.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added — the apparatus

- Apache 2.0 for the code, chosen over MIT for its explicit patent grant, since
  commercial adopters are the intended audience and MIT is silent on patents.
  Weights and corpus get their own instruments when they exist.
- `PROVENANCE.md`: the clean-room rule with an operational definition, the
  permitted papers, the forbidden repositories named, and three categories for
  every constant — `paper`, `derived`, `ours` — so a value read from code can
  never be recorded as a value stated in a paper.
- `CONTRIBUTING.md`: do not read the reference implementations; every constant
  carries a source; no FFHQ image in any form.
- Disclosure of who wrote the code and what they had already seen, with the
  evidence that the work is derivation rather than recall.

### Added — the model

Sixty-three values recorded from four papers, each cited to an equation, a table
or a named subsection.

- **`ogan.layers.equalized`** — equalized learning rate (P3 §4.1) and the
  activation gain, kept in separate units so the gain cannot be applied twice.
- **`ogan.layers.modulated`** — weight modulation and demodulation (P1 Eq. 1-3)
  through App. B's grouped-convolution reshape.
- **`ogan.layers.resample`** — filtered 2x resampling with P2's `[1, 2, 1]`.
- **`ogan.layers.synthesis`** — noise injection, the synthesis layer, and tRGB.
- **`ogan.mapping`** — eight fully connected layers at a 100x lower learning
  rate, with P2 Fig. 1's normalise block on the latent.
- **`ogan.synthesis`** — blocks from 4² upward with output skips (P1 §4.1),
  capacity from P3 Table 2.
- **`ogan.generator`** — the two composed, with P2's 90% style mixing.
- **`ogan.discriminator`** — residual blocks with the `1/√2` junction factor and
  P3's minibatch standard deviation.
- **`ogan.loss`** — non-saturating logistic, R1, path length with P1 §3.2's
  Jacobian-vector identity, and App. B's lazy-regularization compensation.
- **`ogan.ema`** — the averaged generator, in half-lives rather than per-step
  decay.
- **`ogan.augment`**, **`ogan.color_augment`** — P4's controller with pixel
  blitting and colour transforms.
- **`ogan.training`** — one iteration, and a checkpoint that carries everything
  a resumed run needs.

211 tests. Every numerical claim is asserted against the paper that makes it.

### Verified against the papers' own numbers

Three figures the architecture had to agree with, none of them tunable towards:

| Claim | Source | Measured |
|---|---|---|
| Generator 25M → 30M, up 22% | P1 footnote 4 | 24.90M → 30.37M, up 21.97% |
| Discriminator 24M → 29M, up 21% | P1 footnote 4 | 24.04M → 29.01M, up 20.7% |
| R1 weight scaling law vs. P1's tuned value | P4 appendix, P1 App. B | 6.55 against 10 at 1024², inside P4's own bracket |

### Found

Things the papers say and the implementations do not, or the other way round.

- **The resampling filter is `[1, 2, 1]`.** P2 specifies a separable 2nd order
  binomial filter; every implementation uses `[1, 3, 3, 1]`, which is third
  order and appears in no paper. Deriving the normalisation from bilinear
  interpolation forces `[1, 2, 1]` with nothing left over, so the two halves of
  P2's sentence confirm each other. It changes every resampling constant in both
  networks.
- **The path length interval is 8.** P1 App. B says so verbatim; the reference
  implementations use 4.
- **The gain normalises the second moment, not the standard deviation.** For
  unit-normal input the gained LeakyReLU has RMS 1.000 and standard deviation
  0.897, so a validation suite asserting `std ∈ [0.99, 1.01]` fails against a
  correct implementation.
- **The equalized scale cancels under demodulation.** Eq. 3 divides by the norm
  of the weights Eq. 1 produced, so it is homogeneous of degree zero in any
  constant factor. It matters only in the output layers.
- **The minibatch standard deviation is the discriminator's only curved
  operation**, and therefore the only reason R1 reaches any bias. Removing it
  takes the count of biases receiving second-order signal from 7 of 10 to 0.
- **Adam's epsilon undercuts the mapping network's stated learning rate.**
  Scaling the gradient down by 100 lands it near 1.9e-7, where P1's own epsilon
  of 1e-8 is 5.4% of the denominator. Measured ratio 0.0089, not 0.01.
- **`[1, 2, 1]/2` over zero insertion is exactly bilinear interpolation**, which
  is what fixes the normalisation with no freedom left.
- **Filtered resampling has two gains**: 1 for a constant and 0.75 for white
  input when upsampling, 0.375 when downsampling. A layer that upsamples cannot
  hold its input's second moment, and demodulation cannot restore it.
- **P2's appendix is ambiguous** about pixelwise normalisation in a way that
  would contradict its own Figure 1. The reading taken is recorded, along with
  the fact that it is a reading.

### Measured

- **The clean-room premise holds on MPS.** Both shapes of second derivative run
  through a grouped convolution and produce finite gradients, so NVIDIA's
  `conv2d_gradfix` shim is unnecessary — an assertion this project rests on,
  now a measurement. The same check has not been run on CUDA.
- **Reduced precision means bfloat16.** float16 returned non-finite gradients
  through the second derivative; bfloat16 did not.
- **Memory is not the constraint on an M4 Pro.** The paper's minibatch of 32
  fits at 512² with an 8.19 GiB peak against a 37.44 GiB ceiling — and 8.19 GiB
  alone exceeds an 8 GB discrete card in total.
- **The rig holding the discrete cards cannot host the project.** Its host CPU
  is a 2009 Celeron with no AVX2, so torch installs and takes a SIGILL on
  import. Recorded in `docs/spikes/`.

### Corrected

Claims this project made and then had to withdraw, kept here because a provenance
record that hides them is worth less than none.

- An audit asserted that every error in a contaminated reference document fell in
  a code-only value. Reading P1 falsified it: the document was right about the
  path length interval and about Eq. 5's formula, and the audit was wrong about
  the activation gain constant.
- The synthesis layer claimed to preserve second moments on a path where it
  cannot, and three tests asserted properties wider than the code.
- The blitting distributions were invented rather than read. P4 draws each
  transform's parameter after taking it, and translates by at most an eighth of
  the side with reflection — where the first implementation always flipped, never
  rotated by zero, and wrapped around the whole width.
- `Discriminator` accepted `channel_max` and ignored it. Every test still passed.
