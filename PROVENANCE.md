# Provenance

This file is the point of the project. Without it, OpenGFPGAN is just another
face restoration model with an unverifiable claim attached.

## The rule

Clean-room (Chinese wall) has an operational definition, and loose adherence
does not count:

1. A functional specification is derived **from published papers only**. It
   contains equations, tensor shapes and hyperparameters — facts, not expression.
2. Implementation follows that specification. The person writing the code does
   **not** read the reference implementations.

## Permitted primary sources

Papers. Freely readable, and the ideas in them are not restricted by copyright.

| Ref | Paper | arXiv |
|---|---|---|
| P1 | Analyzing and Improving the Image Quality of StyleGAN (StyleGAN2) | 1912.04958 |
| P2 | A Style-Based Generator Architecture for GANs (StyleGAN) | 1812.04948 |
| P3 | Progressive Growing of GANs (ProGAN) | 1710.10196 |
| P4 | Training Generative Adversarial Networks with Limited Data (ADA) | 2006.06676 |
| P5 | Towards Real-World Blind Face Restoration with Generative Facial Prior (GFPGAN) | 2101.04061 |

## Forbidden sources

Not "discouraged" — forbidden, and naming them is part of the record:

- `NVlabs/stylegan2`, `NVlabs/stylegan2-ada-pytorch` (source, and issue threads
  that quote source)
- `rosinality/stylegan2-pytorch`
- `XPixelGroup/BasicSR` — `basicsr/archs/stylegan2_arch.py` in particular, which is
  the file GFPGAN ships and the reason this project exists
- `TencentARC/GFPGAN`
- Annotated reimplementations (labml.ai and similar). An annotated port is still a port.

## Per-component record

Every equation and every hyperparameter carries a source. Three categories, and
the third is the one that needs care:

| Category | Meaning | How it is recorded |
|---|---|---|
| **paper** | stated in a paper | cite ref + equation/table/section |
| **derived** | follows from a paper by our own derivation | cite ref + show the derivation |
| **ours** | exists only in reference implementations, so we re-derive or tune it ourselves | record the measurement or ablation that produced our value |

A value may never be recorded as **paper** when it was in fact read from code.
That substitution is the failure mode this file exists to prevent.

### From P1 — StyleGAN2

| Component | Value | Category | Source |
|---|---|---|---|
| Weight modulation | `w'[i,j,k] = s[i] · w[i,j,k]` | paper | P1 Eq. 1 |
| Output standard deviation | `σ[j] = sqrt(Σ_ik w'[i,j,k]²)` | paper | P1 Eq. 2 |
| Weight demodulation | `w''= w' / sqrt(Σ_ik w'² + ε)` | paper | P1 Eq. 3 |
| Demodulation implementation | grouped convolution: reshape to one sample with N groups, "the reshaping operations do not actually modify the contents" | paper | P1 App. B, *Weight demodulation* |
| Where demodulation applies | all conv layers **except** the tRGB output layers | paper | P1 App. B, *Generator redesign* |
| Z and W dimensionality | 512 | paper | P1 App. B |
| Mapping network | 8 fully connected layers, 100× lower learning rate | paper | P1 App. B |
| LeakyReLU | α = 0.2 | paper | P1 App. B |
| Constant input `c1` | initialised `N(0, 1)` | paper | P1 App. B, *Generator redesign* |
| Weight init | all weights `N(0, 1)` | paper | P1 App. B |
| Bias and noise-scale init | zero — **except** affine transformation biases, initialised to **one** | paper | P1 App. B |
| Noise broadcast | a single shared scaling factor for all feature maps of a layer | paper | P1 App. B |
| Topology, configs E–F | output skips in G, residual connections in D | paper | P1 §4.1, App. B |
| Loss | non-saturating logistic, with R1 | paper | P1 App. B |
| Adam | β1 = 0, β2 = 0.99, ε = 1e-8, minibatch = 32 | paper | P1 App. B |
| Learning rate, configs E–F | λ = 2e-3, fixed (no progressive growing) | paper | P1 App. B, *Progressive growing* |
| Lazy regularization interval | **k = 16 for the discriminator, k = 8 for the generator** | paper | P1 App. B, *Lazy regularization* |
| Lazy regularization compensation | `c = k/(k+1)`, `λ' = cλ`, `β' = β^c`, regularization term multiplied by `k` | paper | P1 App. B |
| Path length objective | `E[(‖J_wᵀ y‖₂ − a)²]`, y random images with normally distributed pixel intensities | paper | P1 Eq. 4 |
| Jacobian-vector identity | `J_wᵀ y = ∇_w(g(w) · y)` | paper | P1 §3.2 |
| Path length target `a` | initialised to zero, tracked per-GPU as an EMA with `β_pl = 0.99` | paper | P1 App. B, *Path length regularization* |
| Path length weight | `γ_pl = ln2 / (r²(ln r − ln 2))` | paper | P1 Eq. 5 |
| Path length and style mixing | computed as an average over all individual synthesis layers | paper | P1 App. B |
| R1 weight | γ = 10 for all runs except config E of Table 1 and LSUN C HURCH / H ORSE, where γ = 100. **At 1024², FFHQ.** | paper | P1 App. B, *Dataset-specific tuning* |
| Cost of the fused CUDA kernels | ~30% training time, ~20% memory footprint, config E at 1024² | paper | P1 App. B, *Performance optimizations* |

That last row is the budget for this project's central bet. The authors measured
what their hand-written kernels bought them; removing those kernels is expected to
give roughly that back. It is a paper-sourced figure, not a guess.

### From P3 — ProGAN

| Component | Value | Category | Source |
|---|---|---|---|
| Weight init | trivial `N(0, 1)` | paper | P3 §4.1 |
| Bias init | zero | paper | P3 §A.1 |
| Equalized learning rate, mechanism | `N(0,1)` init, weights scaled at runtime by the per-layer constant from He's initializer, *not* at initialization | paper | P3 §4.1 |
| Equalized learning rate, constant and direction | `gain/sqrt(fan_in)`, `gain = sqrt(2/(1+α²)) ≈ 1.38675` for α = 0.2, applied **once** | **derived** | see the note below |
| Pixelwise feature normalization | `b[x,y] = a[x,y] / sqrt((1/N)·Σ_j a[x,y,j]² + ε)`, ε = 1e-8, N = number of feature maps | paper | P3 §4.2 |
| Minibatch stddev, computation | standard deviation per feature per spatial location over the minibatch, averaged over all features and locations to **a single value**, replicated and concatenated as one constant feature map. No learnable parameters, no hyperparameters. | paper | P3 §3 |
| Minibatch stddev, placement | at 4×4 resolution, toward the end of the discriminator | paper | P3 §A.1 |
| Generator weight EMA, decay | 0.999 | paper | P3 §A.1 |
| Latent | 512-dimensional, points on a hypersphere | paper | P3 §A.1 |
| Image range | [-1, 1] | paper | P3 §A.1 |
| LeakyReLU | leakiness 0.2 in all layers of both networks, except the last, which is linear | paper | P3 §A.1 |
| Up/downsampling | 2×2 element replication / average pooling — **superseded by P1**, which filters | paper | P3 §A.1 |

**The equalized learning rate is ambiguous in the primary source, and that has to be
said rather than papered over.** P3 §4.1 writes `ŵᵢ = wᵢ/c`, where `c` is "the
per-layer normalization constant from He's initializer". He's constant is
`sqrt(2/fan_in)`. Read literally, dividing by it *amplifies* layers with wide
fan-in — the opposite of the section's stated goal, that "the dynamic range, and
thus the learning speed, is the same for all weights".

So the mechanism is `paper` and the constant is `derived`: we take the stated
intent — unit-variance activations under LeakyReLU with α = 0.2 — and derive
`gain = sqrt(2/(1+α²)) = sqrt(2/1.04) ≈ 1.38675`, scaling by `gain/sqrt(fan_in)`.

Two consequences worth stating explicitly, because both are easy to get wrong:

- `sqrt(2) ≈ 1.41421` is He's constant for plain ReLU. These networks use
  LeakyReLU with α = 0.2 throughout (P3 §A.1), so the α-aware constant is the
  correct derivation, not the ReLU one.
- It is applied **once**. Whether it lives in the weight scale or in the
  activation is a placement choice; applying it in both places squares the gain
  to ≈1.92 per layer, and training diverges.

### From P2 — StyleGAN

| Component | Value | Category | Source |
|---|---|---|---|
| Resampling filter | a separable **2nd order binomial** filter, i.e. `[1, 2, 1]`, lowpass applied **after each upsampling** layer and **before each downsampling** layer | paper | P2, improved baseline (config B) |
| Style mixing, mechanism | two latents `z1, z2` through the mapping network; `w1` controls styles before a randomly selected crossover point in the synthesis network, `w2` after it | paper | P2 §3.1 |
| Style mixing, probability | **90%** of training images (config F). Table 2, one latent at test time: 0% → FID 4.42, 50% → 4.41, **90% → 4.40**, 100% → 4.83 | paper | P2 Table 2 |
| Loss, FFHQ | non-saturating logistic with R1, γ = 10 | paper | P2, config B |
| Learning rate at 512² and 1024² | 0.002 rather than 0.003, for stability | paper | P2, config B |
| Mirror augmentation | enabled for CelebA-HQ and FFHQ, disabled for LSUN | paper | P2, config A |

**The papers say `[1, 2, 1]`, and no paper says `[1, 3, 3, 1]`.** This is the
sharpest paper/code divergence found so far, and it sits in the resampling path,
which touches every layer of both networks.

P2 states the filter explicitly: a separable 2nd order binomial filter. Second
order binomial coefficients are `[1, 2, 1]`. P1 App. B then lists "bilinear
filtering in all up/downsampling layers" among the details it **kept unchanged**
from StyleGAN — so P1 does not redefine it, it inherits it.

The `[1, 3, 3, 1]` kernel that every implementation uses is third order, and it
appears in none of P1, P2 or P3. It is implementation-only.

Consequence for the normalization constants, since they follow from the kernel and
not from taste. For `K = k ⊗ kᵀ` with `k = [1, 2, 1]`, the coefficients sum to
`(1+2+1)² = 16`, so:

- downsampling: `K_down = K/16`, coefficients summing to 1
- upsampling with zero insertion: gain `s² = 4`, so `K_up = 4·(K/16) = K/4`

A `[1, 3, 3, 1]` kernel sums to 64 and gives `/64` and `/16` instead. Starting
from the wrong kernel means every resampling constant in the network is wrong.
We start from `[1, 2, 1]`, cited. Adopting `[1, 3, 3, 1]` later is allowed, as
category `ours`, and only against a measurement of ours.

### Not sourced yet

P1 reuses components from P2 and P3 by citation without restating their values.
P3 closed three, P2 closed the last two. One area remains, and it was always
going to need its own paper:

| Component | Needs |
|---|---|
| Everything about adaptive discriminator augmentation | P4 |

Every value the generator and discriminator need is now either cited to a paper
or explicitly marked `derived` or `ours` below. Nothing is waiting on a guess.

### Exists only in implementations — category `ours`

Needed in practice, absent from P1. Each has to be re-derived or tuned by us, and
the value recorded with the measurement that produced it:

| Component | Why it is `ours` |
|---|---|
| Scaling of `y` by `1/sqrt(H·W)` in the path length term | P1 Eq. 4 specifies only that `y` is a random image with normally distributed pixel intensities |
| Computing path length on a fraction of the minibatch | not in P1; matters because it decides whether the model fits in a small memory budget |
| `γ_R1` at 512² | P1 gives 10 **for 1024²** and says the optimum "vary considerably between datasets and configurations". P4 offers a resolution heuristic; either way this is a sweep, not an inherited constant |
| Activation clamping under reduced precision | not in P1 |
| Choice of ε under reduced precision | P1 gives ε = 1e-8 without stating a precision regime |
| Minibatch stddev **group size** | P3 §3 computes the statistic over the whole minibatch and introduces no subgroup. Splitting the batch into groups of 4 is an implementation-only choice |
| Generator EMA schedule | P3 §A.1 gives a fixed decay of 0.999. Implementations instead use a half-life measured in images, which is a different thing |
| The `[1, 3, 3, 1]` resampling kernel | third order, and in none of the three papers. P2 specifies `[1, 2, 1]` and P1 inherits it unchanged |


## Known contamination in the design record

An architecture document written before this rule existed cites
`rosinality/stylegan2-pytorch/model.py` and `basicsr.archs.stylegan2_arch` in its
own reference list. It was written by reading them, so it is **not** a clean-room
source. It is held locally, is deliberately not published in this repository, and
may never be cited here.

An earlier revision of this file claimed that every error found in that document
fell in a value absent from the papers. **Reading P1 falsified that claim**, and
the record is corrected here rather than quietly amended:

- The generator's lazy regularization interval of **8** is what P1 App. B states.
  The reference implementations use 4. The document was right and the audit was
  wrong, and paper and code genuinely disagree here.
- The path length weight formula **is** P1 Eq. 5. Only the value the document
  tabulates for it is wrong: at r = 1024 the formula gives ≈1.06e-7, not the
  ≈5.37e-7 printed.
- The grouped-convolution implementation of demodulation is stated in P1 App. B.
  It is not an implementation detail that leaked from code.
- The gain constant `≈1.38675` the document uses is a defensible derivation of
  He's constant for LeakyReLU, which the audit disputed in favour of `sqrt(2)`.
  The audit was wrong: these networks are LeakyReLU throughout. What survives is
  the narrower point, that the document applies the gain twice.
- The audit commended the document's FIR normalization constants, `/64` and
  `/16`. The arithmetic is right for the kernel it assumes, but the kernel is
  `[1, 3, 3, 1]`, which no paper states. Commending it was commending a
  code-sourced value — the failure mode this file exists to catch, committed by
  the audit itself.

What survives is narrower and more useful: **paper and code disagree in both
directions**, so neither "the code does X" nor a second-hand summary is a
substitute for reading the source. That is the rule this file enforces, and it
now rests on a measured disagreement rather than on a pattern that did not hold.

## Evaluation metrics

FID loads an ImageNet-trained InceptionV3. It sits outside the gradient path: no
weight of ours derives from it. It is declared here rather than omitted, and any
published result says which metric implementation produced it.
