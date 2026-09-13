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

## Who wrote this, and what they had already seen

The clean-room rule above says the implementer does not read the reference
implementations. That is true of this repository and is checkable: no reference
implementation was opened while any of this code was written. But it is not the
whole disclosure, and a provenance document that omitted the rest would be worth
less than nothing.

**This code was written by a language model, trained on public code that includes
the repositories listed as forbidden below.** "Did not consult them" is accurate.
"Has never seen them" is not, and cannot be made so.

Whether that breaks a clean-room claim is a question about copying expression,
not about exposure. What can be offered is evidence, and the evidence here is
unusually direct: **the code diverges from every published implementation at
exactly the points where the papers and those implementations disagree.**

| Value | Here, from the papers | Every implementation |
|---|---|---|
| Resampling kernel | `[1, 2, 1]` (P2: "2nd order binomial") | `[1, 3, 3, 1]` |
| Path length interval | 8 (P1 App. B, verbatim) | 4 |
| Activation gain | 1.38675 (He, for LeakyReLU α=0.2) | 1.41421 |
| Style inputs at 512² | 23, one per input | 16, indices shared |
| `γ_R1` at 512², batch 32 | 1.638 (P4's law) | 10, inherited from P1 |

Recall reproduces what it has seen. Every row above is a place where doing so
would have been easier and would have produced a different file. The divergences
are not stylistic: `[1, 2, 1]` changes every resampling constant in both
networks, and 23 style inputs changes the shape of `w`.

This is evidence, not proof, and it is offered as such. Anyone auditing the
project can check every row against the papers and against any implementation.

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
| Lazy regularization intervals and compensation | see the App. B rows above; `lazy_adam_hyperparameters` implements them | paper | P1 App. B |
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

### From P3 Table 2 and P1 §4 — capacity and topology

| Component | Value | Category | Source |
|---|---|---|---|
| Feature maps per resolution | 512 at 4², 8², 16², 32²; then 256, 128, 64, 32, 16 at 64² through 1024² | paper | P3 Table 2 |
| The larger configuration | doubles feature maps "in resolutions 64²-1024², while keeping other parts of the networks unchanged" | paper | P1 footnote 4 |
| Generator size at 1024² | **25M parameters, rising to 30M — up 22%** | paper | P1 footnote 4 |
| Output skips | the image is formed by "upsampling and summing the contributions of RGB outputs corresponding to different resolutions", with bilinear filtering in all up and downsampling | paper | P1 §4.1, Fig. 7b |
| Constant input | initialised `N(0, 1)`; bias, noise and normalisation "can also be safely removed without observable drawbacks" | paper | P1 App. B, §2.1 |
| Residual scaling in D | the junction doubles signal variance, "which we cancel by multiplying with 1/√2" | paper | P1 Fig. 7 footnote 3 |

The parameter count is the most useful row in this file. It is a number the whole
architecture has to agree on — channel table, two convolutions per block and one
at 4², a modulated tRGB at every resolution, and whether the mapping network is
counted. Measured: 24.90M and 30.37M, up 21.97%. The synthesis network alone is
22.8M, so P1's figure includes the mapping network.

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

### Derived while implementing

Consequences of the paper's own equations, recorded because they are not obvious
and the code would otherwise look wrong to a later reader.

| Observation | Follows from |
|---|---|
| **The equalized scale cancels in a demodulated convolution.** Eq. 3 divides by the norm of the weights Eq. 1 produced, so the expression is homogeneous of degree zero in any constant factor of the weight — `1/sqrt(fan_in)` included. It is load-bearing only in the output layers, where App. B omits demodulation. Pinned by a test, since it is equally inviting to delete as dead weight or to reapply on the assumption that it matters. | P1 Eq. 1, Eq. 3, App. B |
| **The LeakyReLU gain preserves the second moment, not the standard deviation.** LeakyReLU leaves a positive mean, so unit-normal input gives RMS 1.000 and standard deviation 0.897. The second moment is the quantity that propagates: for a following layer with zero-mean weights, `Var(Σ w·y) = Σ Var(w)·E[y²]`. A validation suite asserting `std ∈ [0.99, 1.01]` after an activation fails against a correct implementation. | P3 §4.1 intent, P3 §A.1 |
| **`[1, 2, 1]/2` over zero insertion *is* bilinear interpolation, which fixes the normalisation with no freedom left.** Writing the 1D kernel as `[a, b, a]`, outputs on an original position get `b·x` and outputs between two originals get `a·(x_left + x_right)`. Bilinear wants the first to be `x` and the second their mean, forcing `b = 1`, `a = 1/2`. So P2's "2nd order binomial filter" and P2's "bilinear sampling" are the same statement, and each confirms the other. Downsampling normalises to unit DC gain instead: `[1, 2, 1]/4`. `[1, 3, 3, 1]` does not satisfy this. | P2, config B |
| **The minibatch standard deviation is the only source of curvature in the discriminator, and therefore the only reason R1 reaches any bias.** Convolutions, resampling and the linear head are linear and LeakyReLU is piecewise linear, so the score's gradient with respect to its input is piecewise constant and its derivative with respect to a bias is zero. The statistic takes a square root of a mean of squares over the batch, which is genuinely curved. Measured: 7 of 10 biases receive second-order signal with it, **0 of 10 without**. The three biases downstream of it get nothing — two exactly zero, and the final linear bias unreachable, since it shifts the score by a constant. | P3 §3, measured |
| **Filtered downsampling attenuates white input to 0.375**, the L2 norm of the normalised kernel, measured 0.3754 — the mirror of the upsampling figure and a stronger effect, since a lowpass discards most of white noise's energy. It is why a residual block fed white noise emerges near 0.42 regardless of the 1/√2 junction factor. `[1, 3, 3, 1]` gives 0.3125. | P2 config B, derived |
| **A half-life in images and a per-step decay are the same mechanism, and the batch size is the conversion.** With `B` images per step and a half-life of `H` images, the average halves over `H/B` steps, so `decay = 0.5^(B/H)`. P4's 20k half-life gives 0.99889 at P1's minibatch of 32, where P3 states 0.999 — itself a half-life of 22.2k at that batch. Two papers, two notations, 10% apart. The half-life form is implemented, because it keeps its meaning when the batch size changes and a fixed decay silently does not. | P3 §A.1, P4, derived |
| **Adam's epsilon damps the mapping network past the 100× P1 asks for.** The multiplier scales the stored gradient by 0.01, landing the mapping network's gradients near 1.9e-7 — where Adam's eps of 1e-8 is 5.4% of the denominator rather than a rounding term. Measured effective-weight ratio: 0.0089 at eps 1e-8, 0.0100 at eps 1e-12. Both the mechanism and the epsilon are the paper's, so this is faithful rather than wrong, but "the mapping network barely moves" is otherwise a puzzle to be rediscovered mid-run. | P1 App. B, measured |
| **Filtered upsampling has two gains: 1 for a constant, 0.75 for white noise.** After zero insertion the four output parities see different subsets of the 2D kernel — weights 1, 1/2, 1/2, 1/4 — so their variances average 9/16, and `sqrt(9/16) = 0.75`. Measured 0.7522. So a layer that upsamples cannot hold its input's second moment: demodulation normalises weights against Eq. 2's unit-variance assumption and never inspects the input. Inherent to the operation, not to the kernel — `[1, 3, 3, 1]` gives 0.625 by the same calculation. Any claim about preserving scale has to say which input it means. | P2 config B, derived |
| **The activation follows both additions.** P1 §2.1 moves bias and noise outside the style block but does not say in prose where the nonlinearity sits relative to them. A bias applied *after* LeakyReLU could only offset the output, never move where the function bends — which is what a bias in a nonlinear network is for. So both additions precede it. Their order relative to *each other* is not a decision at all: addition commutes. | P1 §2.1 |
| **Path length cannot move a layer's bias or its noise gain.** Both enter the graph only inside LeakyReLU, whose derivative is piecewise constant, so differentiating that derivative with respect to either is zero almost everywhere. Measured exactly 0.0 against 10³ for the affine and convolution weights. An identity, not a wiring fault — and a test asserting only `grad is not None` would pass while verifying nothing, since both still receive a first-order gradient and so are allocated a zero tensor. | P1 §3.2, measured |
| **A deep stack of these layers drifts, without being biased.** One finite-width random layer's empirical gain deviates by order `1/sqrt(units)`, ≈4% at 512, and the deviations compound multiplicatively — so ten layers land anywhere in roughly [0.8, 1.3]. The testable claim is unbiasedness across seeds, not exactness through a stack. | measurement, `tests/test_equalized.py` |

### From P4 — StyleGAN2-ADA

| Component | Value | Category | Source |
|---|---|---|---|
| Overfitting heuristic | `r_t = E[sign(D_train)]` — "the portion of the training set that gets positive discriminator outputs". 0 means no overfitting, 1 complete overfitting | paper | P4 Eq. 1, §3 |
| Why this heuristic | "far less sensitive to the chosen target value and other hyperparameters than the obvious alternative of looking at `E[D_train]` directly", and needs no validation set | paper | P4 §3 |
| Averaging window | the mean over **N = 4 consecutive minibatches**, 4 × 64 = 256 images | paper | P4 §3 |
| Target value | **0.6** | paper | P4 §3 |
| `p` initialisation | zero | paper | P4 §3 |
| `p` adjustment cadence | once every four minibatches | paper | P4 §3 |
| `p` adjustment size | fixed, sized "so that p can rise from 0 to 1 sufficiently quickly, e.g., in 500k images" | paper | P4 §3 |
| `p` floor | clamped from below to zero after every step | paper | P4 §3 |
| Full pipeline | 18 transformations in 6 categories: pixel blitting (x-flips, 90° rotations, integer translation), general geometric, colour, image-space filtering, additive noise, cutout | paper | P4 §2 |
| **Categories actually used** | **pixel blitting, geometric and colour only** — "we choose to use only pixel blitting, geometric, and color transforms for the rest of our tests" | paper | P4 §2 |
| Augmentations must be differentiable | "we execute augmentations also when training the generator, which requires the augmentations to be differentiable" | paper | P4 §2 |
| How `p` is applied | each transformation applied with probability `p` or skipped, the same `p` for all, randomised separately per augmentation and per image, in a fixed order | paper | P4 §2 |
| **R1 weight, first guess** | **`γ₀ = 0.0002 · N/M`**, N the pixel count and M the minibatch size, with "different values in the range `[γ₀/5, γ₀·5]`" recommended | paper | P4 appendix |
| Generator weight EMA | a **half-life of 20k images** | paper | P4 |

Two rows change earlier entries in this file.

**`γ_R1` is no longer open.** It was recorded as `ours`, on the grounds that P1
gives 10 for 1024² and says the optimum "vary[s] considerably". P4 gives the
scaling law: `0.0002 · N/M`. At 512² with a minibatch of 32 that is **1.638**,
with a recommended sweep over [0.33, 8.2] — so P1's 10, carried to 512² by the
document this project audited, is six times the paper's own first guess.

The formula cross-checks against P1. At 1024² with a minibatch of 32 it returns
6.55, and P1 chose 10 for FFHQ at that resolution — the same order, inside the
recommended range, from two papers that state it in incompatible forms. That is
the kind of agreement worth noting, since it is the only independent confirmation
either value gets.

**The generator EMA schedule is no longer `ours` either.** It was recorded as a
code-only divergence from P3's fixed decay of 0.999. P4 states the half-life form
directly, so both exist in the papers and the choice between them is a choice
between two cited configurations rather than between a paper and an
implementation.

And one row makes a planning decision for us. The design note for this project
assumed a subset of the augmentations would be implemented first, "expanding only
if the r_t indicates". P4 reaches the same place by measurement: on a 2k set "the
vast majority of the benefit came from pixel blitting and geometric transforms.
Color transforms were modestly beneficial, while image-space filtering, noise, and
cutout were not particularly useful". The subset is what the authors ship.

There is a condition attached that matters for our corpus. At 140k images "the
situation was markedly different: all augmentations were harmful". ADA is a
small-data mechanism, and whether we need it at all depends on what the corpus
measurement returns.

### From P5 — GFPGAN

Read for the restoration architecture. P5 states its full objective and every
loss weight, which is more than P1 does for StyleGAN2.

| Component | Value | Category | Source |
|---|---|---|---|
| Overall shape | a degradation removal module and a pretrained face GAN as prior, bridged by a latent code mapping and several Channel-Split Spatial Feature Transform layers | paper | P5 §3.1, Fig. 2 |
| Degradation removal | U-Net, producing two outputs: `F_latent, F_spatial = U-Net(x)` | paper | P5 Eq. 1, §3.2 |
| Why a U-Net | to "increase receptive field for large blur elimination" and to "generate multi-resolution features" | paper | P5 §3.2 |
| U-Net depth | seven downsamples and seven upsamples, each with a residual block | paper | P5 §4.1 |
| Latent mapping | `W = MLP(F_latent)`, then `F_GAN = StyleGAN(W)` — several MLP layers into the intermediate latent space | paper | P5 Eq. 2, §3.3 |
| Why intermediate `W`, not `Z` | `W` "better preserv[es] semantic property" | paper | P5 §3.3 |
| SFT | `α, β = Conv(F_spatial)`; `F_output = α ⊙ F_GAN + β` | paper | P5 Eq. 3, §3.4 |
| **CS-SFT** | split `F_GAN` in the channel dimension, modulate one half and pass the other through unchanged: `F_output = Concat[Identity(F_GAN^split0), α ⊙ F_GAN^split1 + β]` | paper | P5 Eq. 4, §3.4 |
| Why the split | the modulated half contributes fidelity, the identity half realness; "leave the left split as identity to achieve a good balance of realness and fidelity" | paper | P5 §3.4, §2 |
| CS-SFT placement | at each resolution scale, coarse-to-fine | paper | P5 §3.4 |
| CS-SFT internals | two convolutional layers per layer, generating `α` and `β` | paper | P5 §4.1 |
| Pyramid restoration loss | L1 on the U-Net decoder's output at each resolution scale against a pyramid of the ground truth, as intermediate supervision "in the early stage of training" | paper | P5 §3.2 |
| Reconstruction loss | `L_rec = λ_l1‖ŷ − y‖₁ + λ_per‖φ(ŷ) − φ(y)‖₁` | paper | P5 Eq. 5 |
| Perceptual network `φ` | pretrained VGG-19, `{conv1..conv5}` feature maps **before activation** | paper | P5 Eq. 5, §3.5 |
| Adversarial loss | `L_adv = −λ_adv·E[softplus(D(ŷ))]`, "similar to StyleGAN2" | paper | P5 Eq. 6 |
| Facial component loss | `L_comp = Σ_ROI λ_local·E[log(1 − D_ROI(ŷ_ROI))] + λ_fs‖Gram(ψ(ŷ_ROI)) − Gram(ψ(y_ROI))‖₁` | paper | P5 Eq. 7 |
| Components | `{left_eye, right_eye, mouth}`, cropped by ROI align | paper | P5 Eq. 7, §4.1 |
| Feature style loss | Gram matrices of features from the **learned local discriminators**, not a fixed network. "performs better than previous feature matching loss in terms of generating realistic facial details and reducing unpleasant artifacts" | paper | P5 §3.5 |
| Identity preserving loss | `L_id = λ_id‖η(ŷ) − η(y)‖₁`, `η` = pretrained ArcFace | paper | P5 Eq. 8, §3.5 |
| Total objective | `L_total = L_rec + L_adv + L_comp + L_id` | paper | P5 Eq. 9 |
| `λ_l1` | 0.1 | paper | P5 §3.5 |
| `λ_per` | 1 | paper | P5 §3.5 |
| `λ_adv` | 0.1 | paper | P5 §3.5 |
| `λ_local` | 1 | paper | P5 §3.5 |
| `λ_fs` | 200 | paper | P5 §3.5 |
| `λ_id` | 10 | paper | P5 §3.5 |
| Degradation model | `x = [(y ⊛ k_σ) ↓_r + n_δ]_JPEG_q` — Gaussian blur, downsample, Gaussian noise, JPEG | paper | P5 Eq. 10, §4.1 |
| Degradation ranges | `σ ∈ {0.2:10}`, `r ∈ {1:8}`, `δ ∈ {0:15}`, `q ∈ {60:100}`, sampled per training pair, plus colour jittering | paper | P5 §4.1 |
| Prior resolution | StyleGAN2 with 512² outputs | paper | P5 §4.1 |
| **Prior channel multiplier** | **one**, "for compact model size" — half the width of config-F | paper | P5 §4.1 |
| Minibatch | 12 | paper | P5 §4.1 |
| Training augmentation | horizontal flip and colour jittering | paper | P5 §4.1 |
| Optimiser | Adam, 800k iterations, lr 2e-3, halved at 700k and 750k | paper | P5 §4.1 |

**The ablation says which parts carry the result** (P5 Table 4, LPIPS 0.3646 for
the full model):

| Removed | LPIPS | What it shows |
|---|---|---|
| No spatial modulation (latent code only) | 0.550 | the spatial path is the largest single contributor |
| CS-SFT replaced by plain SFT | 0.387 | the channel split is worth ~0.023 LPIPS |
| No generative facial prior | 0.379 | the prior is worth ~0.015 |
| No pyramid restoration loss | 0.369 | the intermediate supervision is worth ~0.005 |

Useful ordering if the architecture has to be built in stages: spatial
modulation first, the channel split second, and the pyramid loss last.

### Not sourced yet

P1 reuses components from P2 and P3 by citation without restating their values.
P3 closed three, P2 closed the last two, and P5 closed the restoration
architecture. **All five papers have been read, and nothing is waiting on a
paper.**

What remains open is not a citation but a decision, recorded below: the two
pretrained networks P5 puts in the training gradient path.

**One ambiguity in P2, recorded rather than resolved.** P2's appendix states "We
do not use batch normalization, spectral normalization, attention mechanisms,
dropout, or pixelwise feature vector normalization in our networks", in a
paragraph otherwise about the classifier networks trained for the separability
metric. Read as a claim about the generator it would contradict Fig. 1, which
draws a `Normalize` block between the latent and the mapping network.

The reading taken is that these are two different operations: normalising the
input latent once, which Fig. 1 shows and which P3 §A.1's hypersphere supports,
versus applying P3 §4.2 after every convolution inside the generator, which is
what ProGAN did and what AdaIN replaced. Only the first is implemented, and
nothing normalises inside the synthesis path. If the reading is wrong, the
consequence is confined to `ogan/mapping.py`.

Every value the generator and discriminator need is now either cited to a paper
or explicitly marked `derived` or `ours` below. Nothing is waiting on a guess.

### Exists only in implementations — category `ours`

Needed in practice, absent from P1. Each has to be re-derived or tuned by us, and
the value recorded with the measurement that produced it:

| Component | Why it is `ours` |
|---|---|
| Scaling of `y` by `1/sqrt(H·W)` in the path length term | P1 Eq. 4 specifies only that `y` is a random image with normally distributed pixel intensities |
| Computing path length on a fraction of the minibatch | not in P1; matters because it decides whether the model fits in a small memory budget |
| Activation clamping under reduced precision | not in P1 |
| Choice of ε under reduced precision | P1 gives ε = 1e-8 without stating a precision regime |
| Demodulation `ε` = 1e-8 | P1 Eq. 3 calls it "a small constant to avoid numerical issues" and gives no value. 1e-8 matches the magnitude the same authors use for pixel normalisation (P3 §4.2) and for Adam (P1 App. B) |
| Number of style inputs | P2 counts "18 layers — two for each resolution" for a 1024² StyleGAN, where tRGB was a single unmodulated output layer. StyleGAN2 modulates a tRGB at every resolution, so the count does not carry over and P1 does not restate it. Ours: every style input takes its own entry in `w` — 23 at 512² — which is auditable at the cost of a wider `w` than an implementation that shares entries between a block's tRGB and its successor |
| Mapping-network learning-rate mechanism | P1 App. B states the effect — "100× lower learning rate" — and not how it is produced. Ours: store the parameter `1/lr_multiplier` larger and fold the multiplier into the runtime scale, leaving the effective weight unchanged at initialisation. Measured against the effect rather than the construction |
| The logistic loss formulas | P1 App. B names "non-saturating logistic loss [16] with R1 regularization [30]" and writes neither; nor does P2. Derived from the names plus a logit-valued discriminator: `softplus(-s_real) + softplus(s_fake)` and `softplus(-s_fake)`. The papers' own citations are Goodfellow et al. 2014 and Mescheder et al. 2018, neither read |
| Path length: averaging over synthesis layers | App. B says "an average of all individual layers", which reads either as the mean of the per-layer lengths or as the length implied by their mean square. The second is taken, keeping the quantity a length in the sense of Eq. 4. The readings coincide when layers carry comparable gradient and differ by Jensen otherwise |
| Reduced precision format: **bfloat16** | measured — `docs/spikes/2026-09-12-device-viability.md` §3. float16 gave non-finite gradients through the second derivative; bfloat16 did not |
| Minibatch stddev **group size** | P3 §3 computes the statistic over the whole minibatch and introduces no subgroup. Splitting the batch into groups of 4 is an implementation-only choice |
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

## Hyperparameters tuned on a dataset we will not use

A rule that follows from the corpus decision but is not the same rule, and is
easy to miss because it is not about licensing.

**No FFHQ image enters this project, at any stage.** That is the corpus rule, and
it is about data and lineage. Citing a number a paper measured is not using the
dataset it was measured on: reading `γ = 10` touches no image and enters no
weight.

But a hyperparameter fitted to a distribution is fitted to *that* distribution,
and this project will train on a different one. So values the papers tuned per
dataset are not inherited:

| Value | What is used instead |
|---|---|
| `γ_R1 = 10` (P1 App. B, FFHQ at 1024²) | P4's law, `0.0002 · N/M`, as the centre of a sweep |
| Training length (P1 raises it from 12M to 25M images for FFHQ) | to be measured |
| Mirror augmentation (P2 enables it for FFHQ, disables it for LSUN) | a corpus decision, not an inherited one |

Values that are architectural rather than tuned — the channel table, the
equalized learning rate, the resampling filter, the lazy intervals, the path
length weight, which is a closed-form function of resolution — carry over
untouched. The distinction is whether a number was *derived* or *searched for*.

**P5's loss weights are in this category.** `λ_l1 = 0.1`, `λ_per = 1`,
`λ_adv = 0.1`, `λ_local = 1`, `λ_fs = 200`, `λ_id = 10` were balanced against a
70,000-image FFHQ training set (P5 §4.1). `λ_fs = 200` in particular is two
orders of magnitude above its neighbours, which is the signature of a weight
fitted to a specific feature scale rather than derived. They are recorded as
`paper` because P5 states them, and they are flagged here because stating a
number is not the same as it being right for our distribution.

## Dependencies

The published artifact depends on `torch`, which is BSD-3-Clause, and on nothing
else. No third-party source is vendored: `ogan/` imports `torch` and the Python
standard library, and nothing further.

`requires-python = ">=3.11"` is a measured floor, not an aspiration: the suite was
run to completion on 3.11.15 and on 3.14.6 — 221 passed on both, in 22s either way.

`.python-version` pins development to **3.11**, for two reasons. It is the
interpreter the measurements in `docs/spikes/` were taken on, and those numbers
are a regression baseline that should not be compared across interpreters. And
developing at the floor is what keeps the floor honest: on 3.14 nothing would stop
syntax that silently breaks the version the metadata promises to support.

**One asymmetry is worth stating rather than discovering.** On macOS, `torch`
pulls only permissive dependencies. On Linux with CUDA, it additionally pulls
NVIDIA's runtime wheels — `nvidia-cuda-runtime-cu12` declares
`LicenseRef-NVIDIA-Proprietary`, and `nvidia-cublas-cu12` and `nvidia-cudnn-cu12`
declare no licence on PyPI at all.

Nothing about that is unusual, and it is not a defect in this project: it is true
of every PyTorch CUDA user. It is recorded because three things about it are
specific to us.

- It is a property of the **training environment**, not of the artifact. The
  deployment target is Core ML on Apple Silicon, which runs no NVIDIA code at any
  point.
- It never enters the weights. These are arithmetic kernels; their outputs are
  numbers.
- We do not redistribute them. If a bundled environment is ever shipped, NVIDIA's
  terms apply to that bundle and this entry is where to start reading.

## Pretrained networks inside the training gradient — unresolved

P5's objective calls two third-party pretrained networks on every training step:

| Network | Where | What it does |
|---|---|---|
| VGG-19 | `L_rec`, P5 Eq. 5 | perceptual loss, `{conv1..conv5}` before activation |
| ArcFace | `L_id`, P5 Eq. 8 | identity preserving loss in face-embedding space |

**This is a different situation from FID, and the difference matters.** FID runs
after training, outside the gradient path, and changes no weight. These two run
*inside* it. Their gradients shape every parameter the restorer ends up with.

Neither appears in the inference graph, so a shipped model contains no VGG and no
ArcFace code. But "the weights were trained against gradients from network X" is
a provenance statement this project cannot avoid making, and the honest position
is that it is unsettled — the same unsettled question the README already records
about training data and derivative works, arriving by a second route.

ArcFace is the sharper of the two. Its reference implementation ships models that
are commonly released for non-commercial use only, and P5 does not state which
weights it used. VGG-19 is ImageNet-trained, which the README already flags as a
declared dependency for evaluation.

Three options, none chosen yet, all deferred to when the restorer is actually
built:

1. Train without `L_id`, accept whatever identity drift results, and measure it.
2. Substitute a face embedding we can license, or train one on our own corpus.
3. Use them, and state plainly in the weights licence that the model was trained
   against gradients from both.

Recorded here so the decision is made deliberately rather than inherited by
copying an equation.

## Evaluation metrics

FID loads an ImageNet-trained InceptionV3. It sits outside the gradient path: no
weight of ours derives from it. It is declared here rather than omitted, and any
published result says which metric implementation produced it.
