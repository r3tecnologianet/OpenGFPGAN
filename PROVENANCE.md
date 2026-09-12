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

### Not sourced yet

P1 reuses these from P2 and P3 by citation, without restating the values. They are
**open**, and must not be filled in from any implementation:

| Component | Needs |
|---|---|
| Equalized learning rate — the gain constant and where it is applied | P3 |
| Resampling filter coefficients (P1 says only "bilinear filtering") | P2 / P3 |
| Minibatch standard deviation — group size, insertion point | P3 |
| Generator weight EMA — decay | P3 |
| Style mixing regularization — probability | P2 |
| Everything about adaptive discriminator augmentation | P4 |

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

What survives is narrower and more useful: **paper and code disagree in both
directions**, so neither "the code does X" nor a second-hand summary is a
substitute for reading the source. That is the rule this file enforces, and it
now rests on a measured disagreement rather than on a pattern that did not hold.

## Evaluation metrics

FID loads an ImageNet-trained InceptionV3. It sits outside the gradient path: no
weight of ours derives from it. It is declared here rather than omitted, and any
published result says which metric implementation produced it.
