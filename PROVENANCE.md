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

| Component | Value | Category | Source |
|---|---|---|---|
| *(empty — filled as the specification is derived)* | | | |

## Known contamination in the design record

An architecture document written before this rule existed cites
`rosinality/stylegan2-pytorch/model.py` and `basicsr.archs.stylegan2_arch` in its
own reference list. It was written by reading them, so it is **not** a clean-room
source. It is held locally, is deliberately not published in this repository, and
may never be cited here.

Its audit is the empirical reason the paper/code boundary is treated as
load-bearing: it contains three numerical errors and seven omissions, and **every
one of them falls in a value that exists only in code and not in the papers** —
the LeakyReLU gain, the path-length weight, and the path-length interval. Values
traceable to a paper were correct. The boundary is therefore detectable in
practice, which is what makes it enforceable as a rule.

## Evaluation metrics

FID loads an ImageNet-trained InceptionV3. It sits outside the gradient path: no
weight of ours derives from it. It is declared here rather than omitted, and any
published result says which metric implementation produced it.
