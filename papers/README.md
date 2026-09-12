# Primary sources

The permitted sources for this project. Ideas, equations and measurements in a
paper are facts, and facts are not restricted by copyright. Reference
implementations are a different thing, and PROVENANCE.md rules them out.

Run `./fetch.sh` to download all five.

| Ref | Paper | arXiv | Status |
|---|---|---|---|
| **P1** | Analyzing and Improving the Image Quality of StyleGAN (StyleGAN2) | [1912.04958](https://arxiv.org/abs/1912.04958) | **read**, recorded in PROVENANCE.md |
| P2 | A Style-Based Generator Architecture for GANs (StyleGAN) | [1812.04948](https://arxiv.org/abs/1812.04948) | not read |
| P3 | Progressive Growing of GANs (ProGAN) | [1710.10196](https://arxiv.org/abs/1710.10196) | not read |
| P4 | Training GANs with Limited Data (StyleGAN2-ADA) | [2006.06676](https://arxiv.org/abs/2006.06676) | not read |
| P5 | Towards Real-World Blind Face Restoration with Generative Facial Prior (GFPGAN) | [2101.04061](https://arxiv.org/abs/2101.04061) | not read |

P1 cites P2 and P3 for components it reuses without restating them — the equalized
learning rate constant, the resampling filter coefficients, the minibatch standard
deviation parameters, the generator weight EMA decay, and the style mixing
probability. Those values are therefore **not yet sourced**, and PROVENANCE.md
marks them as such rather than guessing.

## Why the PDFs are not committed

arXiv's default submission licence grants arXiv a non-exclusive licence to
distribute. It does not grant us redistribution rights. Some papers carry CC
licences that would permit it; relying on that per paper, in a repository whose
subject is provenance, is not worth the inconsistency.

So: citations and a fetch script are committed, the files are not.
