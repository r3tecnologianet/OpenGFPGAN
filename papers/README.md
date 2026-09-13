# Primary sources

The permitted sources for this project. Ideas, equations and measurements in a
paper are facts, and facts are not restricted by copyright. Reference
implementations are a different thing, and PROVENANCE.md rules them out.

Run `./fetch.sh` to download all five.

| Ref | Paper | arXiv | Status |
|---|---|---|---|
| **P1** | Analyzing and Improving the Image Quality of StyleGAN (StyleGAN2) | [1912.04958](https://arxiv.org/abs/1912.04958) | **read**, recorded in PROVENANCE.md |
| **P2** | A Style-Based Generator Architecture for GANs (StyleGAN) | [1812.04948](https://arxiv.org/abs/1812.04948) | **read**, recorded in PROVENANCE.md |
| **P3** | Progressive Growing of GANs (ProGAN) | [1710.10196](https://arxiv.org/abs/1710.10196) | **read**, recorded in PROVENANCE.md |
| **P4** | Training GANs with Limited Data (StyleGAN2-ADA) | [2006.06676](https://arxiv.org/abs/2006.06676) | **read**, recorded in PROVENANCE.md |
| **P5** | Towards Real-World Blind Face Restoration with Generative Facial Prior (GFPGAN) | [2101.04061](https://arxiv.org/abs/2101.04061) | **read**, recorded in PROVENANCE.md |

**All five have been read.** Between them they close every value the generator,
the discriminator, the losses, the augmentation schedule and the restoration
architecture need. Nothing in this project is waiting on a paper.

P4 supplies the R1 weight's scaling law, which P1 leaves as a per-dataset tuning
problem. P5 states its full objective and every loss weight, which is more than
P1 does for StyleGAN2 — and it also puts two pretrained third-party networks
inside the training gradient, which PROVENANCE.md records as unresolved.

## Why the PDFs are not committed

arXiv's default submission licence grants arXiv a non-exclusive licence to
distribute. It does not grant us redistribution rights. Some papers carry CC
licences that would permit it; relying on that per paper, in a repository whose
subject is provenance, is not worth the inconsistency.

So: citations and a fetch script are committed, the files are not.
