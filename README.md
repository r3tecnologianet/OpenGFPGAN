# OpenGFPGAN

**An alternative to GFPGAN, without the licence restrictions.**

GFPGAN does the job well. Its restrictions are what stop people from using it:
the StyleGAN2 generator sits in its inference graph under NVIDIA's non-commercial
licence, and FFHQ sits upstream of its weights. The goal here is the same
capability — blind face restoration at comparable quality — with provenance that
holds up to being read.

**Status: nothing works yet.** This repository currently holds the licence, the
provenance discipline and the design record. No model, no weights, no training code.

> **Not affiliated with GFPGAN, TencentARC or NVIDIA.** OpenGFPGAN is not derived
> from GFPGAN's code or weights, and shares no lineage with them. The name states
> what it is an alternative to. Any trademarks are the property of their owners.

## Why this exists

Every published blind face restoration model traces back to training data or code
that forbids commercial use:

| Model | Blocker |
|---|---|
| GFPGAN | StyleGAN2 generator in the inference graph — NVIDIA Source Code License-NC |
| CodeFormer | NTU S-Lab License 1.0 — non-commercial |
| GPEN, RestoreFormer++, VQFR, DFDNet | FFHQ upstream (CC BY-NC-SA), no stated weights licence |

FFHQ sits upstream of effectively all of them, and none of them states a weights
licence at all. So there is no model in this family a commercial product can ship
with confidence.

OpenGFPGAN is an attempt at one:

- **No FFHQ**, at any stage, in any form
- **No vendored third-party code** in the inference graph
- **Clean-room implementation** from published papers only, with the limits of that claim stated — see [PROVENANCE.md](PROVENANCE.md)
- **A weights licence, stated affirmatively** — the thing nobody in this family does

## Licences

Three artifacts, three instruments:

| Artifact | Licence |
|---|---|
| Code | Apache 2.0 — see [LICENSE](LICENSE) |
| Weights | Apache 2.0, with an affirmative grant and a training-data statement (when weights exist) |
| Corpus | CDLA-Permissive-2.0, if images are redistributed; upstream terms are preserved per image |

Apache 2.0 rather than MIT for one specific reason: it carries an explicit patent
grant. Commercial adopters are the intended audience, and MIT is silent on patents.

**No indemnity is offered.** Both licences disclaim warranty, and that disclaimer
is meant literally. Clean provenance is not a legal opinion.

## What is not resolved

Stated up front, because a confident wrong answer is worse than a gap:

- **Patents are a separate axis from copyright and contract.** A clean-room
  implementation removes copyright and contract exposure. It does not remove patent
  exposure, and NVIDIA has filed on style-based generator architectures.
- Whether trained weights are a derivative work of training data is genuinely
  unsettled law.
- Evaluation metrics (FID, and LPIPS if ever used) load ImageNet-trained networks.
  They sit outside the gradient path and are declared, not hidden.

## Documentation

- [CHANGELOG.md](CHANGELOG.md) — what exists, what was found, and what was withdrawn
- [PROVENANCE.md](PROVENANCE.md) — the per-component source record and the review rule
- [papers/README.md](papers/README.md) — the five primary sources, and why the PDFs are not committed
- [CONTRIBUTING.md](CONTRIBUTING.md) — read before writing any code, the rule is unusual
