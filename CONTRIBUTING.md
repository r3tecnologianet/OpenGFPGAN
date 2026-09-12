# Contributing

The rule here is unusual, so read this before writing code.

## Do not read the reference implementations

OpenGFPGAN is a clean-room implementation. Its value to a commercial user rests
entirely on that being true, and it stops being true the moment someone ports a
line from NVIDIA's StyleGAN2, `rosinality/stylegan2-pytorch`, BasicSR or GFPGAN.

[PROVENANCE.md](PROVENANCE.md) lists the permitted papers and the forbidden
repositories. Work from the papers.

If you have already read those implementations closely, you can still contribute —
to the corpus tooling, the tests, the export path, the documentation. Just not to
`ogan/`. Say so in your pull request; it is a normal thing to declare, not a
disqualification.

## Every constant carries a source

A pull request that introduces a hyperparameter states where it came from, using
the three categories in PROVENANCE.md: **paper**, **derived**, or **ours**.

"It is what the other implementation uses" is not a source. It is the specific
thing this project cannot accept.

## Corpus contributions

Images enter the corpus only with per-image provenance: source, licence, origin
URL, hash. No exceptions, including for images that are obviously fine.

The corpus must contain **no FFHQ image, in any form** — not the aligned release,
not a subset, not a derivative, not a crop that came from one. This is the
project's central claim.

## Tests

Numerical components ship with numerical tests: a stated tolerance and the paper
equation being verified. A test that only checks that code runs is not a test of a
clean-room port.

## Language

Code, comments, commit messages and all documentation — including design specs
under `docs/specs/` — are written in **English**. The intended audience is
international, and the provenance record has to be readable by anyone auditing it.
