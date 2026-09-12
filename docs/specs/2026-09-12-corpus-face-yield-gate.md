# Gate zero: how many usable faces does PD12M hold?

**Status:** design, not yet implemented.

## The question

PD12M is 12.4M public domain images. This project intends to build a face corpus
from it. **Nobody has published how many faces it contains**, and no amount of
architecture work answers that question.

This spike measures it.

## Why this is gate zero

Every other open task is downstream of the answer:

- The adaptive augmentation in `ogan/augment.py` exists to fight overfitting on a
  small dataset. Whether its missing third transform category is urgent or
  cosmetic depends on the corpus size.
- The training resolution depends on how many faces arrive with enough pixels.
- The proportion of synthetic images needed to supplement PD12M is exactly the
  gap between this number and whatever target is chosen.

The information dependency runs one way. Measuring the corpus changes the
priority of the rest; the rest does not change what the corpus contains.

The precondition for running it has been met: the training assembly converges
(`docs/spikes/2026-09-12-convergence-smoke-run.md`), so a corpus would have
something to train.

## No target is fixed here

This spec deliberately does not state how many faces the project needs. The
measurement produces a curve; a target is chosen afterwards, with the number in
hand. Fixing a target first would only bias which thresholds get reported.

## What it produces

A table of **faces per thousand images as a function of minimum face size**,
extrapolated to an absolute count over the full 12.4M, and the same table broken
down by source organisation.

## The dataset

| Property | Value |
|---|---|
| Size | 12.4M image-caption pairs |
| Metadata | Parquet, on Hugging Face (`Spawning/PD12M`) |
| Images | Hosted in the dataset's own S3 bucket, not scattered web URLs |
| Licence | CDLA-Permissive-2.0 |
| Sources | Wikimedia Commons, cultural heritage organisations, iNaturalist |

Metadata schema: `id`, `url`, `caption`, `width`, `height`, `mime_type`, `hash`,
`license`, `source`.

Two consequences. The images live in a stable bucket, so a seeded sample is
reproducible rather than decaying as links die. And `hash`, `license`, `url` and
`source` are precisely the four fields `CONTRIBUTING.md` requires per corpus
image — the per-image provenance requirement is satisfiable directly from
metadata, with no separate bookkeeping.

iNaturalist is a species-photography archive. A meaningful fraction of the
dataset is expected to contain no human face at all, and stage 1 quantifies that
rather than assuming it.

## Stage 1 — metadata, zero downloads

Read the parquet metadata, projecting only `width`, `height` and `source`.
Parquet prunes columns on read, so this is tens of megabytes rather than the
~2.5 GB the full table would be.

Over all 12.4M rows, compute:

- the fraction with `min(width, height) >= 512`
- the same fraction per `source`

A face of 512px cannot come out of an image smaller than 512px on its short side.
This bound is exact and costs nothing, and it is what makes stage 2 affordable.

## Stage 2 — sample and detect

Draw **5,000 ids at random from the `min(width, height) >= 512` subset**, with a
fixed seed recorded in the spike write-up. Download each from its `url`. Run
`VNDetectFaceRectanglesRequest` on each image.

Write one JSONL line per image:

```json
{"id": "...", "source": "...", "width": 0, "height": 0,
 "faces": [{"w": 0, "h": 0}]}
```

Face box dimensions are recorded **in pixels**, not in Vision's normalised
coordinates — the normalised form is unusable across images of different sizes,
and converting at write time means the analysis never has to re-derive it.

A face box is measured as `min(w, h)`, matching how the image filter measures an
image. Mixing the two conventions would make the thresholds incomparable.

Sample size is chosen for the decision being made, not for precision. If one in
two hundred images in the filtered subset yields a large face, 5,000 images return
roughly 25 of them: enough to separate "a fraction of a percent" from "a tenth of
that" from "essentially none", which is the distinction that changes the plan. It
is not enough to quote a yield to two significant figures, and the write-up will
not.

## Stage 3 — the number

```
yield(t) = P(min-dim >= 512) x (faces with box >= t per sampled image)
count(t) = yield(t) x 12_400_000
```

Reported for t in {128, 256, 384, 512, 768}, overall and per source, where a face
counts at threshold t when `min(w, h) >= t`.

**The formula is exact only for t >= 512.** Stage 1 discards images under 512px on
the short side, and such an image can still hold a 256px face. So the rows below
512 count only the small faces that occur inside large images, and are lower
bounds on the full-dataset yield at those thresholds rather than estimates of it.
They are reported anyway, because the shape of the size distribution inside the
retained subset is what says whether faces cluster near the usable end or trail
off below it. The write-up labels them as bounds.

Also reported: the fraction of sampled images containing **exactly one** face.
Portraits and crowd photographs are both "images with faces" and only one of them
is straightforward to align.

## What this spike deliberately does not do

- No alignment, no cropping, no corpus construction
- No quality, blur, or compression scoring
- No download beyond the 5,000-image sample
- Nothing added to `ogan/`

## The number is an upper bound

Vision detecting a 512px face in a blurred engraving, a damaged scan, or a
half-tone newspaper reproduction counts as a hit here. Those are not training
data.

**So the result is an upper bound on usable faces, not a count of them.** The
write-up must say this in its own conclusion, not only in its method. Quality
filtering is a separate gate and is out of scope; treating this number as final
would be the failure this project is built to avoid.

## Code and dependencies

`docs/spikes/corpus_yield.py` plus `docs/spikes/2026-09-12-corpus-face-yield.md`
for the result, following the existing `device_viability.py` pattern.

New dependencies, in a separate `corpus` extra — **not** in the runtime
dependency list:

| Package | Licence | Why |
|---|---|---|
| `pyarrow` | Apache 2.0 | read the parquet metadata |
| `pyobjc-framework-Vision` | MIT | reach the system face detector |

Downloads use `urllib` from the standard library. `ogan/` keeps its single
dependency on `torch`.

No test file, consistent with `device_viability.py`. A spike is a measurement,
not a component. The script prints its inputs alongside its outputs so the
arithmetic in stage 3 can be rechecked by hand.

## Provenance

**The face detector is a selection tool and never enters the model.** It does not
appear in the inference graph, contributes no gradient, and leaves no trace in any
weight. It belongs on the same shelf as the ImageNet network FID loads, which
`README.md` already declares.

Apple Vision was chosen over the alternatives on that basis. It is an operating
system service: nothing is redistributed and no dataset licence is inherited. The
cross-platform detectors are not equivalent — YuNet's weights are trained on
WIDER FACE, which is research-only, and several InsightFace models are explicitly
non-commercial. Adopting one would import a restriction into the project purely in
order to measure something, which is the exact pattern this project exists to
avoid.

The cost is that the measurement is macOS-only. This is acceptable: the
deployment target is Apple Silicon, the command and seed are recorded, and anyone
on macOS can rerun it.

PD12M itself is CDLA-Permissive-2.0, which is already the corpus licence named in
`README.md`.

## Done when

`docs/spikes/2026-09-12-corpus-face-yield.md` records the stage 1 fractions, the
stage 3 table, the seed, the command, and a conclusion that states the upper-bound
caveat.
