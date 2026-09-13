# PD12M face yield: the order of magnitude

**Question:** how many faces does PD12M hold at a size usable for 512² training?

**Answer: a few hundred thousand, and it is an upper bound.** Not a final count.

## Stage 1 — all 12,400,094 rows, from metadata only

```
python docs/spikes/corpus_yield.py metadata
```

```
total 12,400,094   at least 512px: 11,439,787 (0.9226)
```

| Source | Images | >= 512px |
|---|---|---|
| Wikimedia Commons | 7,011,171 | 0.919 |
| iNaturalist | 2,531,746 | 0.990 |
| Rijksmuseum | 163,198 | 0.999 |
| Metropolitan Museum of Art | 149,458 | 0.947 |

**The size filter does not filter.** 92.3% of PD12M is already large enough to
hold a 512px face, so the two-stage design this spike was built around does not
buy what it was supposed to buy. The yield is decided entirely by detection.

## Stage 2 — 200 images, seed 0, 3 shards

A validation run, not the measurement. 200/200 downloaded, ~5s each.

```
exactly one face  15  (7.5% of downloaded)

 threshold   faces/1000   expected total
       128        115.0        1,315,628   (lower bound)
       256         70.0          800,817   (lower bound)
       384         60.0          686,414   (lower bound)
       512         35.0          400,408
       768         25.0          286,006
```

Rows below 512 are lower bounds: stage 1 discarded images that could have held a
smaller face.

| Source | Sampled | Faces >= 512px |
|---|---|---|
| Wikimedia Commons | 109 | 6 |
| iNaturalist | 48 | 0 |
| everything else | 43 | 1 |

**iNaturalist is 2.5M images and returned no faces at all.** A fifth of the
dataset is species photography that passes the size filter intact and
contributes nothing.

## What this does and does not establish

7 faces at the 512px threshold carries roughly +/-38% relative uncertainty, so
the honest reading is **250k-550k**, not 400k.

And it is a **ceiling, not a count**. Vision registers a face in a blurred
engraving, a damaged scan or a half-tone newspaper reproduction, and none of
those are training data. Measuring quality is a separate gate that has not been
run.

## Also found

PD12M has rows with a null `source`. Two appeared in a 200-image sample. The
per-source table crashed on them until `a39fe31`.
