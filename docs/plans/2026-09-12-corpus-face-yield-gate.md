# Corpus Face Yield Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how many usable faces PD12M holds, so the corpus decision stops being an assumption.

**Architecture:** Two stages, kept apart because they have different costs. Stage 1 reads only the parquet metadata columns `width`, `height` and `source` over all 12.4M rows and computes what fraction of the dataset could hold a 512px face — no images downloaded. Stage 2 draws a seeded 5,000-image sample from that subset, downloads it, and runs the macOS Vision face detector over it. Stage 3 composes the two into a yield table. The pure arithmetic and the sampling are unit-tested; the network and the OS detector are not.

**Tech Stack:** Python 3.11, `pyarrow` for parquet, `pyobjc-framework-Vision` for the system face detector, `urllib` from the standard library for downloads. Both new packages sit in a `corpus` extra, never in the runtime dependency list.

**Spec:** `docs/specs/2026-09-12-corpus-face-yield-gate.md`

---

## Deviation from the spec, declared

The spec says "No test file, consistent with `device_viability.py`." **This plan adds
one anyway**, covering three pure functions and nothing else.

The reason is evidence, not preference: reviewing the spec turned up two arithmetic
errors in the spec's own formula — a face box compared on the wrong dimension, and
a composition that is only valid for thresholds at or above 512. Both would have
produced a wrong headline number that looked entirely plausible. Arithmetic that has
already failed review twice gets a test.

Nothing else is tested. Downloads, parquet IO and the Vision call are left alone.

---

## File structure

| File | Responsibility |
|---|---|
| `docs/spikes/corpus_yield.py` | Create. The whole spike: stage 1, stage 2, stage 3, CLI. |
| `tests/test_corpus_yield.py` | Create. The three pure functions only. |
| `pyproject.toml` | Modify. Add the `corpus` optional-dependency group. |
| `docs/spikes/2026-09-12-corpus-face-yield.md` | Create in the final task. The result. |

**One structural rule for `corpus_yield.py`:** top-level imports are standard library
only. `pyarrow` and `Vision` are imported *inside* the functions that use them. This
is what lets `tests/test_corpus_yield.py` import the module and exercise the
arithmetic in the plain dev environment, without the `corpus` extra installed.

---

## Task 1: The yield arithmetic

The three pure functions, written first because everything else feeds them.

**Files:**
- Create: `docs/spikes/corpus_yield.py`
- Create: `tests/test_corpus_yield.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_corpus_yield.py`:

```python
"""The spike's arithmetic, which is the part that can be wrong quietly.

The spike itself is a measurement and is not otherwise tested. These three
functions are, because a face measured on the wrong dimension or a composition
applied outside its valid range produces a plausible-looking wrong answer.
"""

import importlib.util
import pathlib

_PATH = pathlib.Path(__file__).resolve().parents[1] / "docs" / "spikes" / "corpus_yield.py"
_spec = importlib.util.spec_from_file_location("corpus_yield", _PATH)
corpus_yield = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(corpus_yield)


def record(*faces):
    return {"id": "x", "source": "s", "width": 1024, "height": 1024,
            "faces": [{"w": w, "h": h} for w, h in faces]}


def test_a_face_is_measured_on_its_short_side():
    """A 600x200 box is not a 512px face. Measuring on the long side says it is."""
    wide = [record((600, 200))]
    assert corpus_yield.faces_at_least(wide, 512) == 0
    assert corpus_yield.faces_at_least(wide, 200) == 1


def test_every_face_in_an_image_counts():
    records = [record((600, 600), (520, 700)), record((100, 100))]
    assert corpus_yield.faces_at_least(records, 512) == 2
    assert corpus_yield.faces_at_least(records, 100) == 3


def test_expected_total_composes_both_stages():
    """Stage 1's fraction multiplies stage 2's rate. Dropping either inflates it."""
    records = [record((600, 600)), record(), record(), record()]
    # 1 face over 4 sampled images = 0.25 faces/image, in 40% of 1_000_000 images.
    assert corpus_yield.expected_total(0.4, records, 512, 1_000_000) == 100_000.0


def test_expected_total_is_zero_when_nothing_qualifies():
    assert corpus_yield.expected_total(0.4, [record((10, 10))], 512, 1_000_000) == 0.0


def test_the_sample_is_reproducible_from_its_seed():
    ids = [f"id{n}" for n in range(1000)]
    assert corpus_yield.sample_ids(ids, 20, seed=0) == corpus_yield.sample_ids(ids, 20, seed=0)
    assert corpus_yield.sample_ids(ids, 20, seed=0) != corpus_yield.sample_ids(ids, 20, seed=1)
    assert len(corpus_yield.sample_ids(ids, 20, seed=0)) == 20


def test_the_sample_does_not_exceed_the_population():
    ids = ["a", "b", "c"]
    assert sorted(corpus_yield.sample_ids(ids, 10, seed=0)) == ["a", "b", "c"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_corpus_yield.py -q`

Expected: collection error, `FileNotFoundError` for `docs/spikes/corpus_yield.py`.

- [ ] **Step 3: Write the module with only these three functions**

Create `docs/spikes/corpus_yield.py`:

```python
"""Gate zero: how many usable faces does PD12M hold?

    python docs/spikes/corpus_yield.py metadata   # stage 1, no downloads
    python docs/spikes/corpus_yield.py sample     # stage 2, downloads and detects
    python docs/spikes/corpus_yield.py report     # stage 3, the table

PD12M is 12.4M public domain images and nobody has published its face yield.
Every open decision in this project is downstream of that number: whether the
missing geometric augmentations are urgent, what resolution is reachable, and how
much synthetic data has to make up the difference.

See docs/specs/2026-09-12-corpus-face-yield-gate.md.

The result is an UPPER BOUND. The detector counts a face in a blurred engraving
or a damaged scan, and those are not training data. Quality is a separate gate.

Imports of pyarrow and Vision are deliberately inside the functions that use
them, so the arithmetic below stays importable without the `corpus` extra.
"""

import random

DATASET_SIZE = 12_400_000
MIN_IMAGE_DIM = 512
THRESHOLDS = (128, 256, 384, 512, 768)


def faces_at_least(records, threshold):
    """Total faces whose short side is at least `threshold`, across all records.

    Short side, not long side: a 600x200 box is a sliver, not a 512px face. The
    image filter measures images the same way, and mixing the two conventions
    would make the thresholds incomparable.
    """
    return sum(
        1
        for record in records
        for face in record["faces"]
        if min(face["w"], face["h"]) >= threshold
    )


def expected_total(p_large, records, threshold, dataset_size=DATASET_SIZE):
    """Faces expected across the whole dataset at `threshold`.

    Composes the two measured quantities: `p_large` is stage 1's fraction of
    images big enough to hold such a face, and the rate below is stage 2's faces
    per sampled image. Dropping either term inflates the answer.

    Exact only for `threshold >= MIN_IMAGE_DIM`. Below that, stage 1 has already
    discarded images that could have held a qualifying face, so the result is a
    lower bound. `report` labels those rows.
    """
    if not records:
        return 0.0
    per_image = faces_at_least(records, threshold) / len(records)
    return p_large * per_image * dataset_size


def sample_ids(ids, count, seed):
    """A reproducible sample. The seed goes in the write-up with the result."""
    rng = random.Random(seed)
    return rng.sample(ids, min(count, len(ids)))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_corpus_yield.py -q`

Expected: `6 passed`.

- [ ] **Step 5: Run the whole suite and the linter**

Run: `.venv/bin/pytest -q && .venv/bin/ruff check .`

Expected: `227 passed`, then `All checks passed!`.

- [ ] **Step 6: Commit**

```bash
git add docs/spikes/corpus_yield.py tests/test_corpus_yield.py
git commit -m "spike: the yield arithmetic, and a face is measured on its short side"
```

---

## Task 2: Add the corpus extra

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the optional-dependency group**

In `pyproject.toml`, the block currently reads:

```toml
[project.optional-dependencies]
dev = ["pytest>=8", "ruff>=0.6"]
```

Replace it with:

```toml
[project.optional-dependencies]
dev = ["pytest>=8", "ruff>=0.6"]
# Corpus tooling only. Never imported from ogan/, which keeps its single
# dependency on torch. pyarrow is Apache 2.0; pyobjc is MIT.
corpus = ["pyarrow>=17", "pyobjc-framework-Vision>=10"]
```

- [ ] **Step 2: Install it**

Run: `.venv/bin/pip install -e '.[dev,corpus]'`

Expected: installs `pyarrow` and the `pyobjc` framework packages.

- [ ] **Step 3: Verify ogan/ is untouched by the new packages**

Run: `.venv/bin/python -c "import ogan, sys; print([m for m in sys.modules if 'pyarrow' in m or 'Vision' in m])"`

Expected: `[]` — importing `ogan` pulls in neither.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build: a corpus extra, so ogan/ keeps its single dependency"
```

---

## Task 3: Stage 1, the metadata scan

No image is downloaded in this task.

**Files:**
- Modify: `docs/spikes/corpus_yield.py`

- [ ] **Step 1: Find the real parquet paths**

The metadata file layout has to be observed, not guessed.

Run:

```bash
curl -s "https://huggingface.co/api/datasets/Spawning/PD12M/tree/main" \
  | python3 -c "import json,sys; [print(e['type'], e['path']) for e in json.load(sys.stdin)]"
```

Then list whichever directory holds the metadata, for example:

```bash
curl -s "https://huggingface.co/api/datasets/Spawning/PD12M/tree/main/metadata" \
  | python3 -c "import json,sys; [print(e['path'], e.get('size')) for e in json.load(sys.stdin)]"
```

Record the actual directory name and file count. If the directory is not called
`metadata`, use what the listing shows in Step 2 — do not keep the placeholder name.

- [ ] **Step 2: Write the metadata stage**

Append to `docs/spikes/corpus_yield.py`, above the CLI (which Task 5 adds):

```python
HF_REPO = "Spawning/PD12M"
HF_TREE = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main"
HF_FILE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
METADATA_DIR = "metadata"  # replace with whatever Task 3 Step 1 actually listed


def metadata_files():
    """The parquet shard paths, read from the repository listing rather than guessed."""
    import json
    import urllib.request

    with urllib.request.urlopen(f"{HF_TREE}/{METADATA_DIR}", timeout=60) as response:
        entries = json.load(response)
    return sorted(e["path"] for e in entries if e["path"].endswith(".parquet"))


def scan_metadata(paths, min_dim=MIN_IMAGE_DIM):
    """Stage 1: how much of the dataset could hold a face of `min_dim` pixels?

    Reads three columns. Parquet prunes the rest on read, so this moves tens of
    megabytes rather than the whole table.

    Returns (total_rows, large_rows, {source: [total, large]}).
    """
    import pyarrow.parquet as pq

    total = 0
    large = 0
    by_source = {}

    for path in paths:
        table = pq.read_table(
            f"{HF_FILE}/{path}", columns=["width", "height", "source"]
        )
        widths = table.column("width").to_pylist()
        heights = table.column("height").to_pylist()
        sources = table.column("source").to_pylist()

        for width, height, source in zip(widths, heights, sources):
            counts = by_source.setdefault(source, [0, 0])
            total += 1
            counts[0] += 1
            if width is not None and height is not None and min(width, height) >= min_dim:
                large += 1
                counts[1] += 1

        print(f"  {path}: {total} rows, {large} at least {min_dim}px", flush=True)

    return total, large, by_source
```

- [ ] **Step 3: Run stage 1 on a single shard first**

Before the full scan, confirm the reader works and see what one shard costs:

```bash
.venv/bin/python -c "
import importlib.util, pathlib, time
s = importlib.util.spec_from_file_location('cy', 'docs/spikes/corpus_yield.py')
cy = importlib.util.module_from_spec(s); s.loader.exec_module(cy)
files = cy.metadata_files()
print(len(files), 'shards; first is', files[0])
started = time.perf_counter()
print(cy.scan_metadata(files[:1])[:2], f'{time.perf_counter()-started:.0f}s')
"
```

Expected: a shard count, then `(rows, large)` for one shard and its wall time.
Multiply by the shard count to decide whether the full scan is minutes or hours.
If a shard takes more than about two minutes, stop and report the figure rather
than launching the full scan.

- [ ] **Step 4: Commit**

```bash
git add docs/spikes/corpus_yield.py
git commit -m "spike: stage one, the fraction of PD12M that could hold a 512px face"
```

---

## Task 4: Stage 2, download and detect

**Files:**
- Modify: `docs/spikes/corpus_yield.py`

- [ ] **Step 1: Verify the Vision API on one image before wiring it in**

This API is being used from Python through a bridge, and the call shape should be
confirmed against a real image rather than assumed.

```bash
.venv/bin/python -c "
import urllib.request, Vision, Quartz
url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Altes_Rathaus_Bamberg.jpg/640px-Altes_Rathaus_Bamberg.jpg'
data = urllib.request.urlopen(url, timeout=60).read()
handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
request = Vision.VNDetectFaceRectanglesRequest.alloc().init()
ok, err = handler.performRequests_error_([request], None)
print('ok', ok, 'err', err)
print('results', request.results())
"
```

Expected: `ok True`, `err None`, and an empty or short results list (this image is
a building, so zero faces is the correct answer — the point is that the call
returns without raising).

If `performRequests_error_` has a different arity or the constructor differs, fix
the calls in Step 2 to match what worked here, and note the correction in the
commit message.

- [ ] **Step 2: Write the download and detection stage**

Append to `docs/spikes/corpus_yield.py`:

```python
USER_AGENT = "OpenGFPGAN-corpus-gate/0.0 (+https://github.com/r3tecnologianet/OpenGFPGAN)"


def download(url, timeout=60):
    """Fetch one image. Returns bytes, or None if it could not be fetched."""
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def detect_faces(data, width, height):
    """Face boxes in pixels, via the macOS Vision framework.

    Vision returns normalised coordinates, which are meaningless across images of
    different sizes. They are converted here, once, so no later stage has to
    remember to do it.

    The detector never enters the model: it selects images and contributes no
    gradient and no weight. See PROVENANCE.md.
    """
    import Vision

    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
    request = Vision.VNDetectFaceRectanglesRequest.alloc().init()
    handler.performRequests_error_([request], None)

    faces = []
    for observation in request.results() or []:
        box = observation.boundingBox()
        faces.append({
            "w": round(box.size.width * width),
            "h": round(box.size.height * height),
        })
    return faces


def sample_and_detect(rows, out_path, seed):
    """Stage 2: download the sample and record what the detector found.

    `rows` is a list of dicts with id, url, width, height, source. Writes one
    JSONL line per image, including failures — a download that did not happen is
    data, and dropping it would quietly shrink the denominator.
    """
    import json

    with open(out_path, "w") as out:
        for index, row in enumerate(rows, start=1):
            data = download(row["url"])
            record = {
                "id": row["id"],
                "source": row["source"],
                "width": row["width"],
                "height": row["height"],
                "downloaded": data is not None,
                "faces": detect_faces(data, row["width"], row["height"]) if data else [],
            }
            out.write(json.dumps(record) + "\n")
            out.flush()

            if index % 100 == 0:
                print(f"  {index}/{len(rows)}", flush=True)

    print(f"wrote {out_path}  seed={seed}", flush=True)
```

- [ ] **Step 3: Run it on twenty images end to end**

```bash
.venv/bin/python -c "
import importlib.util, json
s = importlib.util.spec_from_file_location('cy', 'docs/spikes/corpus_yield.py')
cy = importlib.util.module_from_spec(s); s.loader.exec_module(cy)
import pyarrow.parquet as pq
path = cy.metadata_files()[0]
t = pq.read_table(f'{cy.HF_FILE}/{path}', columns=['id','url','width','height','source']).to_pylist()
rows = [r for r in t if r['width'] and r['height'] and min(r['width'], r['height']) >= 512][:20]
cy.sample_and_detect(rows, '/tmp/gate-smoke.jsonl', seed=0)
print(open('/tmp/gate-smoke.jsonl').read()[:600])
"
```

Expected: twenty JSONL lines. Check that `downloaded` is mostly `true` and that
at least one line has a non-empty `faces` list with plausible pixel sizes. If
every line has zero faces, the detector is not working — stop and diagnose rather
than launching the 5,000-image run.

- [ ] **Step 4: Commit**

```bash
git add docs/spikes/corpus_yield.py
git commit -m "spike: stage two, download the sample and convert Vision's boxes to pixels"
```

---

## Task 5: Stage 3 and the CLI

**Files:**
- Modify: `docs/spikes/corpus_yield.py`

- [ ] **Step 1: Write the report and the command line**

Append to `docs/spikes/corpus_yield.py`:

```python
def report(p_large, records, dataset_size=DATASET_SIZE):
    """Stage 3: the table the gate exists to produce."""
    attempted = len(records)
    usable = [r for r in records if r["downloaded"]]
    single = sum(1 for r in usable if len(r["faces"]) == 1)

    print(f"sampled     {attempted}")
    print(f"downloaded  {len(usable)}  ({len(usable) / attempted:.1%})")
    print(f"exactly one face  {single}  ({single / len(usable):.1%} of downloaded)")
    print(f"P(min-dim >= {MIN_IMAGE_DIM}px)  {p_large:.4f}")
    print()
    print(f"{'threshold':>10}  {'faces/1000':>11}  {'expected total':>15}")

    for threshold in THRESHOLDS:
        per_thousand = 1000 * faces_at_least(usable, threshold) / len(usable)
        total = expected_total(p_large, usable, threshold, dataset_size)
        bound = "" if threshold >= MIN_IMAGE_DIM else "   (lower bound)"
        print(f"{threshold:>10}  {per_thousand:>11.1f}  {total:>15,.0f}{bound}")

    by_source = {}
    for record in usable:
        counts = by_source.setdefault(record["source"], [0, 0])
        counts[0] += 1
        counts[1] += sum(
            1 for f in record["faces"] if min(f["w"], f["h"]) >= MIN_IMAGE_DIM
        )

    print()
    print(f"at {MIN_IMAGE_DIM}px by source (small samples, indicative only)")
    print(f"{'source':<40}  {'sampled':>8}  {'faces':>7}")
    for source, (sampled, faces) in sorted(by_source.items(), key=lambda kv: -kv[1][1]):
        print(f"{source[:40]:<40}  {sampled:>8}  {faces:>7}")


def main(argv):
    import json

    command = argv[1] if len(argv) > 1 else "report"

    if command == "metadata":
        files = metadata_files()
        print(f"{len(files)} parquet shards")
        total, large, by_source = scan_metadata(files)
        print(f"\ntotal {total:,}   at least {MIN_IMAGE_DIM}px: {large:,} ({large / total:.4f})")
        print(f"\n{'source':<40}  {'total':>10}  {'large':>10}  {'fraction':>9}")
        for source, (count, big) in sorted(by_source.items(), key=lambda kv: -kv[1][1]):
            print(f"{source[:40]:<40}  {count:>10,}  {big:>10,}  {big / count:>9.4f}")
        return 0

    if command == "report":
        p_large = float(argv[2])
        records = [json.loads(line) for line in open(argv[3])]
        report(p_large, records)
        return 0

    print(__doc__)
    return 1


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv))
```

Note: `sample` is driven from the metadata stage in Task 6 rather than being a
standalone subcommand, because it needs the id list that stage 1 produces.

- [ ] **Step 2: Verify the report against the twenty-image file**

```bash
.venv/bin/python docs/spikes/corpus_yield.py report 0.4 /tmp/gate-smoke.jsonl
```

Expected: the header lines, then five threshold rows, with the 128 and 256 rows
carrying `(lower bound)` and the 512 and 768 rows not. The numbers will be noise
at twenty images — this step checks the table renders and labels correctly, not
the values.

- [ ] **Step 3: Run the linter**

Run: `.venv/bin/ruff check .`

Expected: `All checks passed!`

- [ ] **Step 4: Commit**

```bash
git add docs/spikes/corpus_yield.py
git commit -m "spike: stage three, and the sub-512 rows are labelled as bounds"
```

---

## Task 6: Run the gate and write it up

This is the task the other five exist for.

**Files:**
- Create: `docs/spikes/2026-09-12-corpus-face-yield.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Run stage 1 over the full dataset**

```bash
.venv/bin/python docs/spikes/corpus_yield.py metadata | tee /tmp/gate-metadata.txt
```

Expected: the per-shard progress, the overall fraction, and the per-source table.
Keep the output — the per-source breakdown goes in the write-up.

- [ ] **Step 2: Draw and detect the 5,000-image sample**

```bash
.venv/bin/python -c "
import importlib.util, pyarrow.parquet as pq
s = importlib.util.spec_from_file_location('cy', 'docs/spikes/corpus_yield.py')
cy = importlib.util.module_from_spec(s); s.loader.exec_module(cy)

rows = []
for path in cy.metadata_files():
    t = pq.read_table(f'{cy.HF_FILE}/{path}',
                      columns=['id','url','width','height','source']).to_pylist()
    rows += [r for r in t
             if r['width'] and r['height'] and min(r['width'], r['height']) >= cy.MIN_IMAGE_DIM]
    print(len(rows), 'eligible so far', flush=True)

chosen = set(cy.sample_ids([r['id'] for r in rows], 5000, seed=0))
cy.sample_and_detect([r for r in rows if r['id'] in chosen], 'docs/spikes/pd12m-sample.jsonl', seed=0)
"
```

Expected: a growing eligible count, then progress every 100 images. This is the
long step. The JSONL is written incrementally and flushed, so an interruption
loses only the tail.

- [ ] **Step 3: Produce the table**

Using the fraction printed by Step 1 in place of `<p_large>`:

```bash
.venv/bin/python docs/spikes/corpus_yield.py report <p_large> docs/spikes/pd12m-sample.jsonl \
  | tee /tmp/gate-report.txt
```

Expected: the full table.

- [ ] **Step 4: Keep the sample out of git**

The JSONL is a measurement artifact, not source. Add to `.gitignore`, under the
existing datasets section:

```
docs/spikes/pd12m-sample.jsonl
```

- [ ] **Step 5: Write the result up**

Create `docs/spikes/2026-09-12-corpus-face-yield.md`, following the shape of
`docs/spikes/2026-09-12-convergence-smoke-run.md`. It must contain:

- the question, in one sentence
- the two commands, verbatim, and the seed (0)
- stage 1's fraction and the per-source table from Step 1
- the yield table from Step 3, with the sub-512 rows marked as lower bounds
- the fraction of sampled images with exactly one face
- the download failure rate
- **a conclusion that states the upper-bound caveat in its own words** — the
  detector counts faces in blurred engravings and damaged scans, so this is a
  ceiling on usable faces and not a count of them
- what the number implies for the next decision, stated as a range rather than a
  single figure, given a sample that returns faces in the tens

- [ ] **Step 6: Add a CHANGELOG entry**

`CHANGELOG.md` is organised by what was established. Add the result under
**Measured**, in one or two lines, naming the number and the caveat.

- [ ] **Step 7: Run the suite and the linter**

Run: `.venv/bin/pytest -q && .venv/bin/ruff check .`

Expected: `227 passed`, then `All checks passed!`

- [ ] **Step 8: Commit**

```bash
git add docs/spikes/2026-09-12-corpus-face-yield.md .gitignore CHANGELOG.md
git commit -m "gate: what PD12M actually yields, and why it is a ceiling"
```

---

## What this plan does not do

Named so nobody adds them mid-task:

- No alignment, no cropping, no corpus construction
- No quality, blur, or compression scoring — that is a separate gate
- No download beyond the 5,000-image sample
- Nothing added to `ogan/`
- No target face count is chosen. That decision follows the measurement, in its
  own conversation.
