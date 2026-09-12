"""Gate zero: how many usable faces does PD12M hold?

    python docs/spikes/corpus_yield.py metadata           # stage 1, no downloads
    python docs/spikes/corpus_yield.py report P_LARGE FILE # stage 3, the table

Stage 2 has no subcommand here: it needs the eligible-id list that stage 1
produces, so it is driven from a separate script that calls `sample_ids` and
`sample_and_detect` directly rather than from this CLI.

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

    `records` must be the successfully downloaded images only, drawn from the
    `min(width, height) >= MIN_IMAGE_DIM` subset that `p_large` measures. Both
    preconditions are load-bearing: include failed downloads and the rate is
    diluted, sample from the whole dataset instead of the subset and `p_large`
    double-counts. Neither mistake produces an obviously wrong number.
    """
    if not records:
        raise ValueError(
            "no records: an empty sample cannot distinguish 'PD12M holds no faces' "
            "from 'stage 2 wrote nothing'. Check the JSONL path and the download count."
        )
    per_image = faces_at_least(records, threshold) / len(records)
    return p_large * per_image * dataset_size


def sample_ids(ids, count, seed):
    """A reproducible sample. The seed goes in the write-up with the result."""
    rng = random.Random(seed)
    return rng.sample(ids, min(count, len(ids)))


HF_REPO = "Spawning/PD12M"
HF_TREE = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main"
HF_FILE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
METADATA_DIR = "metadata"  # confirmed by the repo listing: 125 files, pd12m.NNN.parquet


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

    `pyarrow.parquet.read_table` has no HTTP backend of its own, so each shard is
    fetched into memory with `urllib` first and handed to pyarrow as a `BytesIO`;
    `urlopen` already follows the redirect the Hub issues to its CDN.

    Returns (total_rows, large_rows, {source: [total, large]}).
    """
    import io
    import urllib.request

    import pyarrow.parquet as pq

    total = 0
    large = 0
    by_source = {}

    for path in paths:
        with urllib.request.urlopen(f"{HF_FILE}/{path}", timeout=60) as response:
            data = response.read()
        table = pq.read_table(io.BytesIO(data), columns=["width", "height", "source"])
        widths = table.column("width").to_pylist()
        heights = table.column("height").to_pylist()
        sources = table.column("source").to_pylist()

        for width, height, source in zip(widths, heights, sources, strict=True):
            counts = by_source.setdefault(source, [0, 0])
            total += 1
            counts[0] += 1
            if width is not None and height is not None and min(width, height) >= min_dim:
                large += 1
                counts[1] += 1

        print(f"  {path}: {total} rows, {large} at least {min_dim}px", flush=True)

    return total, large, by_source


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
