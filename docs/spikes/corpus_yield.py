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
