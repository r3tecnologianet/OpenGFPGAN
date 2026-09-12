"""The spike's arithmetic, which is the part that can be wrong quietly.

The spike itself is a measurement and is not otherwise tested. These three
functions are, because a face measured on the wrong dimension or a composition
applied outside its valid range produces a plausible-looking wrong answer.
"""

import importlib.util
import pathlib

import pytest

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


def test_an_empty_sample_raises_rather_than_answering_zero():
    """Zero faces and zero records are different findings. Only one is a measurement."""
    with pytest.raises(ValueError, match="empty sample"):
        corpus_yield.expected_total(0.4, [], 512, 1_000_000)


def test_the_sample_is_reproducible_from_its_seed():
    ids = [f"id{n}" for n in range(1000)]
    assert corpus_yield.sample_ids(ids, 20, seed=0) == corpus_yield.sample_ids(ids, 20, seed=0)
    assert corpus_yield.sample_ids(ids, 20, seed=0) != corpus_yield.sample_ids(ids, 20, seed=1)
    assert len(corpus_yield.sample_ids(ids, 20, seed=0)) == 20


def test_the_sample_does_not_exceed_the_population():
    ids = ["a", "b", "c"]
    assert sorted(corpus_yield.sample_ids(ids, 10, seed=0)) == ["a", "b", "c"]
