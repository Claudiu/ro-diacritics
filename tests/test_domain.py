"""Domain rules against the fixtures shared with the Rust crate."""

from collections import Counter
from typing import Any

import pytest

from diacritics.domain.alphabet import PAD_ID, RESERVED, UNK_ID, Alphabet
from diacritics.domain.decode import Prediction, restore
from diacritics.domain.example import encode_example
from diacritics.domain.label import Label
from diacritics.domain.normalize import (
    fix_cedilla,
    is_candidate,
    labels_of,
    strip_diacritics,
)
from diacritics.domain.windows import Window, plan_windows
from tests.conftest import load_jsonl


@pytest.mark.parametrize("case", load_jsonl("normalize.jsonl"), ids=lambda c: repr(c["text"]))
def test_normalize_fixture(case: dict[str, Any]) -> None:
    text = str(case["text"])
    stripped = strip_diacritics(fix_cedilla(text))

    assert stripped == case["stripped"]
    assert [int(label) for label in labels_of(text)] == case["labels"]
    assert [is_candidate(ch) for ch in stripped] == case["candidates"]


@pytest.mark.parametrize("case", load_jsonl("decode.jsonl"), ids=lambda c: repr(c["text"]))
def test_decode_fixture(case: dict[str, Any]) -> None:
    predictions = [
        Prediction(Label(int(label)), float(conf)) for label, conf in case["predictions"]
    ]

    assert restore(case["text"], predictions, case["threshold"]) == case["restored"]


def test_decode_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        restore("ab", [Prediction(Label.NONE, 1.0)], 0.5)


@pytest.mark.parametrize(
    "case", load_jsonl("windows.jsonl"), ids=lambda c: f"{c['length']}/{c['window']}/{c['overlap']}"
)
def test_windows_fixture(case: dict[str, Any]) -> None:
    expected = [Window(*w) for w in case["windows"]]

    assert plan_windows(case["length"], case["window"], case["overlap"]) == expected


@pytest.mark.parametrize("length", range(0, 60))
@pytest.mark.parametrize(("window", "overlap"), [(8, 0), (8, 2), (8, 5), (16, 4), (3, 1)])
def test_windows_partition_the_input(length: int, window: int, overlap: int) -> None:
    windows = plan_windows(length, window, overlap)
    owned = [pos for w in windows for pos in range(w.keep_from, w.keep_to)]

    assert owned == list(range(length))
    for w in windows:
        assert 0 <= w.start <= w.keep_from < w.keep_to <= w.end <= length
        assert w.end - w.start <= window


def test_windows_reject_bad_arguments() -> None:
    with pytest.raises(ValueError):
        plan_windows(10, 0, 0)
    with pytest.raises(ValueError):
        plan_windows(10, 4, 4)


def test_alphabet_encodes_and_round_trips() -> None:
    alphabet = Alphabet.build(Counter("aaab  c"), max_size=4)

    assert alphabet.chars == (" ", "a")
    assert alphabet.size == 2 + RESERVED
    assert alphabet.encode("a z") == [RESERVED + 1, RESERVED, UNK_ID]
    assert Alphabet.from_dict(alphabet.to_dict()) == alphabet
    assert PAD_ID not in alphabet.encode("a")


def test_alphabet_rejects_duplicates() -> None:
    with pytest.raises(ValueError):
        Alphabet(("a", "a"))


def test_encode_example_aligns_everything() -> None:
    alphabet = Alphabet(tuple("acsnă "))
    example = encode_example("În casă", alphabet)

    assert len(example) == 7
    assert example.labels == [2, 0, 0, 0, 0, 0, 1]
    assert example.candidates == [True, False, False, False, True, True, True]
    assert example.input_ids[0] == UNK_ID
