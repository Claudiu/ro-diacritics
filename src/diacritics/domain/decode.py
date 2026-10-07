"""Turn per-character predictions back into Romanian text.

Mirrored in crates/domain/src/decode.rs; test/fixtures/decode.jsonl covers both.
"""

from dataclasses import dataclass

from diacritics.domain.label import Label

_APPLY: dict[tuple[str, Label], str] = {
    ("a", Label.BREVE_COMMA): "ă",
    ("s", Label.BREVE_COMMA): "ș",
    ("t", Label.BREVE_COMMA): "ț",
    ("a", Label.CIRCUMFLEX): "â",
    ("i", Label.CIRCUMFLEX): "î",
    ("A", Label.BREVE_COMMA): "Ă",
    ("S", Label.BREVE_COMMA): "Ș",
    ("T", Label.BREVE_COMMA): "Ț",
    ("A", Label.CIRCUMFLEX): "Â",
    ("I", Label.CIRCUMFLEX): "Î",
}
"""Valid (base letter, label) pairs. Anything else leaves the letter unchanged."""


@dataclass(frozen=True, slots=True)
class Prediction:
    label: Label
    confidence: float


def restore(text: str, predictions: list[Prediction], threshold: float) -> str:
    """Apply predictions to diacritic-free `text`.

    A diacritic is added only when the prediction is valid for that letter and at
    least `threshold` confident; a missed diacritic reads better than a wrong one.
    """
    if len(predictions) != len(text):
        raise ValueError(f"{len(predictions)} predictions for {len(text)} characters")

    out: list[str] = []
    for ch, prediction in zip(text, predictions, strict=True):
        if prediction.confidence < threshold:
            out.append(ch)
            continue

        out.append(_APPLY.get((ch, prediction.label), ch))

    return "".join(out)
