"""Character-level text rules shared with crates/domain/src/normalize.rs.

Every function here maps one character to one character, so input, labels and
output always have the same length. Change this file and the Rust mirror together;
test/fixtures/normalize.jsonl keeps them honest.
"""

from diacritics.domain.label import Label

CEDILLA_TO_COMMA = {
    "ş": "ș",  # ş → ș
    "ţ": "ț",  # ţ → ț
    "Ş": "Ș",  # Ş → Ș
    "Ţ": "Ț",  # Ţ → Ț
}
"""Legacy cedilla forms, still common in older texts, mapped to the correct comma-below ones."""

DIACRITIC_TO_BASE = {
    "ă": "a",
    "â": "a",
    "î": "i",
    "ș": "s",
    "ț": "t",
    "Ă": "A",
    "Â": "A",
    "Î": "I",
    "Ș": "S",
    "Ț": "T",
}

CANDIDATES = frozenset("aistAIST")
"""Letters that can carry a diacritic; the model is trained and scored only on these."""

_BREVE_COMMA = frozenset("ășțĂȘȚ")
_CIRCUMFLEX = frozenset("âîÂÎ")


def fix_cedilla(text: str) -> str:
    """Replace cedilla s/t with comma-below s/t."""
    return "".join(CEDILLA_TO_COMMA.get(ch, ch) for ch in text)


def strip_diacritics(text: str) -> str:
    """Remove Romanian diacritics, keeping length and case."""
    return "".join(DIACRITIC_TO_BASE.get(ch, ch) for ch in text)


def is_candidate(ch: str) -> bool:
    """Whether a diacritic-free character could take a diacritic."""
    return ch in CANDIDATES


def label_of(ch: str) -> Label:
    """The label a character of correct Romanian text carries."""
    if ch in _BREVE_COMMA:
        return Label.BREVE_COMMA
    if ch in _CIRCUMFLEX:
        return Label.CIRCUMFLEX

    return Label.NONE


def labels_of(text: str) -> list[Label]:
    """Per-character labels of correct Romanian text (cedilla forms accepted)."""
    return [label_of(ch) for ch in fix_cedilla(text)]
