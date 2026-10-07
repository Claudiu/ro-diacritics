"""The three per-character classes the model predicts."""

from enum import IntEnum


class Label(IntEnum):
    """What to do with one character of diacritic-free input.

    The base letter is known from the input, so one class covers several letters:
    BREVE_COMMA turns a→ă, s→ș, t→ț; CIRCUMFLEX turns a→â, i→î.
    """

    NONE = 0
    BREVE_COMMA = 1
    CIRCUMFLEX = 2


LABEL_NAMES = tuple(label.name.lower() for label in Label)
"""Names written into config.json so the Rust side agrees on class order."""
