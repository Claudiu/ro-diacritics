"""One training example: input ids, target labels and the candidate mask, from text."""

from dataclasses import dataclass

from diacritics.domain.alphabet import Alphabet
from diacritics.domain.normalize import fix_cedilla, is_candidate, labels_of, strip_diacritics


@dataclass(frozen=True, slots=True)
class Example:
    input_ids: list[int]
    labels: list[int]
    candidates: list[bool]

    def __len__(self) -> int:
        return len(self.input_ids)


def encode_example(text: str, alphabet: Alphabet) -> Example:
    """Encode correct Romanian text as a (diacritic-free input, labels) pair."""
    text = fix_cedilla(text)
    stripped = strip_diacritics(text)

    return Example(
        input_ids=alphabet.encode(stripped),
        labels=[int(label) for label in labels_of(text)],
        candidates=[is_candidate(ch) for ch in stripped],
    )
