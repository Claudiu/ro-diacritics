"""Character vocabulary: maps characters to model input ids and back.

Serialized into config.json; crates/domain/src/alphabet.rs reads the same layout.
"""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Self

PAD_ID = 0
UNK_ID = 1
RESERVED = 2
"""Ids below RESERVED are special; real characters start at RESERVED."""


@dataclass(frozen=True, slots=True)
class Alphabet:
    chars: tuple[str, ...]
    _index: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(set(self.chars)) != len(self.chars):
            raise ValueError("alphabet has duplicate characters")

        index = {ch: i + RESERVED for i, ch in enumerate(self.chars)}
        object.__setattr__(self, "_index", index)

    @classmethod
    def build(cls, counts: Counter[str], max_size: int) -> Self:
        """Keep the most frequent characters so the vocabulary fits `max_size` ids."""
        if max_size <= RESERVED:
            raise ValueError(f"max_size must be > {RESERVED}")

        kept = [ch for ch, _ in counts.most_common(max_size - RESERVED)]

        return cls(tuple(sorted(kept)))

    @classmethod
    def from_texts(cls, texts: Iterable[str], max_size: int) -> Self:
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(text)

        return cls.build(counts, max_size)

    @property
    def size(self) -> int:
        return len(self.chars) + RESERVED

    def encode(self, text: str) -> list[int]:
        return [self._index.get(ch, UNK_ID) for ch in text]

    def to_dict(self) -> dict[str, object]:
        return {"pad_id": PAD_ID, "unk_id": UNK_ID, "chars": "".join(self.chars)}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Self:
        chars = data["chars"]
        if not isinstance(chars, str):
            raise ValueError("alphabet.chars must be a string")

        return cls(tuple(chars))
