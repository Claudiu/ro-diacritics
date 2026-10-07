"""Keep only documents that are usable training data.

Much Romanian web text omits diacritics entirely; training on it teaches the model to
omit them too. A document is kept when enough of its candidate letters carry one.
"""

import unicodedata
import zlib
from collections.abc import Iterator

from diacritics.corpus.source import Document
from diacritics.domain.label import Label
from diacritics.domain.normalize import fix_cedilla, is_candidate, label_of, strip_diacritics


def diacritic_ratio(text: str) -> float:
    """Share of candidate letters (a, i, s, t) that carry a diacritic."""
    candidates = 0
    marked = 0
    for ch in fix_cedilla(text):
        if is_candidate(strip_diacritics(ch)):
            candidates += 1
            if label_of(ch) is not Label.NONE:
                marked += 1

    return marked / candidates if candidates else 0.0


def clean(text: str) -> str:
    """Precomposed characters, comma-below diacritics."""
    return fix_cedilla(unicodedata.normalize("NFC", text))


def keep(document: Document, min_chars: int, min_ratio: float) -> bool:
    text = document.text
    if len(text) < min_chars:
        return False

    return diacritic_ratio(text) >= min_ratio


def is_validation(document: Document, valid_every: int) -> bool:
    """Deterministic split by document id, so train/validation never mix across runs."""
    return zlib.crc32(document.id.encode()) % valid_every == 0


def filtered(
    documents: Iterator[Document], min_chars: int, min_ratio: float, limit: int
) -> Iterator[Document]:
    """Cleaned documents that pass `keep`; at most `limit` of them when `limit` > 0."""
    kept = 0
    for document in documents:
        cleaned = Document(id=document.id, text=clean(document.text))
        if not keep(cleaned, min_chars, min_ratio):
            continue

        yield cleaned
        kept += 1
        if limit and kept >= limit:
            return
