"""Keep only documents that are usable training data.

Much Romanian web text omits diacritics entirely; training on it teaches the model to
omit them too. A document is kept when enough of its candidate letters carry one, and
when ă, ș and ț each show up: old encodings (common in subtitles) lost those three but
kept â and î, which passes a plain ratio and would teach "zapada" for "zăpadă".
"""

import re
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


MIN_MARKED = {"a": ("ă", 0.03), "s": ("ș", 0.03), "t": ("ț", 0.02)}
"""Plain letter → (its diacritic form, least share of that letter carrying it). Wikipedia
and well-encoded subtitles sit near 15-20 %; text that lost the mark sits at 0."""


def marks_every_letter(text: str) -> bool:
    lower = text.lower()
    for plain, (marked, least) in MIN_MARKED.items():
        n_marked = lower.count(marked)
        if n_marked < least * (n_marked + lower.count(plain)):
            return False

    return True


ALWAYS_MARKED = frozenset(
    ("și", "dacă", "fără", "după", "așa", "când", "într", "dintr", "niște", "către")
    + ("decât", "același", "aceeași", "încât", "fiindcă")
)
"""Words with no diacritic-free spelling: seeing them bare means the text drops marks."""
BARE = {strip_diacritics(w): w for w in ALWAYS_MARKED}
MAX_BARE_SHARE = 0.05
WORD = re.compile(r"\w+")


def drops_marks(text: str) -> bool:
    """Too many ALWAYS_MARKED words written bare: marks were typed only some of the time."""
    bare = marked = 0
    for word in WORD.findall(text.lower()):
        bare += word in BARE
        marked += word in ALWAYS_MARKED

    return bare > MAX_BARE_SHARE * (bare + marked)


def clean(text: str) -> str:
    """Precomposed characters, comma-below diacritics."""
    return fix_cedilla(unicodedata.normalize("NFC", text))


def keep(document: Document, min_chars: int, min_ratio: float) -> bool:
    text = document.text
    if len(text) < min_chars:
        return False

    return diacritic_ratio(text) >= min_ratio and marks_every_letter(text) and not drops_marks(text)


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
