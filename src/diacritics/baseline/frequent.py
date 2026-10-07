"""Baseline: each word gets its most frequent diacritic form in the corpus.

The number the model has to beat. Anything it cannot beat is not worth shipping.
"""

import re
from collections import Counter, defaultdict
from collections.abc import Iterator

from diacritics.corpus.source import Document
from diacritics.domain.normalize import fix_cedilla, strip_diacritics

WORD = re.compile(r"[^\W\d_]+")


class FrequentFormBaseline:
    def __init__(self) -> None:
        self._forms: dict[str, Counter[str]] = defaultdict(Counter)
        self._best: dict[str, str] = {}

    def fit(self, documents: Iterator[Document]) -> None:
        for document in documents:
            for match in WORD.finditer(fix_cedilla(document.text)):
                word = match.group().lower()
                self._forms[strip_diacritics(word)][word] += 1

        self._best = {key: counter.most_common(1)[0][0] for key, counter in self._forms.items()}
        self._forms.clear()

    @property
    def size(self) -> int:
        return len(self._best)

    def restore(self, text: str) -> str:
        stripped = strip_diacritics(fix_cedilla(text))

        def replace(match: re.Match[str]) -> str:
            word = match.group()
            form = self._best.get(word.lower())
            # str.lower() can change length ("İ" → "i̇"); such words keep their stripped form.
            if form is None or len(form) != len(word):
                return word

            return "".join(
                f.upper() if original.isupper() else f
                for original, f in zip(word, form, strict=True)
            )

        return WORD.sub(replace, stripped)
