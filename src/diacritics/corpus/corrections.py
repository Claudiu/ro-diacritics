"""Known mistakes: hand-written sentences with correct diacritics, one per line.

Every `*.txt` file in the directory counts; blank lines and lines starting with `#` are
skipped. They bypass the corpus filters (a sentence is shorter than any min_doc_chars) and
are oversampled in training, so the model sees the words it got wrong far more often.
"""

from collections.abc import Iterator
from pathlib import Path

from diacritics.corpus.filter import clean
from diacritics.corpus.source import Document

COMMENT = "#"


class CorrectionsSource:
    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def iter_documents(self) -> Iterator[Document]:
        """Nothing when the directory is missing: corrections are optional."""
        if not self._directory.is_dir():
            return

        for path in sorted(self._directory.glob("*.txt")):
            lines = path.read_text(encoding="utf-8").splitlines()
            for number, line in enumerate(lines, start=1):
                text = line.strip()
                if not text or text.startswith(COMMENT):
                    continue

                yield Document(id=f"{path.stem}:{number}", text=clean(text))
