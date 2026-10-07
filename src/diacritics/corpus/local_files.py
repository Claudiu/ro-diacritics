"""Corpus from a directory of UTF-8 .txt files, one document per file."""

from collections.abc import Iterator
from pathlib import Path

from diacritics.corpus.source import CorpusUnavailableError, Document


class LocalFilesSource:
    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def iter_documents(self) -> Iterator[Document]:
        if not self._directory.is_dir():
            raise CorpusUnavailableError(f"{self._directory} is not a directory")

        for path in sorted(self._directory.glob("*.txt")):
            yield Document(id=path.stem, text=path.read_text(encoding="utf-8"))
