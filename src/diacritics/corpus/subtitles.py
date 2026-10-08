"""Romanian film and TV subtitles from the OPUS OpenSubtitles monolingual dump.

One subtitle line per line, consecutive lines mostly from the same film. Lines are grouped
into documents of `LINES_PER_DOCUMENT`, so the corpus filter judges a passage, not a single
"Da." Everyday dialogue (questions, commands, forms of address) that Wikipedia lacks.
"""

import gzip
import zlib
from collections.abc import Iterator
from pathlib import Path

from diacritics.common.download import download
from diacritics.corpus.source import CorpusUnavailableError, Document

LINES_PER_DOCUMENT = 50


class SubtitlesSource:
    def __init__(self, url: str, cache_dir: Path, every: int) -> None:
        """Keeps one document in `every` (chosen by id, so stable across runs)."""
        self._url = url
        self._cache_dir = cache_dir
        self._every = every

    def iter_documents(self) -> Iterator[Document]:
        path = self._cache_dir / ("opensubtitles-" + self._url.rsplit("/", 1)[-1])
        try:
            download(self._url, path)
        except OSError as err:
            raise CorpusUnavailableError(f"download {self._url}: {err}") from err

        try:
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as lines:
                yield from self._documents(lines)
        except (OSError, EOFError) as err:
            raise CorpusUnavailableError(f"read {path}: {err}") from err

    def _documents(self, lines: Iterator[str]) -> Iterator[Document]:
        chunk: list[str] = []
        number = 0
        for line in lines:
            chunk.append(line.strip())
            if len(chunk) < LINES_PER_DOCUMENT:
                continue

            doc_id = f"subtitles:{number}"
            number += 1
            if zlib.crc32(doc_id.encode()) % self._every == 0:
                yield Document(id=doc_id, text="\n".join(chunk))
            chunk = []
