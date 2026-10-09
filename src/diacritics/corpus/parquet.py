"""Hugging Face parquet shards with `id` and `text` columns: Romanian Wikipedia
(`wikimedia/wikipedia`) and web text (`HuggingFaceFW/fineweb-2`, `ron_Latn`).

Already plain text, so no parsing: download once, stream row batches.
"""

import zlib
from collections.abc import Iterator, Sequence
from pathlib import Path

import pyarrow.parquet as pq

from diacritics.common.download import download
from diacritics.corpus.source import CorpusUnavailableError, Document

BATCH_ROWS = 256


class ParquetSource:
    def __init__(self, urls: Sequence[str], cache_dir: Path, every: int = 1) -> None:
        """Keeps one document in `every` (chosen by id, so stable across runs)."""
        self._urls = tuple(urls)
        self._cache_dir = cache_dir
        self._every = every

    def iter_documents(self) -> Iterator[Document]:
        for url in self._urls:
            path = self._cache_dir / url.rsplit("/", 1)[-1]
            try:
                download(url, path)
            except OSError as err:
                raise CorpusUnavailableError(f"download {url}: {err}") from err

            for document in _iter_parquet(path):
                if self._every == 1 or zlib.crc32(document.id.encode()) % self._every == 0:
                    yield document


def _iter_parquet(path: Path) -> Iterator[Document]:
    try:
        file = pq.ParquetFile(path)
    except OSError as err:
        raise CorpusUnavailableError(f"read {path}: {err}") from err

    for batch in file.iter_batches(batch_size=BATCH_ROWS, columns=["id", "text"]):
        ids = batch.column("id").to_pylist()
        texts = batch.column("text").to_pylist()
        for doc_id, text in zip(ids, texts, strict=True):
            yield Document(id=str(doc_id), text=text)
