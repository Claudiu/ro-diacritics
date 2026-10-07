"""Romanian Wikipedia from the Hugging Face `wikimedia/wikipedia` parquet shards.

Already plain text (no wikitext), so no parsing: download once, stream row batches.
"""

from collections.abc import Iterator, Sequence
from pathlib import Path

import pyarrow.parquet as pq

from diacritics.common.download import download
from diacritics.corpus.source import CorpusUnavailableError, Document

BATCH_ROWS = 256


class WikipediaSource:
    def __init__(self, urls: Sequence[str], cache_dir: Path) -> None:
        self._urls = tuple(urls)
        self._cache_dir = cache_dir

    def iter_documents(self) -> Iterator[Document]:
        for url in self._urls:
            path = self._cache_dir / url.rsplit("/", 1)[-1]
            try:
                download(url, path)
            except OSError as err:
                raise CorpusUnavailableError(f"download {url}: {err}") from err

            yield from _iter_parquet(path)


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
