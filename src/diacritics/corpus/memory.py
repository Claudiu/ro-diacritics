"""In-memory corpus: the fake used by tests."""

from collections.abc import Iterator, Sequence

from diacritics.corpus.source import Document


class InMemorySource:
    def __init__(self, documents: Sequence[Document]) -> None:
        self._documents = tuple(documents)

    def iter_documents(self) -> Iterator[Document]:
        yield from self._documents
