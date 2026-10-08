"""Port: where raw Romanian text comes from."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    """Stable across runs; used for the train/validation split."""
    text: str


class CorpusSource(Protocol):
    """Yields documents one at a time; implementations must not load the corpus in memory."""

    def iter_documents(self) -> Iterator[Document]: ...


class CorpusUnavailableError(Exception):
    """The source could not be read (network, missing file, bad format)."""


class ChainedSource:
    """Several sources, one after another."""

    def __init__(self, *sources: CorpusSource) -> None:
        self._sources = sources

    def iter_documents(self) -> Iterator[Document]:
        for source in self._sources:
            yield from source.iter_documents()
