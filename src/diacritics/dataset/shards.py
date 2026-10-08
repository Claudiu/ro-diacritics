"""Training windows on disk: fixed-size uint8 rows of (input ids, labels).

Written once by `diacritics fetch` as a flat binary file per split plus meta.json;
read back as a memory-mapped array so training never loads the corpus into RAM.
"""

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import numpy.typing as npt

from diacritics.common.logging import log
from diacritics.corpus.source import Document
from diacritics.domain.alphabet import PAD_ID, Alphabet
from diacritics.domain.example import encode_example
from diacritics.domain.label import Label

logger = logging.getLogger(__name__)

INPUT = 0
LABEL = 1
COLUMNS = 2
LOG_EVERY_DOCS = 10_000


@dataclass(frozen=True, slots=True)
class ShardMeta:
    window: int
    rows: int

    @classmethod
    def load(cls, path: Path) -> "ShardMeta":
        data = json.loads(path.read_text())

        return cls(window=int(data["window"]), rows=int(data["rows"]))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"window": self.window, "rows": self.rows}))


def split_paths(processed: Path, split: str) -> tuple[Path, Path]:
    return processed / f"{split}.bin", processed / f"{split}.json"


class ShardWriter:
    """Appends windows of one document at a time; a document's last partial window is
    padded with PAD_ID / Label.NONE."""

    def __init__(self, path: Path, window: int, stride: int) -> None:
        self._path = path
        self._window = window
        self._stride = stride
        self._rows = 0
        self._file: BinaryIO = path.open("wb")

    def add(self, document: Document, alphabet: Alphabet) -> None:
        example = encode_example(document.text, alphabet)
        ids = np.asarray(example.input_ids, dtype=np.uint8)
        labels = np.asarray(example.labels, dtype=np.uint8)

        for start in range(0, len(ids), self._stride):
            row = np.full((self._window, COLUMNS), PAD_ID, dtype=np.uint8)
            row[:, LABEL] = int(Label.NONE)
            chunk = slice(start, start + self._window)
            n = len(ids[chunk])
            row[:n, INPUT] = ids[chunk]
            row[:n, LABEL] = labels[chunk]
            self._file.write(row.tobytes())
            self._rows += 1
            if n < self._window:
                break

    def close(self) -> ShardMeta:
        self._file.close()
        meta = ShardMeta(window=self._window, rows=self._rows)
        meta.save(self._path.with_suffix(".json"))

        return meta


def write_split(
    path: Path, documents: Iterator[Document], alphabet: Alphabet, window: int, stride: int
) -> ShardMeta:
    writer = ShardWriter(path, window, stride)
    for i, document in enumerate(documents, start=1):
        writer.add(document, alphabet)
        if i % LOG_EVERY_DOCS == 0:
            log(logger, "shard progress", split=path.stem, documents=i, rows=writer._rows)

    return writer.close()


def split_rows(processed: Path, split: str) -> int:
    """Rows in a written split; 0 when it was never written."""
    meta_path = split_paths(processed, split)[1]

    return ShardMeta.load(meta_path).rows if meta_path.exists() else 0


def open_split(processed: Path, split: str) -> npt.NDArray[np.uint8]:
    """Memory-mapped (rows, window, 2) uint8 array."""
    bin_path, meta_path = split_paths(processed, split)
    meta = ShardMeta.load(meta_path)
    array: npt.NDArray[np.uint8] = np.memmap(
        bin_path, dtype=np.uint8, mode="r", shape=(meta.rows, meta.window, COLUMNS)
    )

    return array
