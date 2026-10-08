"""`diacritics fetch`: corpus → alphabet + train/validation shards."""

import dataclasses
import json
import logging
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from diacritics.common.logging import log
from diacritics.config.settings import Settings
from diacritics.corpus.filter import filtered, is_validation
from diacritics.corpus.source import CorpusSource, Document
from diacritics.dataset.shards import COLUMNS, ShardMeta, split_paths, write_split
from diacritics.domain.alphabet import Alphabet
from diacritics.domain.normalize import strip_diacritics

logger = logging.getLogger(__name__)

ALPHABET_FILE = "alphabet.json"
STAMP_FILE = "fetch.json"
PARTIAL_FILE = "fetch.partial"
"""Present while shards are being rebuilt, so an interrupted fetch is never adopted."""
BASELINE_FILE = "baseline.json"
CORRECTIONS_SPLIT = "corrections"


def stamp(settings: Settings) -> str:
    """Every setting the shards depend on; a match means fetch has nothing to do."""
    m = settings.model
    corpus = dataclasses.asdict(settings.corpus)

    return json.dumps(
        {**corpus, "vocab_size": m.vocab_size, "window": m.window, "overlap": m.overlap}
    )


def usable(source: CorpusSource, settings: Settings) -> Iterator[Document]:
    c = settings.corpus

    return filtered(
        source.iter_documents(), c.min_doc_chars, c.min_diacritic_ratio, c.max_documents
    )


def build_alphabet(source: CorpusSource, settings: Settings) -> Alphabet:
    """Characters of the diacritic-free input, most frequent first."""
    counts: Counter[str] = Counter()
    for document in usable(source, settings):
        counts.update(strip_diacritics(document.text))

    return Alphabet.build(counts, settings.model.vocab_size)


def complete_unstamped(processed: Path, window: int) -> bool:
    """Shards from before fetch wrote stamps: whole (each .bin exactly rows x window) and
    not left behind by an interrupted rebuild."""
    if (processed / STAMP_FILE).exists() or (processed / PARTIAL_FILE).exists():
        return False
    if not (processed / ALPHABET_FILE).exists():
        return False

    for split in ("train", "valid"):
        bin_path, meta_path = split_paths(processed, split)
        if not (bin_path.exists() and meta_path.exists()):
            return False

        meta = ShardMeta.load(meta_path)
        expected = meta.rows * meta.window * COLUMNS
        if meta.window != window or meta.rows == 0 or bin_path.stat().st_size != expected:
            return False

    return True


def build_shards(source: CorpusSource, settings: Settings) -> None:
    processed = settings.paths.processed
    stamp_path = processed / STAMP_FILE
    if stamp_path.exists() and stamp_path.read_text() == stamp(settings):
        log(logger, "shards up to date", dir=str(processed))
        return
    if complete_unstamped(processed, settings.model.window):
        # Built before stamps existed; rebuilding the whole corpus to learn nothing new
        # would take hours. The row counts are logged because a shard cannot tell how many
        # documents it was built from: delete data/processed to force a rebuild.
        stamp_path.write_text(stamp(settings))
        rows = {
            split: ShardMeta.load(split_paths(processed, split)[1]).rows
            for split in ("train", "valid")
        }
        log(logger, "adopted unstamped shards", dir=str(processed), **rows)
        return

    processed.mkdir(parents=True, exist_ok=True)
    partial = processed / PARTIAL_FILE
    partial.touch()
    stamp_path.unlink(missing_ok=True)
    (processed / BASELINE_FILE).unlink(missing_ok=True)

    alphabet = build_alphabet(source, settings)
    (processed / ALPHABET_FILE).write_text(json.dumps(alphabet.to_dict(), ensure_ascii=False))
    log(logger, "alphabet built", size=alphabet.size)

    window = settings.model.window
    stride = window - settings.model.overlap
    valid_every = settings.corpus.valid_every
    documents = list(usable(source, settings)) if settings.corpus.max_documents else None

    for split, selector in (("train", False), ("valid", True)):
        stream = iter(documents) if documents is not None else usable(source, settings)
        chosen = (d for d in stream if is_validation(d, valid_every) == selector)
        meta = write_split(split_paths(processed, split)[0], chosen, alphabet, window, stride)
        log(logger, "split written", split=split, rows=meta.rows, window=meta.window)

    # Written last, so an interrupted fetch rebuilds next time.
    stamp_path.write_text(stamp(settings))
    partial.unlink()


def build_corrections(source: CorpusSource, settings: Settings) -> int:
    """The known-mistakes split, rebuilt on every fetch: it is a few KB, and an edit to the
    corrections directory must never leave a stale shard. Returns the row count."""
    m = settings.model
    path = split_paths(settings.paths.processed, CORRECTIONS_SPLIT)[0]
    documents = source.iter_documents()
    meta = write_split(path, documents, load_alphabet(settings), m.window, m.window - m.overlap)
    log(logger, "split written", split=CORRECTIONS_SPLIT, rows=meta.rows, window=meta.window)

    return meta.rows


def load_alphabet(settings: Settings) -> Alphabet:
    path = settings.paths.processed / ALPHABET_FILE

    return Alphabet.from_dict(json.loads(path.read_text()))
