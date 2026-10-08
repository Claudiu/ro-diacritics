"""`diacritics fetch`: corpus → alphabet + train/validation shards."""

import dataclasses
import json
import logging
from collections import Counter
from collections.abc import Iterator

from diacritics.common.logging import log
from diacritics.config.settings import Settings
from diacritics.corpus.filter import filtered, is_validation
from diacritics.corpus.source import CorpusSource, Document
from diacritics.dataset.shards import split_paths, write_split
from diacritics.domain.alphabet import Alphabet
from diacritics.domain.normalize import strip_diacritics

logger = logging.getLogger(__name__)

ALPHABET_FILE = "alphabet.json"
STAMP_FILE = "fetch.json"
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


def build_shards(source: CorpusSource, settings: Settings) -> None:
    processed = settings.paths.processed
    stamp_path = processed / STAMP_FILE
    if stamp_path.exists() and stamp_path.read_text() == stamp(settings):
        log(logger, "shards up to date", dir=str(processed))
        return

    processed.mkdir(parents=True, exist_ok=True)
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
