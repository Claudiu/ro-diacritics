"""`diacritics fetch`: corpus → alphabet + train/validation shards."""

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
    processed.mkdir(parents=True, exist_ok=True)

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


def load_alphabet(settings: Settings) -> Alphabet:
    path = settings.paths.processed / ALPHABET_FILE

    return Alphabet.from_dict(json.loads(path.read_text()))
