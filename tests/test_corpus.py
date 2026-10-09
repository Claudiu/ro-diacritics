import gzip
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from diacritics.corpus.filter import (
    diacritic_ratio,
    filtered,
    is_validation,
    keep,
    marks_every_letter,
)
from diacritics.corpus.local_files import LocalFilesSource
from diacritics.corpus.memory import InMemorySource
from diacritics.corpus.parquet import ParquetSource
from diacritics.corpus.source import ChainedSource, CorpusUnavailableError, Document
from diacritics.corpus.subtitles import LINES_PER_DOCUMENT, SubtitlesSource

GOOD = "Țara mea e frumoasă și în ea sunt mulți oameni buni, așa că stăm acasă."
BARE = "Tara mea e frumoasa si in ea sunt multi oameni buni, asa ca stam acasa."


def test_diacritic_ratio() -> None:
    assert diacritic_ratio(BARE) == 0.0
    assert diacritic_ratio("ă") == 1.0
    assert 0.2 < diacritic_ratio(GOOD) < 0.6
    assert diacritic_ratio("xyz") == 0.0


def test_keep_rejects_short_and_bare_documents() -> None:
    assert keep(Document("1", GOOD), min_chars=10, min_ratio=0.05)
    assert not keep(Document("1", GOOD), min_chars=1000, min_ratio=0.05)
    assert not keep(Document("1", BARE), min_chars=10, min_ratio=0.05)


def test_keep_rejects_text_that_lost_breve_and_comma() -> None:
    # Old subtitle encodings kept â and î but dropped ă, ș and ț.
    lost = "Când e ziua platii? Mâine, daca trec banii. Îti fac probleme? Unde e sotia?"

    assert diacritic_ratio(lost) > 0.05
    assert not marks_every_letter(lost)
    assert not keep(Document("1", lost), min_chars=10, min_ratio=0.05)
    assert marks_every_letter(GOOD)


def test_keep_rejects_text_that_drops_marks_now_and_then() -> None:
    sloppy = (
        "Și eu vin, dacă pot, țin minte. Si tu? Daca vrei, fara grabă. "
        "Așa că după masă venim și noi."
    )

    assert marks_every_letter(sloppy)
    assert not keep(Document("1", sloppy), min_chars=10, min_ratio=0.05)


def test_filtered_cleans_and_limits() -> None:
    docs = [Document("1", "aşa " + GOOD), Document("2", BARE), Document("3", GOOD)]
    out = list(filtered(iter(docs), min_chars=10, min_ratio=0.05, limit=1))

    assert [d.id for d in out] == ["1"]
    assert out[0].text.startswith("așa ")

    assert [d.id for d in filtered(iter(docs), 10, 0.05, limit=0)] == ["1", "3"]


def test_validation_split_is_deterministic_and_sparse() -> None:
    docs = [Document(str(i), "") for i in range(1000)]
    chosen = [d.id for d in docs if is_validation(d, 50)]

    assert chosen == [d.id for d in docs if is_validation(d, 50)]
    assert 5 <= len(chosen) <= 60


def test_in_memory_source() -> None:
    docs = [Document("a", "x"), Document("b", "y")]

    assert list(InMemorySource(docs).iter_documents()) == docs


def test_local_files_source(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("bb", encoding="utf-8")
    (tmp_path / "a.txt").write_text("aa", encoding="utf-8")

    assert list(LocalFilesSource(tmp_path).iter_documents()) == [
        Document("a", "aa"),
        Document("b", "bb"),
    ]
    with pytest.raises(CorpusUnavailableError):
        list(LocalFilesSource(tmp_path / "missing").iter_documents())


def test_corrections_source_reads_lines(tmp_path: Path) -> None:
    from diacritics.corpus.corrections import CorrectionsSource

    (tmp_path / "a.txt").write_text("# skip me\n\n  Aşa e.  \nȚara mea.\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("ignored", encoding="utf-8")
    docs = list(CorrectionsSource(tmp_path).iter_documents())

    assert [d.id for d in docs] == ["a:3", "a:4"]
    assert docs[0].text == "Așa e."  # trimmed, cedilla converted to comma below
    assert list(CorrectionsSource(tmp_path / "missing").iter_documents()) == []


def test_subtitles_source_groups_lines_and_samples(tmp_path: Path) -> None:
    url = "https://example.invalid/ro.txt.gz"
    lines = [f"Replica {i}." for i in range(LINES_PER_DOCUMENT * 3 + 7)]
    with gzip.open(tmp_path / "opensubtitles-ro.txt.gz", "wt", encoding="utf-8") as out:
        out.write("\n".join(lines) + "\n")

    every = list(SubtitlesSource(url, tmp_path, every=1).iter_documents())
    assert [d.id for d in every] == ["subtitles:0", "subtitles:1", "subtitles:2"]
    assert every[1].text.splitlines() == lines[LINES_PER_DOCUMENT : 2 * LINES_PER_DOCUMENT]

    sampled = list(SubtitlesSource(url, tmp_path, every=1000).iter_documents())
    assert len(sampled) < len(every)


def test_parquet_source_reads_and_samples(tmp_path: Path) -> None:
    ids = [f"<urn:uuid:{i}>" for i in range(200)]
    table = pa.table({"id": ids, "text": [GOOD] * 200, "url": ["x"] * 200})
    pq.write_table(table, tmp_path / "shard.parquet")
    url = "https://example.invalid/shard.parquet"  # already cached, never downloaded

    every = list(ParquetSource((url,), tmp_path).iter_documents())
    assert [d.id for d in every] == ids
    assert every[0].text == GOOD

    sampled = [d.id for d in ParquetSource((url,), tmp_path, every=4).iter_documents()]
    assert 0 < len(sampled) < len(ids)
    assert sampled == [d.id for d in ParquetSource((url,), tmp_path, every=4).iter_documents()]


def test_chained_source() -> None:
    a = InMemorySource([Document("1", GOOD)])
    b = InMemorySource([Document("2", BARE)])

    assert [d.id for d in ChainedSource(a, b).iter_documents()] == ["1", "2"]
