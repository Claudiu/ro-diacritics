from pathlib import Path

import pytest

from diacritics.corpus.filter import diacritic_ratio, filtered, is_validation, keep
from diacritics.corpus.local_files import LocalFilesSource
from diacritics.corpus.memory import InMemorySource
from diacritics.corpus.source import CorpusUnavailableError, Document

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
