"""End to end on CPU with an in-memory corpus: fetch → train → export → restore → resume."""

import dataclasses
from pathlib import Path

import pytest

from diacritics.artifacts.memory import InMemoryStore
from diacritics.common.progress import LogProgress
from diacritics.config.settings import Paths, Settings
from diacritics.corpus.corrections import CorrectionsSource
from diacritics.corpus.memory import InMemorySource
from diacritics.corpus.source import Document
from diacritics.dataset.build import (
    PARTIAL_FILE,
    STAMP_FILE,
    build_corrections,
    build_shards,
    load_alphabet,
)
from diacritics.dataset.shards import split_rows
from diacritics.export.safetensors import load_checkpoint, load_export
from diacritics.model.infer import Predictor
from diacritics.train.loop import BEST_EXPORT, CHECKPOINT, Trainer

SENTENCES = [
    "Țara mea e frumoasă și în ea sunt mulți oameni buni, așa că stăm acasă.",
    "Lângă casa mea e casa ta și e o casă foarte frumoasă.",
    "În casă era o masă și pe masă erau mere și pere pentru copiii noștri.",
    "Când vine iarna, munții sunt albi și râurile îngheață pe sub poduri.",
]


def tiny_settings(tmp_path: Path) -> Settings:
    base = Settings.from_env()
    model = dataclasses.replace(
        base.model, window=16, overlap=4, d_model=16, n_layers=1, n_heads=2, d_ff=32
    )
    corpus = dataclasses.replace(base.corpus, min_doc_chars=20, valid_every=2, max_documents=0)
    training = dataclasses.replace(
        base.training,
        batch_size=4,
        max_steps=3,
        warmup_steps=1,
        eval_every=2,
        eval_batches=2,
        checkpoint_every=2,
        device="cpu",
        num_workers=0,
    )
    paths = Paths(tmp_path / "data", tmp_path / "artifacts", tmp_path / "corrections")
    settings = Settings(paths, corpus, model, training)
    settings.validate()

    return settings


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_fetch_train_export_restore_resume(tmp_path: Path) -> None:
    settings = tiny_settings(tmp_path)
    docs = [Document(str(i), " ".join(SENTENCES) * 3) for i in range(8)]
    source = InMemorySource(docs)
    store = InMemoryStore()

    build_shards(source, settings)
    alphabet = load_alphabet(settings)
    assert alphabet.size > 20
    assert (settings.paths.processed / "train.bin").stat().st_size > 0
    assert (settings.paths.processed / "valid.bin").stat().st_size > 0

    Trainer(settings, alphabet, store, LogProgress()).run()
    assert store.read(CHECKPOINT) is not None

    model, loaded_alphabet, overlap, threshold = load_export(store, BEST_EXPORT)
    assert loaded_alphabet == alphabet
    predictor = Predictor(model, loaded_alphabet, overlap, threshold)
    text = "Tara mea e frumoasa si in ea sunt multi oameni buni, asa ca stam acasa."
    assert len(predictor.restore(text)) == len(text)

    resumed = dataclasses.replace(
        settings, training=dataclasses.replace(settings.training, max_steps=5)
    )
    Trainer(resumed, alphabet, store, LogProgress()).run()


def test_fetch_skips_when_settings_unchanged(tmp_path: Path) -> None:
    settings = tiny_settings(tmp_path)
    docs = [Document(str(i), " ".join(SENTENCES) * 3) for i in range(8)]
    build_shards(InMemorySource(docs), settings)
    train_bin = settings.paths.processed / "train.bin"
    before = train_bin.stat().st_mtime_ns

    build_shards(InMemorySource([]), settings)
    assert train_bin.stat().st_mtime_ns == before

    changed = dataclasses.replace(settings.corpus, valid_every=3)
    build_shards(InMemorySource(docs), dataclasses.replace(settings, corpus=changed))
    assert train_bin.stat().st_mtime_ns != before


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_corrections_are_mixed_into_training(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = tiny_settings(tmp_path)
    build_shards(
        InMemorySource([Document(str(i), " ".join(SENTENCES) * 3) for i in range(8)]), settings
    )
    source = CorrectionsSource(settings.paths.corrections_dir)

    assert build_corrections(source, settings) == 0  # no directory yet: nothing, no error

    settings.paths.corrections_dir.mkdir()
    (settings.paths.corrections_dir / "known.txt").write_text(
        "# comment\n\nPe masă erau mere.\nFata mea a mâncat o pară.\n", encoding="utf-8"
    )
    rows = build_corrections(source, settings)  # windows of 16: each sentence spans two rows
    assert rows == split_rows(settings.paths.processed, "corrections") == 4

    caplog.set_level("INFO")
    Trainer(settings, load_alphabet(settings), InMemoryStore(), LogProgress()).run()
    fields = [r.__dict__.get("fields", {}) for r in caplog.records]
    mixed = next(f for f in fields if "repeat" in f)
    assert mixed == {"rows": rows, "repeat": settings.training.corrections_repeat}
    assert any("corrections_accuracy" in f for f in fields)


def checkpoint_step(store: InMemoryStore) -> int:
    raw = store.read(CHECKPOINT)
    assert raw is not None

    return int(load_checkpoint(raw)["step"])


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_rerunning_a_finished_run_continues(tmp_path: Path) -> None:
    settings = tiny_settings(tmp_path)
    build_shards(
        InMemorySource([Document(str(i), " ".join(SENTENCES) * 3) for i in range(8)]), settings
    )
    alphabet = load_alphabet(settings)
    store = InMemoryStore()

    def run(finetune_steps: int) -> int:
        training = dataclasses.replace(settings.training, finetune_steps=finetune_steps)
        Trainer(
            dataclasses.replace(settings, training=training), alphabet, store, LogProgress()
        ).run()

        return checkpoint_step(store)

    assert run(finetune_steps=2) == 3  # fresh: stops at max_steps
    assert run(finetune_steps=2) == 5  # finished: two more
    assert run(finetune_steps=0) == 5  # continuing disabled: nothing to do


def test_fetch_adopts_complete_unstamped_shards(tmp_path: Path) -> None:
    settings = tiny_settings(tmp_path)
    docs = [Document(str(i), " ".join(SENTENCES) * 3) for i in range(8)]
    build_shards(InMemorySource(docs), settings)
    processed = settings.paths.processed
    train_bin = processed / "train.bin"
    before = train_bin.stat().st_mtime_ns
    assert not (processed / PARTIAL_FILE).exists()

    (processed / STAMP_FILE).unlink()  # as left by fetch before stamps existed
    build_shards(InMemorySource([]), settings)
    assert train_bin.stat().st_mtime_ns == before
    assert (processed / STAMP_FILE).exists()

    (processed / STAMP_FILE).unlink()
    (processed / PARTIAL_FILE).touch()  # an interrupted rebuild is never adopted
    build_shards(InMemorySource(docs), settings)
    assert train_bin.stat().st_mtime_ns != before
    assert not (processed / PARTIAL_FILE).exists()

    (processed / STAMP_FILE).unlink()
    with train_bin.open("r+b") as f:  # truncated shard: rebuilt too
        f.truncate(train_bin.stat().st_size - 1)
    build_shards(InMemorySource(docs), settings)
    assert (train_bin.stat().st_size % (settings.model.window * 2)) == 0
