"""End to end on CPU with an in-memory corpus: fetch → train → export → restore → resume."""

import dataclasses
from pathlib import Path

import pytest

from diacritics.artifacts.memory import InMemoryStore
from diacritics.config.settings import Paths, Settings
from diacritics.corpus.memory import InMemorySource
from diacritics.corpus.source import Document
from diacritics.dataset.build import build_shards, load_alphabet
from diacritics.export.safetensors import load_export
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
    settings = Settings(Paths(tmp_path / "data", tmp_path / "artifacts"), corpus, model, training)
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

    Trainer(settings, alphabet, store).run()
    assert store.read(CHECKPOINT) is not None

    model, loaded_alphabet, overlap, threshold = load_export(store, BEST_EXPORT)
    assert loaded_alphabet == alphabet
    predictor = Predictor(model, loaded_alphabet, overlap, threshold)
    text = "Tara mea e frumoasa si in ea sunt multi oameni buni, asa ca stam acasa."
    assert len(predictor.restore(text)) == len(text)

    resumed = dataclasses.replace(
        settings, training=dataclasses.replace(settings.training, max_steps=5)
    )
    Trainer(resumed, alphabet, store).run()
