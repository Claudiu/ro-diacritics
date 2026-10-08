"""Typed settings, loaded once from DIACRITICS_* env vars and validated at startup.

Defaults are the real training configuration; .env.example lists every field.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

from diacritics.common import env

WIKIPEDIA_RO_URLS = (
    "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.ro/train-00000-of-00002.parquet",
    "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.ro/train-00001-of-00002.parquet",
)


@dataclass(frozen=True, slots=True)
class Paths:
    data_dir: Path
    artifacts_dir: Path
    corrections_dir: Path
    """Tracked known mistakes, one correct sentence per line (see corpus/corrections.py)."""

    @property
    def raw(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed(self) -> Path:
        return self.data_dir / "processed"


@dataclass(frozen=True, slots=True)
class CorpusSettings:
    urls: tuple[str, ...]
    min_doc_chars: int
    min_diacritic_ratio: float
    max_documents: int
    """0 means all documents; set small for smoke tests."""
    valid_every: int
    """One document in `valid_every` goes to the validation split."""


@dataclass(frozen=True, slots=True)
class ModelSettings:
    vocab_size: int
    window: int
    overlap: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    dropout: float


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    batch_size: int
    lr: float
    warmup_steps: int
    max_steps: int
    eval_every: int
    eval_batches: int
    checkpoint_every: int
    seed: int
    device: str
    num_workers: int
    threshold: float
    corrections_repeat: int
    """How many times the corrections shard appears in each training epoch; 0 disables it."""


@dataclass(frozen=True, slots=True)
class Settings:
    paths: Paths
    corpus: CorpusSettings
    model: ModelSettings
    training: TrainingSettings

    @classmethod
    def from_env(cls) -> Self:
        settings = cls(
            paths=Paths(
                data_dir=Path(env.read_str("DATA_DIR", "data")),
                artifacts_dir=Path(env.read_str("ARTIFACTS_DIR", "artifacts")),
                corrections_dir=Path(env.read_str("CORRECTIONS_DIR", "corrections")),
            ),
            corpus=CorpusSettings(
                urls=WIKIPEDIA_RO_URLS,
                min_doc_chars=env.read_int("MIN_DOC_CHARS", 200),
                min_diacritic_ratio=env.read_float("MIN_DIACRITIC_RATIO", 0.05),
                max_documents=env.read_int("MAX_DOCUMENTS", 0),
                valid_every=env.read_int("VALID_EVERY", 50),
            ),
            model=ModelSettings(
                vocab_size=env.read_int("VOCAB_SIZE", 256),
                window=env.read_int("WINDOW", 256),
                overlap=env.read_int("OVERLAP", 32),
                d_model=env.read_int("D_MODEL", 192),
                n_layers=env.read_int("N_LAYERS", 4),
                n_heads=env.read_int("N_HEADS", 4),
                d_ff=env.read_int("D_FF", 512),
                dropout=env.read_float("DROPOUT", 0.1),
            ),
            training=TrainingSettings(
                batch_size=env.read_int("BATCH_SIZE", 256),
                lr=env.read_float("LR", 3e-4),
                warmup_steps=env.read_int("WARMUP_STEPS", 1000),
                max_steps=env.read_int("MAX_STEPS", 30000),
                eval_every=env.read_int("EVAL_EVERY", 1000),
                eval_batches=env.read_int("EVAL_BATCHES", 50),
                checkpoint_every=env.read_int("CHECKPOINT_EVERY", 1000),
                seed=env.read_int("SEED", 42),
                device=env.read_str("DEVICE", "auto"),
                num_workers=env.read_int("NUM_WORKERS", 2),
                threshold=env.read_float("THRESHOLD", 0.5),
                corrections_repeat=env.read_int("CORRECTIONS_REPEAT", 20),
            ),
        )
        settings.validate()

        return settings

    def validate(self) -> None:
        m, t, c = self.model, self.training, self.corpus
        problems = []
        if m.vocab_size > 256:
            problems.append("VOCAB_SIZE must be <= 256 (ids are stored as uint8)")
        if not 0 <= m.overlap < m.window:
            problems.append("OVERLAP must be in [0, WINDOW)")
        if m.d_model % m.n_heads:
            problems.append("D_MODEL must be divisible by N_HEADS")
        if not 0 <= m.dropout < 1:
            problems.append("DROPOUT must be in [0, 1)")
        if t.batch_size <= 0 or t.max_steps <= 0:
            problems.append("BATCH_SIZE and MAX_STEPS must be positive")
        if t.corrections_repeat < 0:
            problems.append("CORRECTIONS_REPEAT must be >= 0")
        if not 0 <= t.threshold <= 1:
            problems.append("THRESHOLD must be in [0, 1]")
        if t.device not in ("auto", "cpu", "cuda", "mps"):
            problems.append("DEVICE must be auto, cpu, cuda or mps")
        if c.valid_every < 2:
            problems.append("VALID_EVERY must be >= 2")
        if not 0 <= c.min_diacritic_ratio <= 1:
            problems.append("MIN_DIACRITIC_RATIO must be in [0, 1]")

        if problems:
            raise ValueError("invalid settings: " + "; ".join(problems))
