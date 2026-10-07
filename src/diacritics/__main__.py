"""Entry point: config, wiring, subcommands. The only module naming implementations."""

import argparse
import dataclasses
import json
import logging
import sys
from collections.abc import Sequence

from diacritics.artifacts.local_dir import LocalDirStore
from diacritics.baseline.frequent import FrequentFormBaseline
from diacritics.common.logging import console, log, setup
from diacritics.common.progress import LogProgress, Progress, RichProgress
from diacritics.config.settings import Settings
from diacritics.corpus.filter import is_validation
from diacritics.corpus.wikipedia import WikipediaSource
from diacritics.dataset.build import BASELINE_FILE, build_shards, load_alphabet, usable
from diacritics.domain.alphabet import Alphabet
from diacritics.export.safetensors import export, load_export
from diacritics.metrics.candidates import CandidateMetrics
from diacritics.model.encoder import CharEncoder, ModelConfig
from diacritics.model.infer import Predictor
from diacritics.train.loop import BEST_EXPORT, Trainer

logger = logging.getLogger("diacritics")

EVAL_DOCUMENTS = 500


def parse(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="diacritics")
    parser.add_argument("--max-documents", type=int, help="override DIACRITICS_MAX_DOCUMENTS")
    parser.add_argument("--max-steps", type=int, help="override DIACRITICS_MAX_STEPS")
    parser.add_argument("--device", help="override DIACRITICS_DEVICE")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="download the corpus and build alphabet + shards")
    sub.add_parser("baseline", help="most-frequent-form baseline, evaluated on validation")
    sub.add_parser("train", help="train (resumes from the latest checkpoint)")
    sub.add_parser("eval", help="evaluate the exported model on validation documents")
    sub.add_parser("restore", help="restore diacritics in stdin using the exported model")
    fixture = sub.add_parser("fixture", help="write a tiny random-weight export for parity tests")
    fixture.add_argument("directory")

    return parser.parse_args(argv)


def apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    corpus = settings.corpus
    training = settings.training
    if args.max_documents is not None:
        corpus = dataclasses.replace(corpus, max_documents=args.max_documents)
    if args.max_steps is not None:
        training = dataclasses.replace(training, max_steps=args.max_steps)
    if args.device is not None:
        training = dataclasses.replace(training, device=args.device)

    settings = dataclasses.replace(settings, corpus=corpus, training=training)
    settings.validate()

    return settings


def main(argv: Sequence[str] | None = None) -> int:
    log_format = setup()
    args = parse(sys.argv[1:] if argv is None else argv)
    settings = apply_overrides(Settings.from_env(), args)

    source = WikipediaSource(settings.corpus.urls, settings.paths.raw)
    store = LocalDirStore(settings.paths.artifacts_dir)

    match args.command:
        case "fetch":
            build_shards(source, settings)
        case "baseline":
            run_baseline(settings, source)
        case "train":
            progress: Progress = (
                RichProgress(console()) if log_format == "pretty" else LogProgress()
            )
            Trainer(settings, load_alphabet(settings), store, progress).run()
        case "eval":
            run_eval(settings, source, store)
        case "restore":
            model, alphabet, overlap, threshold = load_export(store, BEST_EXPORT)
            predictor = Predictor(model, alphabet, overlap, threshold)
            for line in sys.stdin:
                print(predictor.restore(line.rstrip("\n")))
        case "fixture":
            write_fixture(args.directory)

    return 0


def run_baseline(settings: Settings, source: WikipediaSource) -> None:
    # Cached next to the shards; fetch deletes it whenever it rebuilds them.
    cached = settings.paths.processed / BASELINE_FILE
    if cached.exists():
        log(logger, "baseline", cached=True, **json.loads(cached.read_text()))
        return

    valid_every = settings.corpus.valid_every
    baseline = FrequentFormBaseline()
    baseline.fit(d for d in usable(source, settings) if not is_validation(d, valid_every))
    log(logger, "baseline fitted", words=baseline.size)

    metrics = CandidateMetrics()
    for i, document in enumerate(
        d for d in usable(source, settings) if is_validation(d, valid_every)
    ):
        if i >= EVAL_DOCUMENTS:
            break

        metrics.add_text(document.text, baseline.restore(document.text))

    summary = metrics.summary()
    cached.write_text(json.dumps(summary))
    log(logger, "baseline", **summary)


def run_eval(settings: Settings, source: WikipediaSource, store: LocalDirStore) -> None:
    model, alphabet, overlap, threshold = load_export(store, BEST_EXPORT)
    predictor = Predictor(model, alphabet, overlap, threshold)
    valid_every = settings.corpus.valid_every

    metrics = CandidateMetrics()
    for i, document in enumerate(
        d for d in usable(source, settings) if is_validation(d, valid_every)
    ):
        if i >= EVAL_DOCUMENTS:
            break

        metrics.add_text(document.text, predictor.restore(document.text))

    log(logger, "eval", threshold=threshold, **metrics.summary())


def write_fixture(directory: str) -> None:
    """A tiny deterministic model plus its outputs on sample text, so the Rust forward
    pass can be checked against PyTorch without a real training run."""
    import torch

    torch.manual_seed(0)
    alphabet = Alphabet(tuple(" .,abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    cfg = ModelConfig(
        vocab_size=alphabet.size, window=16, d_model=16, n_layers=2, n_heads=2, d_ff=32, dropout=0.0
    )
    model = CharEncoder(cfg).eval()
    overlap = 4
    store = LocalDirStore(__import__("pathlib").Path(directory))
    export(store, ".", model, alphabet, overlap, threshold=0.5)

    predictor = Predictor(model, alphabet, overlap, threshold=0.5)
    texts = ["Langa casa mea e casa ta si e o casa foarte frumoasa.", "Ana are mere.", "x"]
    cases = [
        {
            "text": text,
            "predictions": [[int(p.label), p.confidence] for p in predictor.predict(text)],
            "restored": predictor.restore(text),
        }
        for text in texts
    ]
    store.write("cases.json", json.dumps(cases, ensure_ascii=False, indent=2).encode())


if __name__ == "__main__":
    sys.exit(main())
