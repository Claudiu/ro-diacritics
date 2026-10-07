# diacritics

Romanian diacritics restoration (`casa` → `casă`, `tara` → `țara`) with a small
character-level transformer. Trained in PyTorch, served from Rust via candle: natively,
and compiled to WASM so the whole thing runs in the browser with no server.

- Input: text without diacritics (or with partial/cedilla ones; they're stripped first).
- Per character, the model predicts one of three classes: none, breve/comma (ă ș ț),
  circumflex (â î). The base letter decides which diacritic that means.
- Decoding applies a change only above a confidence threshold, so a missed diacritic
  is preferred to a wrong one.

## Train (plug and play on a GPU box)

```sh
git clone <repo> && cd diacritics
scripts/train.sh            # uv sync, fetch Romanian Wikipedia, baseline, train, eval, export
```

Needs only [`uv`](https://docs.astral.sh/uv/). `torch` from PyPI ships CUDA on Linux, so
the same lockfile works on a 5090 and on a CPU-only laptop (`DIACRITICS_DEVICE=cpu`).
Re-running resumes from `artifacts/checkpoints/latest.pt`. The result is
`artifacts/export/{config.json,model.safetensors}`.

Every knob is a `DIACRITICS_*` env var; see `.env.example`. Smoke test on CPU:

```sh
DIACRITICS_MAX_DOCUMENTS=2000 DIACRITICS_MAX_STEPS=200 DIACRITICS_EVAL_EVERY=100 scripts/train.sh
```

Commands behind the script: `uv run diacritics fetch | baseline | train | eval | restore`.

## Run

```sh
cargo run --release -p diacritics-cli -- restore --model artifacts/export < input.txt
scripts/build-wasm.sh && scripts/serve-web.sh     # browser demo at http://localhost:8000/
```

The wasm build needs `wasm-pack` (`cargo install wasm-pack`) and the `wasm32-unknown-unknown`
target (`rustup target add wasm32-unknown-unknown`).

## Develop

```sh
scripts/check.sh       # ruff, mypy, pytest, cargo fmt/clippy/test
scripts/parity.sh      # regenerate the PyTorch fixture and check candle reproduces it
scripts/benchmark.sh   # native inference throughput at the production model shape
```

Layout: `src/diacritics/` (Python: corpus, dataset, model, training, export),
`crates/{domain,model,cli,wasm}` (Rust: text rules, candle inference, CLI, browser),
`web/` (demo page), `test/fixtures/` (cases both languages must pass).
See `docs/architecture.md` for the design.
