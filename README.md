# Diacritificatorul

Romanian diacritics restoration (`casa` → `casă`, `tara` → `țara`) with a small
character-level transformer that runs in the browser. Trained in PyTorch, served from Rust
via [candle](https://github.com/huggingface/candle): natively, and compiled to WebAssembly,
so the text never leaves your machine.

Demo: https://claudiu.github.io/ro-diacritics/ · write-up (Romanian):
https://claudiu.github.io/ro-diacritics/about.html

It is still a PoC.

## Why I picked it up again

In 2018 I built [keras-diacritics](https://github.com/Claudiu/keras-diacritics), a BiLSTM in
Keras behind a Python server. It worked well enough to show the idea had potential, and I
didn't take it further. I had an RTX 2060 then and hit its limits quickly. In 2026 I have an
RTX 5090 and wanted to see what a model trained from scratch can do when it is small enough
to run in the browser: no server, no account.

## 2018 vs. 2026

| | 2018 | 2026 |
|---|---|---|
| Model | BiLSTM | Character-level transformer |
| Parameters | — | 10.8M |
| Training | Keras, RTX 2060 | PyTorch, RTX 5090, ~4 hours |
| Inference | Python server | Browser, Rust + WebAssembly |
| Training data | `train.txt` | Wikipedia, OpenSubtitles, manual corrections |
| Size | — | 6.2 MB, weights quantized to 4 bits |

## How it works

- Input: text without diacritics (partial or cedilla diacritics are stripped first).
- Only `a i s t` (any case) can carry a diacritic. For each of them the model predicts one
  of three classes: `none`, `breve_comma` (ă ș ț) or `circumflex` (â î). The base letter
  decides which diacritic that means.
- The context window is 256 characters, which is what separates „fata” / „fată” / „față”.
  Longer texts are split into windows that overlap by 32 characters; each character is
  predicted by the window where it sits furthest from an edge.
- Decoding applies a change only above a confidence threshold, so a missed diacritic is
  preferred to a wrong one. The output has the same length as the input.

Architecture: pre-LayerNorm transformer encoder, 6 blocks, width 384, 6 heads, feed-forward
384 → 1536 → 384 with GELU, a linear head to 3 classes. Attention is not causal: every
letter sees context on both sides. 10.8M parameters.

Data to browser: Wikipedia + OpenSubtitles (1 in 4 chunks) + manual corrections (×50),
filtered to drop texts without ă/ș/ț and texts with broken encoding → 7.6M training windows
and 262k validation windows → bf16 training, 80,000 steps, ~4 hours → safetensors export
with 4-bit weights → candle in a Web Worker.

## Results

- 99.5% per-letter accuracy on `a i s t` on the validation set, up from 97.8% during
  training. The baseline (the most frequent form of each word) gets 98.4%.
- ~0.5 s for a 500-character text in the browser.
- On 10,223 sentences the first model got wrong, the 1.5M-parameter variant still missed
  5,963; the 10.8M one missed none.
- Adding subtitles cut the share of subtitle lines with at least one mistake from 6.4% to
  3.7% on my evaluation set.

Model size against latency, same text, my PC:

| Parameters | Latency |
|---|---|
| 1.8M | 76 ms |
| 4.7M | 209 ms |
| 10.8M | ~420 ms |
| 25M | 901 ms |

25M was too slow for what I wanted, so it stayed at 10.8M.

Download size:

| Format | Size |
|---|---|
| FP32 | 43 MB |
| FP16 | 21.7 MB |
| 4-bit | 6.2 MB |

4-bit kept the same accuracy on the same validation set: 99.55% in FP32 and 99.55% at 4 bits
(measured before subtitles were added, hence not the 99.5% above). Each matrix is split into
groups of 32 values; each group stores an FP16 `scale = max|w| / 7` and `q = round(w / scale)`
as 4 bits, two per byte (128 B → 18 B per group). The browser rebuilds the weights as
`q × scale` before inference.

## Why not an LLM?

An LLM can add diacritics too, and it does well on ambiguous words where meaning decides
the form. The difference is what you have to run for it:

- **Your text stays with you:** a large LLM runs on a server, so the text goes there. This
  model runs in the browser.
- **Size:** 6.2 MB, downloaded once. A local LLM is hundreds of MB to a few GB and needs a
  GPU or a lot of RAM.
- **It doesn't rewrite the text:** only `a i s t` can change and the output keeps the input's
  length. An LLM can rephrase or skip parts, so its output needs checking.
- **Speed and cost:** ~0.5 s for 500 characters, no account, no API key, no per-request cost.

Where meaning decides the correct form, an LLM can choose better. For the rest, a small
model built only for diacritics is enough.

## What it still gets wrong

- **Ambiguous words:** „peste” / „pește”, „fata” / „fată” / „față”, „sa” / „să”. It picks
  from context and doesn't always get it right.
- **Very short texts:** in „Ai pisat piperul?” there is little context to go on.
- **Names, abbreviations, text mixed with English:** it can add diacritics where none belong.
- **No spellcheck:** it adds diacritics; it doesn't fix other spelling mistakes.

In early tests „Și acum strivim usturoiul pisat” came out as „pișat”, probably because that
form was more frequent in the training data, subtitles included. 17 cooking sentences and a
~30-minute fine-tune fixed it. If you find a mistake,
[open an issue](https://github.com/Claudiu/ro-diacritics/issues/new) with the original
sentence, what the model produced and the correct form; it can go into `corrections/` for
the next fine-tune.

## Train (plug and play on a GPU box)

```sh
git clone https://github.com/Claudiu/ro-diacritics diacritics && cd diacritics
scripts/train.sh            # uv sync, fetch Romanian Wikipedia + subtitles, baseline, train, eval, export
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
