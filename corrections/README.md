# Known mistakes

Sentences the model got wrong, written out correctly. Training mixes them in on top of
Wikipedia, repeated `DIACRITICS_CORRECTIONS_REPEAT` times per epoch (default 20), so the
model sees these words in context far more often than the corpus alone would show them.

- One sentence per line, with correct diacritics (ș and ț with comma below; ş/ţ with a
  cedilla are converted automatically).
- Any `*.txt` file in this directory counts. Blank lines and lines starting with `#` are
  skipped.
- Write the whole sentence, not just the word: most mistakes are pairs like masa/masă or
  rău/râu that only context can settle.

Workflow:

```sh
uv run diacritics corrections      # which known mistakes the exported model still makes
scripts/train.sh                   # fetch rebuilds the corrections shard, training uses it
```

Every eval during training logs `corrections_accuracy` next to the validation numbers.
It is measured on training data, so it shows whether the mistakes are learned, not how
well the model generalises; the best model is still picked on validation alone.
