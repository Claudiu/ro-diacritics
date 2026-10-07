"""Restore diacritics in arbitrary-length text with a trained CharEncoder.

The Python reference for crates/model: same windowing, same decoding.
"""

import torch

from diacritics.domain.alphabet import PAD_ID, Alphabet
from diacritics.domain.decode import Prediction, restore
from diacritics.domain.label import Label
from diacritics.domain.normalize import fix_cedilla, strip_diacritics
from diacritics.domain.windows import plan_windows
from diacritics.model.encoder import CharEncoder


class Predictor:
    def __init__(
        self, model: CharEncoder, alphabet: Alphabet, overlap: int, threshold: float
    ) -> None:
        self._model = model
        self._alphabet = alphabet
        self._overlap = overlap
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    @torch.no_grad()
    def predict(self, stripped: str) -> list[Prediction]:
        """One prediction per character of diacritic-free `stripped`."""
        if not stripped:
            return []

        window = self._model.cfg.window
        ids = self._alphabet.encode(stripped)
        windows = plan_windows(len(ids), window, self._overlap)
        device = next(self._model.parameters()).device

        width = min(len(ids), window)
        batch = torch.full((len(windows), width), PAD_ID, dtype=torch.long)
        for i, w in enumerate(windows):
            batch[i, : w.end - w.start] = torch.tensor(ids[w.start : w.end])

        self._model.eval()
        probs = torch.softmax(self._model(batch.to(device)).float(), dim=-1).cpu()

        out: list[Prediction] = []
        for i, w in enumerate(windows):
            local = probs[i, w.keep_from - w.start : w.keep_to - w.start]
            confidence, label = local.max(dim=-1)
            out.extend(
                Prediction(Label(int(label_id)), float(conf))
                for label_id, conf in zip(label.tolist(), confidence.tolist(), strict=True)
            )

        return out

    def restore(self, text: str) -> str:
        """Diacritics restored in `text`; existing ones are stripped first, so the model
        decides every letter."""
        stripped = strip_diacritics(fix_cedilla(text))

        return restore(stripped, self.predict(stripped), self._threshold)
