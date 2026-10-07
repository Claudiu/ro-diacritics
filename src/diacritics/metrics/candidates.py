"""Precision and recall per class on candidate letters, plus word accuracy.

Accuracy over all characters is meaningless here (most can never carry a diacritic),
so every number is computed on a, i, s, t only.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from diacritics.domain.label import LABEL_NAMES, Label
from diacritics.domain.normalize import is_candidate, labels_of, strip_diacritics


@dataclass(slots=True)
class CandidateMetrics:
    """Confusion counts, accumulated incrementally; `confusion[true][pred]`."""

    confusion: list[list[int]] = field(default_factory=lambda: [[0] * len(Label) for _ in Label])
    words: int = 0
    words_correct: int = 0

    def add(self, true: Sequence[int], pred: Sequence[int]) -> None:
        for t, p in zip(true, pred, strict=True):
            self.confusion[t][p] += 1

    def add_text(self, reference: str, restored: str) -> None:
        """Compare restored text against the reference, on candidate letters and words."""
        if len(reference) != len(restored):
            raise ValueError("reference and restored text differ in length")

        true = labels_of(reference)
        pred = labels_of(restored)
        for ch, t, p in zip(strip_diacritics(reference), true, pred, strict=True):
            if is_candidate(ch):
                self.confusion[t][p] += 1

        for ref_word, out_word in zip(reference.split(), restored.split(), strict=True):
            self.words += 1
            self.words_correct += ref_word == out_word

    def precision(self, label: Label) -> float:
        predicted = sum(row[label] for row in self.confusion)

        return self.confusion[label][label] / predicted if predicted else 0.0

    def recall(self, label: Label) -> float:
        actual = sum(self.confusion[label])

        return self.confusion[label][label] / actual if actual else 0.0

    def accuracy(self) -> float:
        total = sum(map(sum, self.confusion))
        correct = sum(self.confusion[i][i] for i in range(len(Label)))

        return correct / total if total else 0.0

    def word_accuracy(self) -> float:
        return self.words_correct / self.words if self.words else 0.0

    def summary(self) -> dict[str, float]:
        out = {"candidate_accuracy": self.accuracy()}
        for label, name in zip(Label, LABEL_NAMES, strict=True):
            out[f"precision_{name}"] = self.precision(label)
            out[f"recall_{name}"] = self.recall(label)
        if self.words:
            out["word_accuracy"] = self.word_accuracy()

        return out
