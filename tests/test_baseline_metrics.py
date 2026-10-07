import pytest

from diacritics.baseline.frequent import FrequentFormBaseline
from diacritics.corpus.source import Document
from diacritics.domain.label import Label
from diacritics.metrics.candidates import CandidateMetrics


def test_baseline_picks_most_frequent_form_and_keeps_case() -> None:
    baseline = FrequentFormBaseline()
    baseline.fit(iter([Document("1", "casă casă casa Țara țara"), Document("2", "aşa")]))

    assert baseline.size == 3
    assert baseline.restore("Casa tara asa, xyz") == "Casă țara așa, xyz"
    assert baseline.restore("CASA") == "CASĂ"


def test_baseline_survives_lowercase_that_changes_length() -> None:
    baseline = FrequentFormBaseline()
    baseline.fit(iter([Document("1", "İstanbul")]))

    assert baseline.restore("İstanbul") == "İstanbul"


def test_metrics_from_text() -> None:
    metrics = CandidateMetrics()
    metrics.add_text("casă în țara", "casa în tara")
    summary = metrics.summary()

    assert metrics.words == 3
    assert metrics.words_correct == 1
    assert summary["word_accuracy"] == pytest.approx(1 / 3)
    assert metrics.recall(Label.CIRCUMFLEX) == 1.0
    assert metrics.recall(Label.BREVE_COMMA) == 0.0
    assert metrics.precision(Label.NONE) < 1.0


def test_metrics_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        CandidateMetrics().add_text("ab", "a")
