import io
import logging

import pytest
from rich.console import Console

from diacritics.common.logging import format_value, resolve_format
from diacritics.common.progress import LogProgress, RichProgress


def test_resolve_format() -> None:
    assert resolve_format("auto", is_terminal=True) == "pretty"
    assert resolve_format("auto", is_terminal=False) == "json"
    assert resolve_format("json", is_terminal=True) == "json"
    with pytest.raises(ValueError, match="LOG_FORMAT"):
        resolve_format("fancy", is_terminal=True)


def test_format_value() -> None:
    assert format_value(0.123456789) == "0.1235"
    assert format_value({"a": 1, "b": 0.5}) == "{a=1 b=0.5}"


def test_log_progress_reports_rate_and_eta(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    with LogProgress(interval=0).track("train", total=100, completed=10) as bar:
        bar.update(20, loss=0.5)
        bar.update(30)

    fields = [r.__dict__["fields"] for r in caplog.records]
    assert fields[-1]["completed"] == 30
    assert fields[-1]["loss"] == 0.5
    assert fields[-1]["eta"] is not None


def test_rich_progress_nests_bars() -> None:
    out = io.StringIO()
    progress = RichProgress(Console(file=out, force_terminal=True, width=160))
    with progress.track("train", total=4) as train:
        train.update(2, loss=0.25)
        with progress.track("eval", total=2, transient=True) as evaluation:
            evaluation.update(2)
        train.update(4, best=0.9)

    assert "train" in out.getvalue()
    assert "best 0.9" in out.getvalue()
