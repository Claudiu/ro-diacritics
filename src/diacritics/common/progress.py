"""Progress reporting: a live bar with ETA on a terminal, periodic log lines otherwise."""

import logging
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from datetime import timedelta
from typing import Protocol

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.progress import Progress as RichLive
from rich.text import Text

from diacritics.common.logging import format_value, log

logger = logging.getLogger(__name__)

LOG_INTERVAL_SECONDS = 30.0


class Bar(Protocol):
    def update(self, completed: int, **stats: float) -> None:
        """Move to `completed` units; `stats` replace the ones shown so far."""


class Progress(Protocol):
    def track(
        self, label: str, total: int, completed: int = 0, transient: bool = False
    ) -> AbstractContextManager[Bar]:
        """A bar for `total` units, already `completed` of them (resumed runs)."""


class _RateColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        speed = task.speed
        if speed is None:
            return Text("-- it/s", style="dim")

        return Text(f"{speed:.1f} it/s", style="magenta")


class _RichBar:
    def __init__(self, live: RichLive, task_id: TaskID) -> None:
        self._live = live
        self._task_id = task_id
        self._stats: dict[str, float] = {}

    def update(self, completed: int, **stats: float) -> None:
        if stats:
            self._stats.update(stats)
            line = " · ".join(f"{k} {format_value(v)}" for k, v in self._stats.items())
            self._live.update(self._task_id, completed=completed, stats=line)
        else:
            self._live.update(self._task_id, completed=completed)


class RichProgress:
    """Live bars pinned under the log lines of the shared console.

    Rich allows one live display at a time, so nested bars (eval inside train) share it.
    """

    def __init__(self, console: Console) -> None:
        self._live = RichLive(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[dim]•"),
            _RateColumn(),
            TextColumn("[dim]•"),
            TimeElapsedColumn(),
            TextColumn("[dim]eta"),
            TimeRemainingColumn(),
            TextColumn("{task.fields[stats]}", style="cyan"),
            console=console,
        )
        self._open = 0

    @contextmanager
    def track(
        self, label: str, total: int, completed: int = 0, transient: bool = False
    ) -> Iterator[Bar]:
        if self._open == 0:
            self._live.start()
        self._open += 1
        task_id = self._live.add_task(label, total=total, completed=completed, stats="")
        try:
            yield _RichBar(self._live, task_id)
        finally:
            if transient:
                self._live.remove_task(task_id)
            self._open -= 1
            if self._open == 0:
                self._live.stop()
                for task_id in list(self._live.task_ids):
                    self._live.remove_task(task_id)


class _LogBar:
    def __init__(self, label: str, total: int, completed: int, interval: float) -> None:
        self._label = label
        self._total = total
        self._first = completed
        self._interval = interval
        self._started = time.monotonic()
        self._last_logged = self._started
        self._stats: dict[str, float] = {}

    def update(self, completed: int, **stats: float) -> None:
        self._stats.update(stats)
        now = time.monotonic()
        if now - self._last_logged < self._interval:
            return

        self._last_logged = now
        rate = (completed - self._first) / max(now - self._started, 1e-9)
        eta = (self._total - completed) / rate if rate > 0 else None
        log(
            logger,
            self._label,
            completed=completed,
            total=self._total,
            per_second=round(rate, 2),
            elapsed=str(timedelta(seconds=round(now - self._started))),
            eta=str(timedelta(seconds=round(eta))) if eta is not None else None,
            **{k: round(v, 6) for k, v in self._stats.items()},
        )


class LogProgress:
    """No live display (pipes, log files): a log line with rate and ETA every `interval`."""

    def __init__(self, interval: float = LOG_INTERVAL_SECONDS) -> None:
        self._interval = interval

    @contextmanager
    def track(
        self, label: str, total: int, completed: int = 0, transient: bool = False
    ) -> Iterator[Bar]:
        yield _LogBar(label, total, completed, self._interval)
