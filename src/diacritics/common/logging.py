"""Logging set up once by the entry point: colored lines on a terminal, JSON lines otherwise.

DIACRITICS_LOG_FORMAT picks `pretty`, `json` or `auto` (pretty when stdout is a terminal).
"""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any, Literal

from rich.console import Console, ConsoleRenderable
from rich.logging import RichHandler
from rich.text import Text

from diacritics.common import env

Format = Literal["pretty", "json"]

_console = Console()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line: dict[str, Any] = {
            "time": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            line.update(extra)
        if record.exc_info:
            line["exception"] = self.formatException(record.exc_info)

        return json.dumps(line, ensure_ascii=False)


class PrettyHandler(RichHandler):
    """Bold message followed by `key=value` fields, values colored by type."""

    def render_message(self, record: logging.LogRecord, message: str) -> ConsoleRenderable:
        text = Text(message, style="bold")
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            for key, value in fields.items():
                text.append(f"  {key}=", style="dim")
                text.append(format_value(value), style=value_style(value))

        return text


def format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, dict):
        return "{" + " ".join(f"{k}={format_value(v)}" for k, v in value.items()) + "}"

    return str(value)


def value_style(value: object) -> str:
    if isinstance(value, bool):
        return "magenta"
    if isinstance(value, int | float):
        return "cyan"
    if isinstance(value, dict):
        return "blue"

    return "green"


def console() -> Console:
    """The console logs and progress bars share, so bars stay pinned below log lines."""
    return _console


def setup(level: str = "info") -> Format:
    fmt = resolve_format(env.read_str("LOG_FORMAT", "auto"), sys.stdout.isatty())
    handler: logging.Handler
    if fmt == "pretty":
        handler = PrettyHandler(
            console=_console,
            show_path=False,
            omit_repeated_times=False,
            log_time_format="[%H:%M:%S]",
            rich_tracebacks=True,
        )
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    return fmt


def resolve_format(requested: str, is_terminal: bool) -> Format:
    match requested:
        case "pretty" | "json":
            return requested
        case "auto":
            return "pretty" if is_terminal else "json"
        case _:
            raise ValueError(
                f"DIACRITICS_LOG_FORMAT must be auto, pretty or json, got {requested!r}"
            )


def log(logger: logging.Logger, message: str, **fields: object) -> None:
    """Log `message` with structured `fields`."""
    logger.info(message, extra={"fields": fields})
