"""Typed reads of environment variables with a shared prefix."""

import os

PREFIX = "DIACRITICS_"


def read_str(name: str, default: str) -> str:
    return os.environ.get(PREFIX + name, default)


def read_int(name: str, default: int) -> int:
    raw = os.environ.get(PREFIX + name)
    if raw is None:
        return default

    try:
        return int(raw)
    except ValueError as err:
        raise ValueError(f"{PREFIX}{name} must be an integer, got {raw!r}") from err


def read_float(name: str, default: float) -> float:
    raw = os.environ.get(PREFIX + name)
    if raw is None:
        return default

    try:
        return float(raw)
    except ValueError as err:
        raise ValueError(f"{PREFIX}{name} must be a number, got {raw!r}") from err
