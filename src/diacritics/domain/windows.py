"""Split a long sequence into fixed windows with overlap, and decide which window owns
each position's prediction.

Mirrored in crates/domain/src/windows.rs; test/fixtures/windows.jsonl covers both.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Window:
    """A slice `start..end` of the input. Predictions for `keep_from..keep_to` (absolute
    positions) come from this window; the rest is context shared with its neighbours."""

    start: int
    end: int
    keep_from: int
    keep_to: int


def plan_windows(length: int, window: int, overlap: int) -> list[Window]:
    """Cover `length` positions with windows of size `window`, consecutive ones sharing
    `overlap` positions. The owned ranges partition the input: a position belongs to the
    window where it sits furthest from an edge, so the model sees context on both sides."""
    if window <= 0:
        raise ValueError("window must be positive")
    if not 0 <= overlap < window:
        raise ValueError("overlap must be in [0, window)")
    if length <= 0:
        return []
    if length <= window:
        return [Window(0, length, 0, length)]

    stride = window - overlap
    starts = list(range(0, length - window, stride))
    starts.append(length - window)

    half = overlap // 2
    owners = [0] + [start + half for start in starts[1:]] + [length]

    return [
        Window(start, start + window, owners[i], owners[i + 1]) for i, start in enumerate(starts)
    ]
