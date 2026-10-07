"""Download a URL to a file atomically: a partial download never looks complete."""

import logging
import shutil
import urllib.request
from pathlib import Path

from diacritics.common.logging import log

CHUNK_BYTES = 1 << 20
TIMEOUT_SECONDS = 60

logger = logging.getLogger(__name__)


def download(url: str, dest: Path) -> None:
    if dest.exists():
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    log(logger, "download start", url=url, dest=str(dest))

    request = urllib.request.Request(url, headers={"User-Agent": "diacritics/0.1"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        with partial.open("wb") as out:
            shutil.copyfileobj(response, out, CHUNK_BYTES)

    partial.replace(dest)
    log(logger, "download done", dest=str(dest), bytes=dest.stat().st_size)
