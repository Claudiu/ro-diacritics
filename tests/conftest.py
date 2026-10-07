import json
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).resolve().parent.parent / "test" / "fixtures"


def load_jsonl(name: str) -> list[dict[str, Any]]:
    lines = (FIXTURES / name).read_text(encoding="utf-8").splitlines()

    return [json.loads(line) for line in lines if line.strip()]


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES
