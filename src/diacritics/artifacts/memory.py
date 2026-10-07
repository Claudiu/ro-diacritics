"""In-memory artifact store: the fake used by tests."""


class InMemoryStore:
    def __init__(self) -> None:
        self._blobs: dict[str, bytes] = {}

    def write(self, name: str, data: bytes) -> None:
        self._blobs[name] = data

    def read(self, name: str) -> bytes | None:
        return self._blobs.get(name)
