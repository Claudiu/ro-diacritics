"""Artifact store on a local directory. Writes are atomic (write .tmp, rename)."""

from pathlib import Path

from diacritics.artifacts.store import ArtifactUnavailableError


class LocalDirStore:
    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def write(self, name: str, data: bytes) -> None:
        path = self._path(name)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        except OSError as err:
            raise ArtifactUnavailableError(f"write {path}: {err}") from err

    def read(self, name: str) -> bytes | None:
        path = self._path(name)
        if not path.exists():
            return None

        try:
            return path.read_bytes()
        except OSError as err:
            raise ArtifactUnavailableError(f"read {path}: {err}") from err

    def _path(self, name: str) -> Path:
        path = (self._directory / name).resolve()
        if self._directory.resolve() not in path.parents:
            raise ArtifactUnavailableError(f"artifact name escapes the store: {name!r}")

        return path
