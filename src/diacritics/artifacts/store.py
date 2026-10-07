"""Port: where checkpoints and exported models are kept."""

from typing import Protocol


class ArtifactStore(Protocol):
    """Named blobs. Local directory now; S3 or a model hub later."""

    def write(self, name: str, data: bytes) -> None: ...

    def read(self, name: str) -> bytes | None:
        """The blob, or None when it does not exist."""
        ...


class ArtifactUnavailableError(Exception):
    """The store could not be read or written."""
