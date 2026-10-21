"""Media library discovery and byte-range serving.

The browser never receives an arbitrary filesystem path. It picks a media ID
from the library; the session issues an unguessable token that maps back to the
validated path. Range support is required by HTML5 video (seek, duration).
"""

from __future__ import annotations

import mimetypes
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

VIDEO_MIME_OVERRIDES = {
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".ts": "video/mp2t",
    ".mpg": "video/mpeg",
    ".mpeg": "video/mpeg",
}


class MediaError(ValueError):
    """Invalid media ID or unreadable file."""


class RangeNotSatisfiable(ValueError):
    """The requested byte range cannot be served."""

    def __init__(self, size: int) -> None:
        super().__init__("requested range is outside the file")
        self.size = size


@dataclass(slots=True, frozen=True)
class MediaItem:
    id: str
    name: str
    path: Path
    size: int
    modified: float

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "size": self.size,
            "modified": self.modified,
            "mime": content_type_for(self.path),
        }


def content_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in VIDEO_MIME_OVERRIDES:
        return VIDEO_MIME_OVERRIDES[suffix]
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


class MediaLibrary:
    """Read-only view of the configured media directory."""

    def __init__(self, root: Path, extensions: tuple[str, ...]) -> None:
        self.root = root.resolve()
        self.extensions = tuple(ext.lower() for ext in extensions)

    def refresh(self) -> list[MediaItem]:
        self.root.mkdir(parents=True, exist_ok=True)
        items: list[MediaItem] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            for filename in sorted(filenames):
                if filename.startswith("."):
                    continue
                path = Path(dirpath) / filename
                if path.suffix.lower() not in self.extensions:
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                rel = path.relative_to(self.root).as_posix()
                items.append(
                    MediaItem(
                        id=rel,
                        name=path.stem,
                        path=path,
                        size=stat.st_size,
                        modified=stat.st_mtime,
                    )
                )
        items.sort(key=lambda item: (item.name.casefold(), item.id))
        return items

    def list(self) -> list[MediaItem]:
        return self.refresh()

    def resolve(self, media_id: str) -> MediaItem:
        if not media_id or "\x00" in media_id:
            raise MediaError("missing media id")
        candidate = Path(media_id)
        if candidate.is_absolute():
            raise MediaError("media id must be relative to the library")
        try:
            path = (self.root / candidate).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise MediaError(f"media not found: {media_id}") from exc
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise MediaError("media id escapes the library") from exc
        if not path.is_file() or path.suffix.lower() not in self.extensions:
            raise MediaError(f"not a supported media file: {media_id}")
        stat = path.stat()
        return MediaItem(
            id=path.relative_to(self.root).as_posix(),
            name=path.stem,
            path=path,
            size=stat.st_size,
            modified=stat.st_mtime,
        )


class MediaTokenStore:
    """Session-scoped, unguessable URL tokens for validated media paths."""

    def __init__(self) -> None:
        self._by_token: dict[str, MediaItem] = {}
        self._by_path: dict[Path, str] = {}

    def issue(self, item: MediaItem) -> str:
        existing = self._by_path.get(item.path)
        if existing is not None:
            return existing
        token = secrets.token_urlsafe(24)
        self._by_token[token] = item
        self._by_path[item.path] = token
        return token

    def resolve(self, token: str) -> MediaItem | None:
        return self._by_token.get(token)

    def revoke(self, token: str) -> None:
        item = self._by_token.pop(token, None)
        if item is not None:
            self._by_path.pop(item.path, None)


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """Parse one ``bytes=`` range.

    Returns inclusive ``(start, end)``, or ``None`` for a full response.
    Raises :class:`RangeNotSatisfiable` for invalid/unsatisfiable ranges.
    """
    if not header:
        return None
    value = header.strip()
    if not value.lower().startswith("bytes="):
        return None
    spec = value[6:].strip()
    if not spec or "," in spec:
        raise RangeNotSatisfiable(size)
    if "-" not in spec:
        raise RangeNotSatisfiable(size)
    start_text, end_text = spec.split("-", 1)
    start_text = start_text.strip()
    end_text = end_text.strip()
    try:
        if not start_text:
            suffix = int(end_text)
            if suffix <= 0 or size <= 0:
                raise RangeNotSatisfiable(size)
            start = max(0, size - suffix)
            end = size - 1
        else:
            start = int(start_text)
            if start < 0 or start >= size:
                raise RangeNotSatisfiable(size)
            end = size - 1 if not end_text else int(end_text)
            if end < start:
                raise RangeNotSatisfiable(size)
            end = min(end, size - 1)
    except ValueError as exc:
        if isinstance(exc, RangeNotSatisfiable):
            raise
        raise RangeNotSatisfiable(size) from exc
    return start, end


def content_disposition(path: Path) -> str:
    return f"inline; filename*=UTF-8''{quote(path.name)}"
