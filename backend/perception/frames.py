"""Frame acquisition behind one interface.

File playback is the M1 source. Screen capture (M9) will implement the same
``FrameSource`` protocol, so perception, the observer and the timeline never
need to know where a frame came from.

Frames are downscaled so the long edge is at most ``max_edge`` pixels before
they cross the process boundary. The observer only needs enough detail to name
the action; sending a 4K frame would burn prefill time for nothing.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class FrameError(RuntimeError):
    """A frame could not be extracted from the source."""


@dataclass(slots=True, frozen=True)
class Frame:
    t: float
    data: bytes
    mime: str = "image/jpeg"

    @property
    def size(self) -> int:
        return len(self.data)


class FrameSource(Protocol):
    """A source that can produce the frame at a playback timestamp."""

    async def frame_at(self, t: float) -> Frame: ...

    async def close(self) -> None: ...


class FileFrameSource:
    """Extract one frame from a local video with ffmpeg."""

    def __init__(
        self,
        path: Path,
        *,
        ffmpeg: str = "ffmpeg",
        max_edge: int = 512,
        quality: int = 4,
        timeout_s: float = 15.0,
    ) -> None:
        self.path = Path(path)
        self.ffmpeg = ffmpeg
        self.max_edge = max(64, int(max_edge))
        self.quality = min(max(int(quality), 2), 31)
        self.timeout_s = timeout_s

    def command(self, t: float) -> list[str]:
        # -ss before -i is the fast seek path. ffmpeg still decodes from the
        # preceding keyframe and discards up to the requested timestamp, so the
        # emitted frame is the frame at t rather than the keyframe.
        return [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-ss",
            f"{max(0.0, float(t)):.3f}",
            "-i",
            str(self.path),
            "-frames:v",
            "1",
            "-an",
            "-sn",
            "-dn",
            "-vf",
            (
                f"scale={self.max_edge}:{self.max_edge}:"
                "force_original_aspect_ratio=decrease"
            ),
            "-q:v",
            str(self.quality),
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "pipe:1",
        ]

    async def frame_at(self, t: float) -> Frame:
        if not self.path.is_file():
            raise FrameError(f"video not found: {self.path}")
        try:
            process = await asyncio.create_subprocess_exec(
                *self.command(t),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise FrameError(f"cannot start {self.ffmpeg}: {exc}") from exc

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout_s
            )
        except TimeoutError as exc:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            with contextlib.suppress(Exception):
                await process.wait()
            raise FrameError(f"ffmpeg timed out extracting frame at {t:.2f}s") from exc

        if process.returncode != 0 or not stdout:
            detail = stderr.decode("utf-8", "replace").strip()
            raise FrameError(
                f"ffmpeg could not extract frame at {t:.2f}s"
                + (f": {detail[-400:]}" if detail else "")
            )
        return Frame(t=float(t), data=stdout, mime="image/jpeg")

    async def close(self) -> None:
        # One short-lived ffmpeg process per frame; nothing to keep open yet.
        return None
