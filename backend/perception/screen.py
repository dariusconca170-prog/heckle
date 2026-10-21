"""Screen capture behind the same ``FrameSource`` interface (M9, §10.6).

Windows uses ``dxcam`` (DXGI) and other platforms fall back to ``mss``. Both
return raw pixels; ffmpeg performs the downscale/JPEG encode so this module does
not need OpenCV. A frame is the *current* screen, not a look-ahead frame: file
playback can read ahead, a live desktop cannot.

A privacy blocklist is checked against the active window title on every grab.
When it matches, ``frame_at`` raises ``FrameError`` instead of returning pixels,
so nothing from that window reaches the model.
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import subprocess
import sys
from collections.abc import Callable
from typing import Any

from .frames import Frame, FrameError

TitleProvider = Callable[[], str]


def is_blocked(title: str, blocklist: tuple[str, ...] | list[str]) -> bool:
    """Case-insensitive substring match; an empty blocklist blocks nothing."""
    if not title:
        return False
    folded = title.casefold()
    return any(item and item.casefold() in folded for item in blocklist)


def active_window_title() -> str:
    """Best-effort foreground window title. Never raises."""
    if sys.platform == "win32":
        try:
            import win32gui  # type: ignore[import-not-found]

            return str(win32gui.GetWindowText(win32gui.GetForegroundWindow()) or "")
        except Exception:
            return ""
    tool = shutil.which("xdotool")
    if tool:
        try:
            result = subprocess.run(
                [tool, "getactivewindow", "getwindowname"],
                check=False,
                capture_output=True,
                text=True,
                timeout=0.4,
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            return ""
    return ""


class ScreenFrameSource:
    """Grab the current desktop frame, encoded as JPEG."""

    def __init__(
        self,
        *,
        capture: Callable[[], Any] | None = None,
        encoder: Callable[[bytes, int, int, str], Any] | None = None,
        title_provider: TitleProvider | None = None,
        blocklist: tuple[str, ...] | list[str] = (),
        ffmpeg: str = "ffmpeg",
        max_edge: int = 512,
    ) -> None:
        self._capture = capture
        self._encoder = encoder
        self._title_provider = title_provider or active_window_title
        self.blocklist = tuple(blocklist)
        self.ffmpeg = ffmpeg
        self.max_edge = max(64, int(max_edge))
        self._backend = "injected" if capture is not None else ""
        self._camera: Any = None
        self._mss: Any = None
        self._monitor: Any = None
        self.last_title = ""
        self.last_blocked = ""
        self._setup_error = ""

    @property
    def backend(self) -> str:
        return self._backend

    def ensure_ready(self) -> str:
        if self._capture is None and not self._backend:
            self._setup()
        return self._backend

    @property
    def setup_error(self) -> str:
        return self._setup_error

    def _setup(self) -> None:
        if self._capture is not None:
            return
        try:
            import dxcam  # type: ignore[import-not-found]

            self._camera = dxcam.create(output_color="RGB", output_size=(1280, 720))
            if self._camera is not None:
                self._backend = "dxcam"
                return
        except Exception:
            self._camera = None
        try:
            import mss  # type: ignore[import-not-found]

            self._mss = mss.mss()
            self._monitor = self._mss.monitors[1]
            self._backend = "mss"
            return
        except Exception as exc:
            self._backend = "unavailable"
            self._setup_error = (
                f"no screen capture backend ({exc}); install dxcam (Windows) or mss"
            )

    def _grab(self) -> tuple[bytes, int, int, str]:
        if self._capture is not None:
            value = self._capture()
            if not isinstance(value, tuple) or len(value) != 4:
                raise FrameError("screen capture returned an unexpected frame")
            raw, width, height, pix_fmt = value
            return bytes(raw), int(width), int(height), str(pix_fmt)
        if not self._backend:
            self._setup()
        if self._backend == "dxcam" and self._camera is not None:
            frame = self._camera.grab()
            if frame is None:
                frame = self._camera.grab()
            if frame is None:
                raise FrameError("screen capture returned no frame")
            height, width = frame.shape[0], frame.shape[1]
            return frame.tobytes(), int(width), int(height), "rgb24"
        if self._backend == "mss" and self._mss is not None:
            shot = self._mss.grab(self._monitor)
            return bytes(shot.rgb), int(shot.width), int(shot.height), "rgb24"
        raise FrameError(self._setup_error or "screen capture is unavailable")

    async def frame_at(self, t: float) -> Frame:
        title = ""
        with contextlib.suppress(Exception):
            title = str(self._title_provider() or "")
        self.last_title = title
        if is_blocked(title, self.blocklist):
            self.last_blocked = title
            raise FrameError(f"screen capture paused by privacy blocklist: {title}")
        self.last_blocked = ""
        raw, width, height, pix_fmt = self._grab()
        if self._encoder is not None:
            encoded = self._encoder(raw, width, height, pix_fmt)
            if asyncio.iscoroutine(encoded):
                encoded = await encoded
        else:
            encoded = await _encode_jpeg(
                raw,
                width,
                height,
                pix_fmt,
                ffmpeg=self.ffmpeg,
                max_edge=self.max_edge,
            )
        return Frame(t=float(t), data=bytes(encoded), mime="image/jpeg")

    async def close(self) -> None:
        camera = self._camera
        self._camera = None
        if camera is not None:
            with contextlib.suppress(Exception):
                camera.release()
        mss = self._mss
        self._mss = None
        if mss is not None:
            with contextlib.suppress(Exception):
                mss.close()


async def _encode_jpeg(
    raw: bytes,
    width: int,
    height: int,
    pix_fmt: str,
    *,
    ffmpeg: str,
    max_edge: int,
) -> bytes:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix_fmt,
        "-s",
        f"{width}x{height}",
        "-i",
        "pipe:0",
        "-frames:v",
        "1",
        "-vf",
        (
            f"scale={max_edge}:{max_edge}:"
            "force_original_aspect_ratio=decrease"
        ),
        "-q:v",
        "4",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "pipe:1",
    ]
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise FrameError(f"cannot start {ffmpeg}: {exc}") from exc
    stdout, stderr = await process.communicate(raw)
    if process.returncode != 0 or not stdout:
        detail = stderr.decode("utf-8", "replace").strip()
        raise FrameError(f"screen frame encode failed: {detail[-300:]}")
    return stdout


def screen_backend_available() -> bool:
    for name in ("dxcam", "mss"):
        try:
            __import__(name)
            return True
        except Exception:
            continue
    return False


__all__ = [
    "ScreenFrameSource",
    "active_window_title",
    "is_blocked",
    "screen_backend_available",
]
