from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from backend.perception.frames import FileFrameSource, FrameError

FFMPEG = shutil.which("ffmpeg")


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg is not installed")
async def test_file_frame_source_extracts_a_downscaled_jpeg(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=10",
            "-t",
            "2",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ],
        check=True,
    )
    source = FileFrameSource(clip, ffmpeg=FFMPEG, max_edge=128)
    frame = await source.frame_at(1.0)
    assert frame.data.startswith(b"\xff\xd8")
    assert frame.mime == "image/jpeg"
    assert frame.t == 1.0
    assert 0 < len(frame.data) < 100_000
    assert "scale=128:128" in " ".join(source.command(1.0))


async def test_file_frame_source_errors_cleanly(tmp_path: Path) -> None:
    source = FileFrameSource(tmp_path / "missing.mp4", ffmpeg=FFMPEG or "ffmpeg")
    with pytest.raises(FrameError):
        await source.frame_at(1.0)
