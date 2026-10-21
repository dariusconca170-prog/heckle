#!/usr/bin/env python3
"""Generate a deterministic demo clip at data/media/sample.mp4.

The clip is only for M0 playback/clock testing. Real perception fixtures belong
under data/perception_eval/ (M1). It is intentionally small (~few MB).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "media" / "sample.mp4"


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists() and "--force" not in sys.argv:
        print(f"{OUTPUT} already exists (use --force to regenerate)")
        return 0
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=1280x720:rate=30",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=330:sample_rate=48000",
        "-t",
        "30",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "26",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(OUTPUT),
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print("ffmpeg failed; is ffmpeg installed and on PATH?", file=sys.stderr)
        return result.returncode
    size = OUTPUT.stat().st_size
    print(f"wrote {OUTPUT} ({size / (1024 * 1024):.1f} MB, 30 s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
