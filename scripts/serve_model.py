#!/usr/bin/env python3
"""Run the winning llama-server profile in the foreground.

Used by the app's launch script once M0.5 has written data/model_profile.json.
The process stays alive until Ctrl+C, then shuts the sidecar down cleanly.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import Settings  # noqa: E402
from backend.llm.profile import ModelProfile, load_profile  # noqa: E402
from backend.llm.server_manager import ServerManager  # noqa: E402


async def run(profile_path: str, executable: str | None) -> int:
    profile: ModelProfile = load_profile(profile_path)
    manager = ServerManager(profile, executable=executable)
    info = await manager.start()
    print(
        f"llama-server ready: {manager.base_url} profile={info.profile} "
        f"pid={info.pid} startup={info.startup_s:.1f}s "
        f"mmproj_on_gpu={info.mmproj_on_gpu} slots={info.slots} ctx/slot={info.ctx_per_slot}",
        flush=True,
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):  # Windows proactor loop
            loop.add_signal_handler(sig, stop.set)
    try:
        await stop.wait()
    finally:
        await manager.stop()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "profile",
        nargs="?",
        default=str(ROOT / "data" / "model_profile.json"),
        help="winner JSON or a config/models/<name>.toml profile",
    )
    parser.add_argument("--llama-server")
    args = parser.parse_args()
    # ``--llama-server`` wins; otherwise honor TWITCHSIM_LLAMA_SERVER from the
    # environment or .env, which is how every other setting is overridden.
    executable = args.llama_server or Settings().llama_server
    try:
        return asyncio.run(run(args.profile, executable))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
