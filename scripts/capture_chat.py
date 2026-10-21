#!/usr/bin/env python3
"""Record a live chat transcript against the playback clock.

This is an evaluation harness, not part of the runtime. It drives the same
WebSocket clock the browser sends, records the messages the dynamics service
actually releases, and writes a text transcript plus machine-readable metadata.

Example:
    .venv/bin/python scripts/capture_chat.py --duration 180 \
        --media big_buck_bunny_180s.mp4 --out data/eval/ai_chat_3min
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import contextlib
import json
import time
from pathlib import Path
from typing import Any

import httpx
import websockets

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "eval" / "ai_chat_3min"


async def capture(
    *,
    base_url: str,
    media_id: str,
    duration: float,
    viewer_count: int,
    title: str,
    category: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    scenes: list[dict[str, Any]] = []
    levels: list[dict[str, Any]] = []
    catching_up = 0

    async with httpx.AsyncClient(base_url=base_url, trust_env=False, timeout=60) as client:
        # Make a repeatable run even if a previous session was left behind.
        with contextlib.suppress(httpx.HTTPError):
            await client.delete("/api/session")
        response = await client.post(
            "/api/session",
            json={
                "media_id": media_id,
                "title": title,
                "category": category,
                "viewer_count": viewer_count,
            },
        )
        response.raise_for_status()
        session = response.json()["session"]
        print(f"session {session['id']} · {media_id} · {duration:.0f}s")

    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
    try:
        async with websockets.connect(
            ws_url,
            max_size=4_000_000,
            ping_interval=10,
            ping_timeout=20,
            close_timeout=3,
        ) as ws:
            hello = json.loads(await ws.recv())
            print(f"connected · dynamics={hello.get('dynamics', {}).get('status', 'unknown')}")

            async def send_clock() -> None:
                started = time.monotonic()
                while True:
                    t_play = min(duration - 0.05, time.monotonic() - started)
                    await ws.send(
                        json.dumps(
                            {
                                "type": "time",
                                "t_play": t_play,
                                "playing": True,
                                "rate": 1.0,
                                "duration": duration,
                            }
                        )
                    )
                    if t_play >= duration - 0.1:
                        break
                    await asyncio.sleep(0.25)

            sender = asyncio.create_task(send_clock())
            started = time.monotonic()
            try:
                while time.monotonic() - started < duration + 6:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    except TimeoutError:
                        continue
                    message = json.loads(raw)
                    if message.get("type") != "event":
                        continue
                    event_type = message.get("event_type")
                    payload = message.get("payload", {})
                    if event_type == "chat.message":
                        messages.append(payload)
                    elif event_type == "scene.update":
                        scenes.append(payload)
                        scene = payload.get("scene", {})
                        if scene:
                            print(f"  scene {scene.get('t', 0):6.1f} · {scene.get('summary', '')[:80]}")
                    elif event_type == "chat.buffer_level":
                        levels.append(payload)
                    elif event_type == "chat.catching_up":
                        catching_up += 1
            finally:
                sender.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await sender
                await ws.close()
    finally:
        async with httpx.AsyncClient(base_url=base_url, trust_env=False, timeout=60) as client:
            dynamics = (await client.get("/api/dynamics")).json()
            health = (await client.get("/health")).json()
            await client.delete("/api/session")

    summary = {
        "session_id": session["id"],
        "media_id": media_id,
        "duration_s": duration,
        "viewer_count": viewer_count,
        "messages": len(messages),
        "message_rate": round(len(messages) / duration, 3) if duration else 0.0,
        "scenes": len(scenes),
        "catching_up_events": catching_up,
        "dynamics": dynamics,
        "perception": health.get("perception", {}),
        "bins_5s": [
            {"start_s": start, "count": count}
            for start, count in sorted(
                collections.Counter(int(float(m.get("release_t", 0)) // 5) * 5 for m in messages).items()
            )
        ],
        "sources": dict(collections.Counter(str(m.get("source", "")) for m in messages)),
        "handles": len({str(m.get("handle", "")) for m in messages}),
    }
    return messages, scenes, levels, summary


def write_outputs(
    out: Path,
    messages: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    levels: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    duration: float,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    messages = [m for m in messages if float(m.get("release_t", 0)) <= duration + 0.01]
    with (out.with_suffix(".jsonl")).open("w", encoding="utf-8") as handle:
        for message in messages:
            handle.write(json.dumps(message, ensure_ascii=False) + "\n")
    with (out.with_suffix(".txt")).open("w", encoding="utf-8") as handle:
        for message in messages:
            text = " ".join(str(message.get("text", "")).split())
            if text:
                handle.write(text + "\n")
    with (out.with_suffix(".scenes.jsonl")).open("w", encoding="utf-8") as handle:
        for payload in scenes:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    with (out.with_suffix(".levels.jsonl")).open("w", encoding="utf-8") as handle:
        for payload in levels:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    summary = {**summary, "messages": len(messages)}
    (out.with_suffix(".summary.json")).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--media", default="big_buck_bunny_180s.mp4")
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--viewers", type=int, default=1284)
    parser.add_argument("--title", default="Big Buck Bunny")
    parser.add_argument("--category", default="animation")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    messages, scenes, levels, summary = asyncio.run(
        capture(
            base_url=args.base_url,
            media_id=args.media,
            duration=args.duration,
            viewer_count=args.viewers,
            title=args.title,
            category=args.category,
        )
    )
    write_outputs(args.out, messages, scenes, levels, summary, duration=args.duration)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {args.out.with_suffix('.txt')}")
    print(f"wrote {args.out.with_suffix('.summary.json')}")


if __name__ == "__main__":
    main()
