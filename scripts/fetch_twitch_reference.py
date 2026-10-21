"""Collect a small, real Twitch chat reference corpus for M3.

The M3 acceptance in PLAN.md §12 needs >= 200 real Twitch chat lines before the
dynamics engine starts. This script fetches the public Hugging Face dataset
``S1lver404/twitch_chat`` and keeps short, human-looking lines. It is a style
reference only: nothing here is used as training data, and no usernames or
channel identifiers are stored.

Usage:
    .venv/bin/python scripts/fetch_twitch_reference.py --target 240
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urlencode

import httpx

DATASET = "S1lver404/twitch_chat"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET}"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "reference"

_URL = re.compile(r"https?://|discord\.gg|youtu\.be", re.IGNORECASE)
_BOT_PREFIXES = ("[", "%", "!", "/", ".")
_SKIP_MARKERS = (
    "you answered the question correctly",
    "official merch store",
    "amazon prime account",
    "check out ",
)


def _clean(message: str) -> str | None:
    text = " ".join(message.split()).strip()
    if not text or len(text) > 180:
        return None
    lowered = text.casefold()
    if text.startswith(_BOT_PREFIXES) or _URL.search(text):
        return None
    if any(marker in lowered for marker in _SKIP_MARKERS):
        return None
    if lowered in {"first", "!first"}:
        # Keep "first" available for the stream-start rule, but do not let the
        # reference corpus imply that it is 1% of normal chat.
        return None
    return text


def _page(offset: int, length: int = 100) -> list[str]:
    query = urlencode(
        {
            "dataset": DATASET,
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": length,
        }
    )
    with httpx.Client(timeout=60.0, trust_env=False) as client:
        response = client.get(
            f"{ROWS_URL}?{query}",
            headers={"User-Agent": "twitch-sim-reference/0.1"},
        )
        response.raise_for_status()
        data = response.json()
    rows = data.get("rows", [])
    out: list[str] = []
    for row in rows:
        value = row.get("row", {}).get("Message")
        if isinstance(value, str):
            cleaned = _clean(value)
            if cleaned:
                out.append(cleaned)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=240)
    args = parser.parse_args()
    if args.target < 200:
        parser.error("PLAN.md requires at least 200 lines")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    lines: list[str] = []
    offset = 0
    while len(lines) < args.target and offset < 2_000:
        batch = _page(offset)
        for text in batch:
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            lines.append(text)
            if len(lines) >= args.target:
                break
        offset += 100
        time.sleep(0.05)

    if len(lines) < 200:
        raise SystemExit(f"only found {len(lines)} usable lines; need 200")

    jsonl_path = OUT_DIR / "twitch_chat_200.jsonl"
    txt_path = OUT_DIR / "twitch_chat_200.txt"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for index, text in enumerate(lines, start=1):
            handle.write(
                json.dumps(
                    {"id": index, "text": text, "source": DATASET},
                    ensure_ascii=False,
                )
                + "\n"
            )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} lines to {jsonl_path} and {txt_path}")
    print(f"source: {DATASET_URL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
