#!/usr/bin/env python3
"""Build a blind A/B packet from the captured AI transcript and real Twitch lines.

The two panels are equal length and contain only visible chat text. No handles,
timestamps, sources or rates are shown. The answer key is written to a separate
subdirectory so the reviewer can read the packet without seeing the answer.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load_ai(path: Path) -> list[str]:
    texts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            text = " ".join(str(payload.get("text", "")).split())
            if text:
                texts.append(text)
    return texts


def load_real(path: Path) -> list[str]:
    return [" ".join(line.split()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evenly_sample(items: list[str], count: int) -> list[str]:
    if count >= len(items):
        return list(items)
    # Keep the whole timeline represented instead of taking only the opening.
    step = len(items) / count
    return [items[min(len(items) - 1, int(index * step))] for index in range(count)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ai", type=Path, default=ROOT / "data" / "eval" / "ai_chat_3min.jsonl")
    parser.add_argument("--real", type=Path, default=ROOT / "data" / "reference" / "twitch_chat_200.txt")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "eval" / "blind_ab.md")
    parser.add_argument("--count", type=int, default=240, help="lines per panel")
    parser.add_argument("--seed", type=int, default=20261020)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ai_all = load_ai(args.ai)
    real_all = load_real(args.real)
    if not ai_all:
        raise SystemExit(f"no AI messages in {args.ai}")
    if not real_all:
        raise SystemExit(f"no real messages in {args.real}")

    count = min(args.count, len(ai_all), len(real_all))
    ai = evenly_sample(ai_all, count)
    real = evenly_sample(real_all, count)

    rng = random.Random(args.seed)
    ai_is_a = rng.random() < 0.5
    panels = {"A": ai if ai_is_a else real, "B": real if ai_is_a else ai}

    lines = [
        "# Blind chat A/B",
        "",
        "One panel is a real Twitch chat sample. One is the local simulator.",
        "Both are the same number of lines and contain visible text only.",
        "",
        "Read both panels and answer **A** or **B**: which one is the simulator?",
        "Do not open `data/eval/keys/blind_ab_key.json` until after answering.",
        "",
        f"Panel size: {count} lines each.",
        "",
        "## Transcript A",
        "",
        "```text",
        *panels["A"],
        "```",
        "",
        "## Transcript B",
        "",
        "```text",
        *panels["B"],
        "```",
        "",
        "## Answer",
        "",
        "Write the panel letter here, then score it with:",
        "",
        "```text",
        ".venv/bin/python scripts/score_blind_ab.py --answer A",
        "```",
        "",
        "Answer: ____",
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")

    key_dir = args.out.parent / "keys"
    key_dir.mkdir(parents=True, exist_ok=True)
    key_path = key_dir / f"{args.out.stem}_key.json"
    key: dict[str, Any] = {
        "seed": args.seed,
        "panel_size": count,
        "ai_panel": "A" if ai_is_a else "B",
        "real_panel": "B" if ai_is_a else "A",
        "ai_source": str(args.ai),
        "real_source": str(args.real),
        "ai_total_messages": len(ai_all),
        "real_total_messages": len(real_all),
        "ai_sampled": len(ai),
        "real_sampled": len(real),
        "caveat": (
            "The AI transcript was generated for Big Buck Bunny; the real corpus is from a "
            "different stream. Content mismatch can reveal the answer. Treat a result as "
            "style evidence only if both panels remain plausible to the reviewer."
        ),
    }
    key_path.write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({count} lines/panel, AI is panel {key['ai_panel']})")
    print(f"wrote {key_path} — keep this closed during review")


if __name__ == "__main__":
    main()
