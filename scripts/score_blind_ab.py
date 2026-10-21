#!/usr/bin/env python3
"""Score one reviewer's answer to the blind A/B packet."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer", choices=("A", "B"), required=True)
    parser.add_argument("--key", type=Path, default=ROOT / "data" / "eval" / "keys" / "blind_ab_key.json")
    parser.add_argument("--reviewer", default="anonymous")
    parser.add_argument("--results", type=Path, default=ROOT / "data" / "eval" / "blind_ab_results.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    key = json.loads(args.key.read_text(encoding="utf-8"))
    correct = args.answer == key["ai_panel"]
    result = {
        "reviewer": args.reviewer,
        "answer": args.answer,
        "ai_panel": key["ai_panel"],
        "correct": correct,
        "recorded_at": datetime.now(UTC).isoformat(),
        "panel_size": key.get("panel_size"),
        "caveat": key.get("caveat", ""),
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with args.results.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result) + "\n")
    print(f"answer={args.answer} ai_panel={key['ai_panel']} correct={correct}")
    print(f"recorded in {args.results}")
    print("One answer is not the M3 gate; the gate is the reviewer's identification accuracy over the packet.")


if __name__ == "__main__":
    main()
