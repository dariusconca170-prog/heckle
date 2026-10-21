#!/usr/bin/env python3
"""Deterministic style and guard checks for a captured chat transcript.

Human ratings still decide whether the voice is good. This script only proves
the cheap gates: length, no refusals/narration, model-message repetition, burst
shape, and source mix. It prints both AI and reference-corpus stats so the
comparison is honest about what it can and cannot measure.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.chatter.guard import classify, dedupe_key, normalize_text  # noqa: E402
from backend.dynamics.emote_tables import EMOTES  # noqa: E402

WORD = re.compile(r"\S+")
TRAILING_PUNCT = re.compile(r"[.!?]$")
EMOTE_LOOKUP = {emote.casefold() for emote in EMOTES if emote not in {"W", "L", "?"}}


def load_ai(path: Path) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                messages.append(json.loads(line))
    return messages


def load_reference(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_emote_only(text: str) -> bool:
    tokens = text.split()
    return bool(tokens) and all(token.strip(".,!?").casefold() in EMOTE_LOOKUP for token in tokens)


def contains_emote(text: str) -> bool:
    return any(token.strip(".,!?").casefold() in EMOTE_LOOKUP for token in text.split())


def text_stats(texts: list[str]) -> dict[str, Any]:
    word_counts = [len(WORD.findall(text)) for text in texts]
    if not word_counts:
        return {"messages": 0}
    return {
        "messages": len(texts),
        "words_mean": round(statistics.mean(word_counts), 2),
        "words_median": statistics.median(word_counts),
        "words_p90": sorted(word_counts)[max(0, math.ceil(len(word_counts) * 0.9) - 1)],
        "words_max": max(word_counts),
        "pct_1_4_words": round(100 * sum(1 for count in word_counts if 1 <= count <= 4) / len(texts), 1),
        "pct_le_12_words": round(100 * sum(1 for count in word_counts if count <= 12) / len(texts), 1),
        "pct_emote_only": round(
            100 * sum(1 for text in texts if is_emote_only(text)) / len(texts), 1
        ),
        "pct_contains_emote": round(
            100 * sum(1 for text in texts if contains_emote(text)) / len(texts), 1
        ),
        "pct_no_trailing_punct": round(100 * sum(1 for text in texts if not TRAILING_PUNCT.search(text)) / len(texts), 1),
        "pct_lowercase_start": round(100 * sum(1 for text in texts if text[:1].islower()) / len(texts), 1),
    }


def repeat_stats(messages: list[dict[str, Any]], *, source_filter: str | None = None) -> dict[str, Any]:
    window = 50
    keys: list[str] = []
    checked = 0
    repeats = 0
    examples: list[str] = []
    for message in messages:
        key = dedupe_key(str(message.get("text", "")))
        if not key:
            continue
        counts = not source_filter or message.get("source") == source_filter
        if counts:
            checked += 1
            if key in keys[-window:]:
                repeats += 1
                if len(examples) < 5:
                    examples.append(str(message.get("text", "")))
        # The guard's recent window contains every visible message, not only
        # the tier being measured.
        keys.append(key)
    return {
        "messages_checked": checked,
        "repeats_within_50": repeats,
        "repeat_pct": round(100 * repeats / checked, 2) if checked else 0.0,
        "examples": examples,
    }


def rate_stats(messages: list[dict[str, Any]]) -> dict[str, Any]:
    times = sorted(float(message.get("release_t", 0)) for message in messages)
    if len(times) < 2:
        return {"bins_5s": [], "max_gap_s": None, "bursts": 0}
    bins = collections.Counter(int(t // 5) * 5 for t in times)
    gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
    return {
        "bins_5s": [{"start_s": start, "count": bins.get(start, 0)} for start in range(0, int(times[-1]) + 5, 5)],
        "max_gap_s": round(max(gaps), 2),
        "gaps_over_10s": sum(1 for gap in gaps if gap > 10),
        "bursts": sum(1 for start in bins if bins[start] >= 8),
        "max_bin": max(bins.values()),
    }


def source_stats(messages: list[dict[str, Any]]) -> dict[str, int]:
    return dict(collections.Counter(str(message.get("source", "")) for message in messages))


def guard_stats(texts: list[str]) -> dict[str, Any]:
    reasons = collections.Counter(classify(text) for text in texts)
    bad = {reason: count for reason, count in reasons.items() if reason != "ok"}
    return {"guard_reasons": dict(reasons), "guard_bad": bad}


def analyze_ai(messages: list[dict[str, Any]]) -> dict[str, Any]:
    texts = [normalize_text(str(message.get("text", ""))) for message in messages]
    texts = [text for text in texts if text]
    model_texts = [
        normalize_text(str(message.get("text", "")))
        for message in messages
        if message.get("source") == "model"
    ]
    model_texts = [text for text in model_texts if text]
    durations = [float(message.get("release_t", 0)) for message in messages]
    duration = max(durations) if durations else 0.0
    return {
        "text": text_stats(texts),
        "model_text": text_stats(model_texts),
        "guard": guard_stats(texts),
        "model_guard": guard_stats(model_texts),
        "repeats_all": repeat_stats(messages),
        "repeats_model": repeat_stats(messages, source_filter="model"),
        "rate": rate_stats(messages),
        "sources": source_stats(messages),
        "handles": len({str(message.get("handle", "")) for message in messages}),
        "message_rate_per_s": round(len(messages) / duration, 3) if duration else 0.0,
    }


def analyze_reference(texts: list[str]) -> dict[str, Any]:
    cleaned = [normalize_text(text) for text in texts]
    cleaned = [text for text in cleaned if text]
    return {"text": text_stats(cleaned), "guard": guard_stats(cleaned)}


def write_markdown(path: Path, ai: dict[str, Any], reference: dict[str, Any], summary: dict[str, Any]) -> None:
    def row(label: str, ai_value: Any, ref_value: Any = "—") -> str:
        return f"| {label} | {ai_value} | {ref_value} |"

    lines = [
        "# Chat style check",
        "",
        "Deterministic checks only. The blind human A/B is separate and this file",
        "does not claim it passed.",
        "",
        "| Metric | AI transcript | Real reference |",
        "| --- | ---: | ---: |",
        row("Messages", ai["text"]["messages"], reference["text"]["messages"]),
        row("Mean words", ai["text"]["words_mean"], reference["text"]["words_mean"]),
        row("Median words", ai["text"]["words_median"], reference["text"]["words_median"]),
        row("p90 words", ai["text"]["words_p90"], reference["text"]["words_p90"]),
        row("Max words", ai["text"]["words_max"], reference["text"]["words_max"]),
        row("1–4 words %", ai["text"]["pct_1_4_words"], reference["text"]["pct_1_4_words"]),
        row("≤ 12 words %", ai["text"]["pct_le_12_words"], reference["text"]["pct_le_12_words"]),
        row("No trailing punctuation %", ai["text"]["pct_no_trailing_punct"], reference["text"]["pct_no_trailing_punct"]),
        row("Lowercase start %", ai["text"]["pct_lowercase_start"], reference["text"]["pct_lowercase_start"]),
        row("Emote-only %", ai["text"]["pct_emote_only"], "n/a"),
        row("Contains known emote %", ai["text"]["pct_contains_emote"], "n/a"),
        "",
        "## AI transcript internals",
        "",
        f"- Message rate: **{ai['message_rate_per_s']:.2f} msg/s** over the captured window",
        f"- Handles: {ai['handles']}",
        f"- Sources: `{json.dumps(ai['sources'])}`",
        f"- 5 s bins: `{json.dumps(ai['rate']['bins_5s'])}`",
        f"- Max silent gap: {ai['rate']['max_gap_s']} s; gaps > 10 s: {ai['rate']['gaps_over_10s']}",
        f"- Bursts (5 s bin ≥ 8): {ai['rate']['bursts']}; largest bin: {ai['rate']['max_bin']}",
        "",
        "## Guard and repetition",
        "",
        f"- All messages: guard reasons `{json.dumps(ai['guard']['guard_reasons'])}`",
        f"- Model messages: guard reasons `{json.dumps(ai['model_guard']['guard_reasons'])}`",
        f"- Model repeats within 50: {ai['repeats_model']['repeats_within_50']} "
        f"({ai['repeats_model']['repeat_pct']}% of {ai['repeats_model']['messages_checked']})",
        f"- All-message repeats within 50: {ai['repeats_all']['repeats_within_50']} "
        f"({ai['repeats_all']['repeat_pct']}%) — copy-pasta is expected to repeat",
        "",
        "## Reference notes",
        "",
        f"- Reference guard reasons: `{json.dumps(reference['guard']['guard_reasons'])}`",
        "- The reference corpus has no per-line timestamps and no known emote dictionary,",
        "  so rate and emote comparisons are not possible from it.",
        "",
        "## Capture summary",
        "",
        "```json",
        json.dumps(summary, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ai", type=Path, default=ROOT / "data" / "eval" / "ai_chat_3min.jsonl")
    parser.add_argument("--reference", type=Path, default=ROOT / "data" / "reference" / "twitch_chat_200.txt")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "eval" / "chat_quality_3min")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    messages = load_ai(args.ai)
    summary_path = args.ai.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    ai = analyze_ai(messages)
    reference = analyze_reference(load_reference(args.reference))
    result = {"ai": ai, "reference": reference, "capture": summary}
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(args.out.with_suffix(".md"), ai, reference, summary)
    print(json.dumps({"ai": ai["text"], "guard": ai["guard"], "model_guard": ai["model_guard"], "repeats": ai["repeats_model"], "rate": ai["rate"]}, indent=2))
    print(f"wrote {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
