"""Structured persona memory (M4).

Memory is not raw chat history. At the end of a stream (and every few minutes
for crash safety) each viewer's messages are reduced to a handful of durable
facts the model can use next time: catchphrases they repeat, opinions they
voice, and lore they witnessed. That keeps the prompt block small and keeps
the model from parroting old transcript lines.

The extraction is deterministic on purpose: same transcript + same events =
same memory rows, which is what makes the M4 acceptance test repeatable. No
model call, no embeddings (PLAN.md §7.9 keeps those optional).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .guard import normalize_text, word_count

MAX_MEMORY_KINDS = ("opinion", "catchphrase", "grudge", "lore", "preference")

_TOKEN = re.compile(r"[a-zA-Z][a-zA-Z'\-]+")
_STOPWORDS = frozenset(
    {
        "about", "after", "again", "also", "and", "are", "because", "been",
        "before", "bro", "but", "can", "chat", "could", "did", "does", "doing",
        "dont", "even", "ever", "for", "from", "get", "gets", "getting", "got",
        "had", "has", "have", "her", "here", "him", "his", "how", "into", "its",
        "just", "like", "lol", "more", "most", "much", "not", "now", "off",
        "one", "only", "other", "our", "out", "over", "really", "right", "she",
        "should", "some", "still", "such", "than", "that", "the", "their",
        "them", "then", "there", "these", "they", "thing", "things", "this",
        "those", "too", "under", "very", "was", "way", "well", "were", "what",
        "when", "where", "which", "while", "who", "why", "will", "with",
        "would", "you", "your",
    }
)
_POSITIVE = frozenset(
    {
        "based", "best", "clean", "fire", "goat", "good", "great", "like",
        "likes", "love", "loves", "nice", "peak", "sick", "solid", "win",
        "wins", "w",
    }
)
_NEGATIVE = frozenset(
    {
        "ass", "awful", "bad", "cringe", "garbage", "hate", "hates", "l",
        "mid", "sucks", "trash", "worst", "wrong",
    }
)
_SENTIMENT = _POSITIVE | _NEGATIVE
# Words that appear in many memories and in ordinary chat; matching one of them
# is not evidence that a line referenced a memory ("just like last stream"
# must not count as referencing a catchphrase that happens to contain "last").
_GENERIC_MEMORY_WORDS = frozenset(
    {
        "again", "back", "bro", "here", "last", "like", "lore", "memory",
        "next", "over", "right", "says", "still", "stream", "that", "this",
    }
)


@dataclass(slots=True, frozen=True)
class MemoryMessage:
    """One released message, as stored for a stream."""

    handle: str
    t: float
    text: str
    archetype: str = ""


@dataclass(slots=True, frozen=True)
class MemoryEvent:
    """A derived scene event (PLAN.md §5.4) with its playback timestamp."""

    t: float
    kind: str
    intensity: float = 0.5
    detail: str = ""


@dataclass(slots=True, frozen=True)
class MemoryCandidate:
    handle: str
    kind: str
    key: str
    value: str
    weight: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "handle": self.handle,
            "kind": self.kind,
            "key": self.key,
            "value": self.value,
            "weight": round(float(self.weight), 3),
        }


def extract_memories(
    messages: list[MemoryMessage],
    events: list[MemoryEvent],
    *,
    title: str = "",
    lore_handles: tuple[str, ...] | list[str] = (),
    max_per_viewer: int = 10,
    max_total: int = 400,
) -> list[MemoryCandidate]:
    """Reduce one stream to durable persona facts.

    ``lore_handles`` are the viewers who keep lore — they remember salient
    streams even if they stayed quiet, which is what lets the M4 acceptance
    test ("second stream references the first") hold without rigging a chat
    line.
    """
    keepers = {handle for handle in lore_handles if handle}
    grouped: dict[str, list[MemoryMessage]] = {}
    for message in messages:
        if not message.handle or not message.text.strip():
            continue
        grouped.setdefault(message.handle, []).append(message)
    for handle in keepers:
        grouped.setdefault(handle, [])

    salient = sorted(
        (event for event in events if event.intensity >= 0.35 and event.kind != "topic_shift"),
        key=lambda event: (-event.intensity, event.t),
    )[:6]

    found: dict[tuple[str, str, str], MemoryCandidate] = {}

    def add(handle: str, kind: str, key: str, value: str, weight: float) -> None:
        value = normalize_text(value)[:90]
        if not value:
            return
        slot = (handle, kind, key[:80])
        existing = found.get(slot)
        if existing is None or weight > existing.weight:
            found[slot] = MemoryCandidate(
                handle=handle,
                kind=kind,
                key=key[:80],
                value=value,
                weight=max(0.05, min(1.0, float(weight))),
            )

    for handle, viewer_messages in grouped.items():
        keeper = handle in keepers

        # Catchphrase: the same short line used more than once. Emote-only
        # repetition is the most Twitch thing there is, so it counts.
        counts = Counter(
            normalize_text(message.text).casefold()
            for message in viewer_messages
            if normalize_text(message.text)
        )
        catchphrases = 0
        for text_key, count in counts.most_common():
            if catchphrases >= 2 or count < 2:
                continue
            if len(text_key) > 48 or not 1 <= word_count(text_key) <= 8:
                continue
            original = next(
                (
                    normalize_text(message.text)
                    for message in viewer_messages
                    if normalize_text(message.text).casefold() == text_key
                ),
                text_key,
            )
            add(
                handle,
                "catchphrase",
                text_key,
                original,
                min(1.0, 0.45 + 0.15 * count),
            )
            catchphrases += 1

        # Opinion: a sentiment word plus the nearest concrete noun
        # ("those trees are trash" -> hates trees). Cheap and deterministic.
        opinions = 0
        for message in viewer_messages:
            if opinions >= 3:
                break
            tokens = [token.casefold() for token in _TOKEN.findall(message.text)]
            for index, token in enumerate(tokens):
                if token not in _SENTIMENT:
                    continue
                noun = _nearest_noun(tokens, index)
                if noun is None:
                    continue
                sentiment = "likes" if token in _POSITIVE else "hates"
                add(handle, "opinion", f"{noun}:{sentiment}", f"{sentiment} {noun}", 0.35)
                opinions += 1
                break

        # Lore: what actually happened while they were around.
        lore_limit = 4 if keeper else 2
        used = 0
        for event in salient:
            if used >= lore_limit:
                continue
            detail = normalize_text(event.detail)[:80] or event.kind
            add(
                handle,
                "lore",
                f"{event.kind}:{round(event.t)}",
                f"{event.kind}: {detail}",
                min(1.0, 0.45 + 0.35 * float(event.intensity)),
            )
            used += 1
        if keeper and used == 0:
            add(handle, "lore", "stream", title.strip()[:60] or "the last stream", 0.3)

    ordered = sorted(found.values(), key=lambda item: (item.handle, -item.weight, item.key))
    per_viewer: Counter[str] = Counter()
    result: list[MemoryCandidate] = []
    for candidate in ordered:
        if per_viewer[candidate.handle] >= max(1, int(max_per_viewer)):
            continue
        per_viewer[candidate.handle] += 1
        result.append(candidate)
        if len(result) >= max(1, int(max_total)):
            break
    return result


def describe_memory(candidate: MemoryCandidate) -> str:
    """Turn a row into one short prompt fragment."""
    if candidate.kind == "catchphrase":
        return f'often says "{candidate.value}"'
    if candidate.kind == "lore":
        return f"remembers {candidate.value}"
    if candidate.kind == "grudge":
        return f"still salty about {candidate.value}"
    if candidate.kind == "preference":
        return f"keeps bringing up {candidate.value}"
    return candidate.value


def prompt_memory(
    candidates: list[MemoryCandidate],
    *,
    limit: int = 3,
    max_chars: int = 140,
) -> list[str]:
    """Describe the top rows within a small character budget."""
    described: list[str] = []
    used = 0
    for candidate in candidates:
        piece = describe_memory(candidate)
        if not piece:
            continue
        if used + len(piece) > max_chars and described:
            break
        described.append(piece)
        used += len(piece) + 2
        if len(described) >= max(1, int(limit)):
            break
    return described


def mentions_memory(text: str, candidates: list[MemoryCandidate]) -> bool:
    """True when a line actually references stored memory.

    One shared token is enough for lore and opinions ("cave" is specific), but
    catchphrases need the whole phrase or two distinctive tokens: a template
    line that happens to contain "last" is not a reference to "likes last".
    """
    cleaned = normalize_text(text).casefold()
    tokens = {token.casefold() for token in _TOKEN.findall(cleaned)}
    if not tokens:
        return False
    for candidate in candidates:
        distinctive = {
            token.casefold()
            for token in _TOKEN.findall(f"{candidate.key} {candidate.value}")
            if len(token) >= 4
            and token.casefold() not in _STOPWORDS
            and token.casefold() not in _GENERIC_MEMORY_WORDS
        }
        if not distinctive:
            continue
        if candidate.kind == "catchphrase":
            phrase = normalize_text(candidate.value).casefold()
            if len(phrase) >= 10 and phrase in cleaned:
                return True
            if len(distinctive) >= 2 and len(distinctive & tokens) >= 2:
                return True
        elif distinctive & tokens:
            return True
    return False


def lore_reference(candidates: list[MemoryCandidate]) -> str:
    """One in-voice fallback line that references an old stream.

    Used only when the model was explicitly asked to reference memory and
    ignored the request. The line is deliberately short and generic enough to
    sound like chat: "bro still says ..." / "still not over that fail".
    """
    catchphrase = next(
        (
            candidate
            for candidate in candidates
            if candidate.kind == "catchphrase" and 1 <= word_count(candidate.value) <= 8
        ),
        None,
    )
    if catchphrase is not None:
        return f'bro still says "{catchphrase.value}"'
    lore = next((candidate for candidate in candidates if candidate.kind == "lore"), None)
    if lore is None:
        return ""
    kind = lore.key.split(":", 1)[0].strip()
    if kind == "stream":
        return "we were here last stream right"
    return f"still not over that {kind}"


def _nearest_noun(tokens: list[str], index: int, *, window: int = 4) -> str | None:
    """Find the closest concrete-looking token before/after a sentiment word."""
    for distance in range(1, window + 1):
        for position in (index - distance, index + distance):
            if not 0 <= position < len(tokens):
                continue
            token = tokens[position]
            if token in _STOPWORDS or token in _SENTIMENT or len(token) < 4:
                continue
            if token.endswith("ing") or token.endswith("ed"):
                continue
            return token
    return None


__all__ = [
    "MemoryCandidate",
    "MemoryEvent",
    "MemoryMessage",
    "describe_memory",
    "extract_memories",
    "lore_reference",
    "mentions_memory",
    "prompt_memory",
]
