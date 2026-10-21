"""Reply threads, arguments and @mentions (M5, PLAN.md §10.3).

The model writes text; this module decides *who is replying to whom*. Keeping
that decision here means threads are reproducible from a seed and never depend
on whether the model noticed a question. The only hard rules are structural:

* no self-replies,
* no reply to a message that has not been released yet,
* an argument always ends with a third person telling both sides to stop.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from ..chatter.personas import Persona

MAX_QUOTE_CHARS = 90


@dataclass(slots=True, frozen=True)
class ParentRef:
    """A message that is already visible and can be replied to."""

    id: str
    handle: str
    text: str
    release_t: float
    archetype: str = ""

    @property
    def quote(self) -> str:
        value = " ".join(self.text.split())
        if len(value) <= MAX_QUOTE_CHARS:
            return value
        return value[: MAX_QUOTE_CHARS - 1].rstrip() + "…"


@dataclass(slots=True, frozen=True)
class ArgumentLine:
    """One message in a scripted argument thread."""

    persona: Persona
    text: str
    reply_index: int | None
    delay_s: float


_REPLY_MARKERS = (
    "?",
    "you",
    "bro",
    "ratio",
    "cope",
    "trash",
    "actually",
    "wrong",
    "no way",
    "l take",
    "skill issue",
    "imagine",
)

_ARGUMENT_OPENERS: dict[str, tuple[str, ...]] = {
    "angry": ("that was actually trash", "uninstall already", "that was awful"),
    "backseater": ("why would you push there", "you had time to heal", "that was free"),
    "doomer": ("we're cooked", "this run is over", "it was over before it started"),
    "hype": ("THAT WAS INSANE", "LETS GO", "W play"),
    "wholesome": ("that was actually clean", "you got this", "gg that was close"),
    "meme": ("he really did that", "actual goblin behavior", "no thoughts just vibes"),
}
_ARGUMENT_REPLIES: dict[str, tuple[str, ...]] = {
    "angry": ("no it wasn't", "cope harder", "you're actually blind", "L take"),
    "backseater": ("that is just wrong", "watch the replay", "you missed the cooldown"),
    "doomer": ("doesn't matter we lose", "cope", "still over"),
    "hype": ("NO WAY YOU SAID THAT", "massive L take", "ARE YOU BLIND"),
    "wholesome": ("no need to be mean", "it was a good try", "let people enjoy it"),
    "meme": ("ratio", "this thread is a mess", "i am eating chips for this"),
    "normie": ("what is happening", "why are you two fighting", "it was fine"),
    "parser": ("you spelled that wrong", "ratio", "grammar aside that was bad"),
}
_MELEE_END = ("shut up both of you", "both of you are wrong", "can we talk about the game")
_ARGUMENT_PAIRS = (
    ("angry", "wholesome"),
    ("angry", "backseater"),
    ("backseater", "hype"),
    ("doomer", "hype"),
)
_THIRD_PARTY = ("normie", "meme", "parser", "lurker", "doomer")


def is_replyable(text: str, *, archetype: str = "") -> bool:
    """Whether a line is the kind of thing chat answers (PLAN.md §10.3)."""
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return False
    lowered = cleaned.casefold()
    if "?" in cleaned:
        return True
    if any(marker in lowered for marker in _REPLY_MARKERS):
        return True
    if archetype in {"angry", "backseater", "hype"}:
        return True
    # Short exclamations are not questions, but they are quotable.
    return len(cleaned.split()) >= 4 and cleaned.isascii()


class ReplyPlanner:
    """Seeded reply/mention decisions shared by every author tier."""

    def __init__(
        self,
        rng: random.Random,
        *,
        reply_rate: float = 0.20,
        mention_rate: float = 0.04,
        min_gap_s: float = 1.0,
        max_age_s: float = 30.0,
    ) -> None:
        self.rng = rng
        self.reply_rate = min(1.0, max(0.0, float(reply_rate)))
        self.mention_rate = min(1.0, max(0.0, float(mention_rate)))
        self.min_gap_s = max(0.0, float(min_gap_s))
        self.max_age_s = max(0.0, float(max_age_s))

    def maybe_reply(
        self,
        handle: str,
        release_t: float,
        candidates: list[ParentRef] | tuple[ParentRef, ...],
    ) -> ParentRef | None:
        if self.rng.random() >= self.reply_rate:
            return None
        eligible = [
            parent
            for parent in candidates
            if parent.handle != handle
            and parent.release_t <= release_t - self.min_gap_s
            and release_t - parent.release_t <= self.max_age_s
            and parent.text.strip()
        ]
        if not eligible:
            return None
        weights = [
            3.0 if is_replyable(parent.text, archetype=parent.archetype) else 1.0
            for parent in eligible
        ]
        return self.rng.choices(eligible, weights=weights, k=1)[0]

    def maybe_mention(self, channel: str) -> str | None:
        if not channel:
            return None
        if self.rng.random() >= self.mention_rate:
            return None
        return f"@{channel.lstrip('@')}"


def build_argument(
    rng: random.Random,
    *,
    pairs: tuple[tuple[str, str], ...] = _ARGUMENT_PAIRS,
    personas_by_archetype: dict[str, list[Persona]] | None = None,
    lookup: Callable[[str], Persona | None] | None = None,
    length: int | None = None,
) -> list[ArgumentLine]:
    """Build a 5+ message argument: opener, escalation, third-party stop.

    ``lookup`` maps an archetype to a Persona. ``personas_by_archetype`` is the
    fallback for callers that already have a roster. The returned list is in
    release order; ``reply_index`` points at the line being answered.
    """
    first, second = rng.choice(pairs)
    instigator = _persona_for(first, personas_by_archetype, lookup)
    opponent = _persona_for(second, personas_by_archetype, lookup)
    third_archetypes = [item for item in _THIRD_PARTY if item not in {first, second}]
    third = _persona_for(
        rng.choice(third_archetypes or _THIRD_PARTY),
        personas_by_archetype,
        lookup,
    )
    if instigator.handle in {opponent.handle, third.handle}:
        return []
    if third.handle == opponent.handle:  # pragma: no cover - lookup collision
        return []

    count = length if length is not None else rng.randint(5, 7)
    count = max(5, count)
    lines: list[ArgumentLine] = []
    opener = rng.choice(_ARGUMENT_OPENERS.get(first, ("that was rough",)))
    lines.append(
        ArgumentLine(persona=instigator, text=opener, reply_index=None, delay_s=0.4)
    )
    for index in range(count - 2):
        author = opponent if index % 2 == 0 else instigator
        pool = _ARGUMENT_REPLIES.get(author.archetype, ("no it wasn't",))
        text = rng.choice(pool)
        if text in {line.text for line in lines}:
            text = rng.choice(tuple(pool))
        lines.append(
            ArgumentLine(
                persona=author,
                text=text,
                reply_index=index,
                delay_s=rng.uniform(1.0, 3.6),
            )
        )
    stop = rng.choice(_MELEE_END)
    lines.append(
        ArgumentLine(
            persona=third,
            text=stop,
            reply_index=len(lines) - 1,
            delay_s=rng.uniform(1.2, 2.8),
        )
    )
    return lines


def _persona_for(
    archetype: str,
    personas_by_archetype: dict[str, list[Persona]] | None,
    lookup: Callable[[str], Persona | None] | None,
) -> Persona:
    if lookup is not None:
        found = lookup(archetype)
        if found is not None:
            return found
    if personas_by_archetype:
        pool = personas_by_archetype.get(archetype)
        if pool:
            return pool[0]
    return Persona(archetype, archetype, "short chat", "neutral", 5)


__all__ = [
    "ArgumentLine",
    "ParentRef",
    "ReplyPlanner",
    "build_argument",
    "is_replyable",
]
