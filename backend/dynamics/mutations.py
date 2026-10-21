"""Deterministic humanization applied after text generation (PLAN.md §4.2).

The model is good at content and bad at rhythm. These mutations are seeded and
local: they never call the model, and the scheduler can reproduce a transcript
exactly if it restores the same seed. Keep the transformations small and
readable; they are intentionally not a content filter.
"""

from __future__ import annotations

import random
import string

from ..chatter.guard import MAX_WORDS, word_count
from ..chatter.personas import Persona
from .emote_tables import EMOTE_ARRAY, EMOTES, pick_emote

_ALPHABET = string.ascii_letters


def mutate(
    text: str,
    persona: Persona,
    *,
    rng: random.Random,
    allow_typos: bool = True,
) -> str:
    """Apply length, casing, punctuation, typo and emote rules in place."""
    value = " ".join(str(text).strip().strip("\"'").split())
    if not value:
        value = "LUL"

    if word_count(value) > MAX_WORDS:
        value = " ".join(value.split()[:MAX_WORDS])

    if persona.archetype == "emote-only":
        value = _emote_only(value, rng)

    if persona.caps:
        value = value.upper()
    elif rng.random() < 0.60:
        value = _lower_first(value)

    if rng.random() < 0.80:
        value = value.rstrip(".!")
    if value and value[-1] == "." and rng.random() < 0.9:
        value = value[:-1]

    if allow_typos and value and not _is_only_emotes(value) and rng.random() < 0.06:
        value = _typo(value, rng)
        if rng.random() < 0.02:
            value = _typo(value, rng)

    if persona.archetype != "emote-only":
        value = _add_emotes(value, persona, rng)

    value = " ".join(value.split())
    if word_count(value) > MAX_WORDS:
        value = " ".join(value.split()[:MAX_WORDS])
    return value.strip() or "LUL"


def _emote_only(text: str, rng: random.Random) -> str:
    words = text.split()
    if words and all(_looks_like_emote(word) for word in words):
        return text
    count = 1
    roll = rng.random()
    if roll < 0.08:
        count = rng.randint(3, 7)
    elif roll < 0.30:
        count = 2
    return " ".join(rng.choice(EMOTE_ARRAY) for _ in range(count))


def _add_emotes(text: str, persona: Persona, rng: random.Random) -> str:
    roll = rng.random()
    if roll < 0.08:
        emote = pick_emote(rng, persona.emotes)
        return f"{text} " + " ".join([emote] * rng.randint(3, 5))
    if roll < 0.18:
        return f"{text} {pick_emote(rng, persona.emotes)} {pick_emote(rng, persona.emotes)}"
    if roll < 0.43:
        return f"{text} {pick_emote(rng, persona.emotes)}"
    return text


def _lower_first(text: str) -> str:
    if not text:
        return text
    first_token = text.split(maxsplit=1)[0].strip(string.punctuation)
    # Never lowercase an emote, acronym or number-led name: ``LUL`` must not
    # become ``lUL`` and ``W`` must stay a W.
    if first_token in EMOTES or first_token.isupper() or any(char.isdigit() for char in first_token):
        return text
    first = text[0]
    if not first.isalpha():
        return text
    return first.lower() + text[1:]


def _typo(text: str, rng: random.Random) -> str:
    """Swap, delete, or double one interior character."""
    candidates = [
        index
        for index, char in enumerate(text)
        if char in _ALPHABET and index > 0
    ]
    if not candidates:
        return text
    index = rng.choice(candidates)
    operation = rng.choice(("swap", "delete", "double"))
    if operation == "swap":
        if index + 1 >= len(text) or text[index + 1] not in _ALPHABET:
            return text
        chars = list(text)
        chars[index], chars[index + 1] = chars[index + 1], chars[index]
        return "".join(chars)
    if operation == "delete":
        return text[:index] + text[index + 1 :]
    return text[:index] + text[index] + text[index:]


def _looks_like_emote(token: str) -> bool:
    cleaned = token.strip(string.punctuation)
    if not cleaned:
        return True
    if cleaned in EMOTES:
        return True
    return cleaned.casefold() in {emote.casefold() for emote in EMOTES}


def _is_only_emotes(text: str) -> bool:
    return all(_looks_like_emote(token) for token in text.split())


def copy_pasta_count(rng: random.Random) -> int:
    return rng.randint(3, 6)
