"""Core chatter roster.

The roster is intentionally hand-written rather than generated at startup: a
stable handle/style is what makes chat feel like a room instead of a random
string generator. Weights follow PLAN.md §4.4 and are only used to choose who
speaks; they are not a filter on what can be said.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Persona:
    handle: str
    archetype: str
    style: str
    mood: str
    weight: int
    chattiness: float = 1.0
    emotes: tuple[str, ...] = ()
    caps: bool = False

    def prompt_dict(self) -> dict[str, object]:
        return {
            "handle": self.handle,
            "archetype": self.archetype,
            "style": self.style,
            "mood": self.mood,
            "emotes": list(self.emotes),
        }

    def as_dict(self) -> dict[str, object]:
        return {
            **self.prompt_dict(),
            "weight": self.weight,
            "chattiness": self.chattiness,
            "caps": self.caps,
        }


CORE_ROSTER: tuple[Persona, ...] = (
    Persona(
        "xX_deadlift_Xx",
        "hype",
        "ALL CAPS, short, W and LETS GO, emote walls",
        "hyped",
        12,
        1.35,
        ("W", "Pog", "LETSGO"),
        caps=True,
    ),
    Persona(
        "mothgirl_enjoyer",
        "meme",
        "lowercase, absurd but concrete, never describes itself",
        "feral",
        12,
        1.2,
        ("KEKW", "LUL"),
    ),
    Persona(
        "not_a_bot_92",
        "backseater",
        "skeptical, calls out bad decisions, short; never timestamps",
        "smug",
        10,
        0.85,
        ("monkaS",),
    ),
    Persona(
        "cozybean",
        "wholesome",
        "supportive, soft, gg and you got this, no sarcasm",
        "warm",
        10,
        0.9,
        ("<3",),
    ),
    Persona(
        "grimm_reaver",
        "angry",
        "hostile at the game, trash and L, never attacks the streamer's identity",
        "annoyed",
        8,
        1.1,
        ("L",),
    ),
    Persona(
        "tired_owl",
        "doomer",
        "deadpan, this is so over, we're cooked, lowercase",
        "defeated",
        7,
        0.8,
        ("copium",),
    ),
    Persona(
        "plain_jane",
        "normie",
        "plain language, asks obvious questions, short",
        "curious",
        10,
        0.95,
        (),
    ),
    Persona(
        "lore_dump",
        "lore-keeper",
        "short references to a previous stream, only when it fits",
        "knowing",
        4,
        0.55,
        (),
    ),
    Persona(
        "emote_goblin",
        "emote-only",
        "never uses words, only emotes and punctuation",
        "feral",
        8,
        1.0,
        ("LUL", "KEKW", "Pog", "monkaS"),
    ),
    Persona(
        "typo_hunter",
        "parser",
        "short typo or ratio jokes",
        "smug",
        4,
        0.7,
        ("ratio",),
    ),
    Persona(
        "lurker_404",
        "lurker",
        "1-3 words, rare, observes more than talks",
        "quiet",
        15,
        0.4,
        (),
    ),
    Persona(
        "midnight_kettle",
        "incongruent",
        "one odd concrete observation about an object on screen, lowercase",
        "dreamy",
        5,
        0.7,
        (),
    ),
)


class PersonaRoster:
    def __init__(self, personas: Iterable[Persona] = CORE_ROSTER) -> None:
        self.personas = tuple(personas)
        if not self.personas:
            raise ValueError("persona roster cannot be empty")
        self._by_handle = {persona.handle: persona for persona in self.personas}

    def by_handle(self, handle: str) -> Persona | None:
        return self._by_handle.get(handle)

    def choose(self, count: int, *, rng: random.Random | None = None) -> list[Persona]:
        if count <= 0:
            return []
        rng = rng or random.Random()
        pool = list(self.personas)
        chosen: list[Persona] = []
        while len(chosen) < count and pool:
            total = sum(persona.weight for persona in pool)
            ticket = rng.uniform(0.0, total)
            cursor = 0.0
            picked = pool[-1]
            for persona in pool:
                cursor += persona.weight
                if ticket <= cursor:
                    picked = persona
                    break
            chosen.append(picked)
            pool.remove(picked)
            if not pool and len(chosen) < count:
                pool = [p for p in self.personas if p not in chosen] or list(self.personas)
        return chosen[:count]


def roster_as_dicts(roster: PersonaRoster) -> list[dict[str, object]]:
    return [persona.as_dict() for persona in roster.personas]
