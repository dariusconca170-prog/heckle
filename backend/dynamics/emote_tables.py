"""Emote tables and crowd-tier chat templates (PLAN.md §4.3).

Crowd personas never touch the GPU. They provide the noise floor: short
reactions, emote spam, copy-pasta seeds and ``first``. Everything here is
deliberately cheap and deterministic; the scheduler decides which line is used
and when it appears.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

EMOTES: tuple[str, ...] = (
    "LUL",
    "KEKW",
    "Pog",
    "PogChamp",
    "monkaS",
    "Sadge",
    "Copium",
    "PepeHands",
    "catJAM",
    "Kreygasm",
    "BibleThump",
    "WeirdChamp",
    "5Head",
    "EZ",
    "Clap",
    "o7",
    "GIGACHAD",
    "<3",
    "W",
    "L",
    "?",
    "?!",
)

EMOTE_ARRAY: tuple[str, ...] = (
    "LUL",
    "KEKW",
    "monkaS",
    "Sadge",
    "Copium",
    "catJAM",
    "PogChamp",
    "PepeHands",
    "5Head",
    "Clap",
)

# Shares from PLAN.md §4.4. Core personas cover the first 11 archetypes; crowd
# personas are generated from the same table so a viewer who appears once can
# appear again with a consistent voice.
ARCHETYPE_SHARES: tuple[tuple[str, int], ...] = (
    ("hype", 12),
    ("meme", 12),
    ("wholesome", 10),
    ("angry", 8),
    ("doomer", 7),
    ("backseater", 10),
    ("lurker", 15),
    ("normie", 10),
    ("emote-only", 8),
    ("lore-keeper", 4),
    ("parser", 4),
)

TEMPLATES: dict[str, tuple[str, ...]] = {
    "hype": (
        "LETS GO",
        "W",
        "POG",
        "W W W",
        "we are so back",
        "LETS GOOO",
        "HUGE",
    ),
    "meme": (
        "bro",
        "im dead",
        "what",
        "no way",
        "it's so over",
        "we're so back",
        "the {object} is judging me",
        "this is a vibe",
    ),
    "wholesome": (
        "gg",
        "you got this",
        "nice",
        "good run",
        "<3",
        "proud of you",
    ),
    "angry": (
        "L",
        "trash",
        "uninstall",
        "this game is cooked",
        "why would you do that",
        "painful to watch",
    ),
    "doomer": (
        "it's joever",
        "we're cooked",
        "this is so over",
        "copium",
        "saw that coming",
        "it was over before it started",
    ),
    "backseater": (
        "should have waited",
        "classic",
        "why push there",
        "told you",
        "bad timing",
        "that was the moment",
    ),
    "lurker": (
        "lol",
        "W",
        "same",
        "hm",
        "true",
        "ok",
    ),
    "normie": (
        "what happened",
        "what is this",
        "why",
        "is that the {object}",
        "wait what",
        "who is winning",
    ),
    "emote-only": EMOTES,
    "lore-keeper": (
        "just like last stream",
        "remember when",
        "the {object} arc continues",
        "we have seen this before",
        "this is the same as last time",
    ),
    "parser": (
        "typo",
        "ratio",
        "grammar",
        "based",
        "you spelled it wrong",
        "source?",
    ),
    "incongruent": (
        "the moon is loud",
        "i can hear colors",
        "the {object} knows",
        "this is a dream",
        "numbers taste weird",
    ),
}

# A few handles for the stream-start ritual and crowd-only messages. The crowd
# generator produces the full set; these are stable names for special events.
SPECIAL_HANDLES = (
    "first_timer",
    "pixel_goblin",
    "toast_enjoyer",
    "raccoon_king",
    "noodle_incident",
    "moss_cam",
    "the_real_deal",
)


@dataclass(slots=True, frozen=True)
class CrowdName:
    handle: str
    archetype: str


def pick_emote(rng: random.Random, persona_emotes: tuple[str, ...] = ()) -> str:
    if persona_emotes and rng.random() < 0.6:
        return rng.choice(persona_emotes)
    return rng.choice(EMOTES)


def emote_wall(rng: random.Random) -> str:
    return rng.choice(EMOTE_ARRAY)


def crowd_line(
    archetype: str,
    *,
    scene_objects: list[str] | None = None,
    rng: random.Random | None = None,
) -> str:
    rng = rng or random.Random()
    choices = TEMPLATES.get(archetype) or TEMPLATES["normie"]
    template = rng.choice(choices)
    objects = scene_objects or ["thing"]
    obj = rng.choice(objects) if objects else "thing"
    text = template.format(object=obj.lower())
    if archetype == "emote-only":
        if rng.random() < 0.08:
            return " ".join(rng.choice(EMOTE_ARRAY) for _ in range(rng.randint(3, 12)))
        if rng.random() < 0.3:
            return f"{text} {rng.choice(EMOTE_ARRAY)}"
    return text


def weighted_archetype(rng: random.Random) -> str:
    total = sum(share for _, share in ARCHETYPE_SHARES)
    ticket = rng.uniform(0.0, total)
    cursor = 0.0
    for archetype, share in ARCHETYPE_SHARES:
        cursor += share
        if ticket <= cursor:
            return archetype
    return ARCHETYPE_SHARES[-1][0]
