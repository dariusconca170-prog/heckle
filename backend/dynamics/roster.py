"""Author identity for the dynamics engine.

The core roster is the twelve hand-written LLM personas. The crowd tier is
generated once at startup and never calls the model. Both share the same
weighted author selection, so a crowd viewer can recur as the same handle.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..chatter.personas import CORE_ROSTER, Persona
from ..perception.scene import SceneState
from .emote_tables import (
    EMOTE_ARRAY,
    SPECIAL_HANDLES,
    crowd_line,
    weighted_archetype,
)

_PREFIXES = (
    "xX",
    "not",
    "the",
    "real",
    "lil",
    "big",
    "just",
    "its",
    "maybe",
    "definitely",
    "slightly",
    "actually",
)
_NOUNS = (
    "toast",
    "goblin",
    "pixel",
    "moth",
    "kettle",
    "noodle",
    "wizard",
    "raccoon",
    "sock",
    "turnip",
    "comet",
    "gremlin",
    "otter",
    "pebble",
    "lantern",
    "waffle",
    "custard",
    "banjo",
    "velvet",
    "static",
)
_SUFFIXES = ("", "_", "69", "92", "404", "_tv", "0", "77", "_uwu")
_STYLES = {
    "hype": "short, loud, emote walls",
    "meme": "lowercase, absurd, non-sequitur",
    "wholesome": "supportive, soft",
    "angry": "hostile at the game, never at the streamer",
    "doomer": "deadpan, pessimistic",
    "backseater": "knows the game better than the streamer",
    "lurker": "one to three words, rare",
    "normie": "plain language, obvious questions",
    "emote-only": "emotes only",
    "lore-keeper": "remembers old streams",
    "parser": "points out typos and ratios",
    "incongruent": "soft surrealism",
}


def _handle(rng: random.Random, used: set[str]) -> str:
    for _ in range(500):
        prefix = rng.choice(_PREFIXES)
        noun = rng.choice(_NOUNS)
        suffix = rng.choice(_SUFFIXES)
        separator = "_" if prefix in {"lil", "big", "just", "its", "the", "real"} and rng.random() < 0.6 else ""
        candidate = f"{prefix}{separator}{noun}{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
    return f"viewer_{rng.randrange(10_000, 99_999)}"


def _weighted_archetype(rng: random.Random) -> str:
    return weighted_archetype(rng)


class CrowdRoster:
    """A stable set of crowd viewers with no GPU cost."""

    def __init__(self, count: int = 72, *, seed: int = 20261020) -> None:
        self.count = max(12, int(count))
        self.rng = random.Random(seed)
        used = {persona.handle for persona in CORE_ROSTER}
        people: list[Persona] = []
        for _index in range(self.count):
            archetype = _weighted_archetype(self.rng)
            handle = _handle(self.rng, used)
            active_weight = {
                "lurker": 5,
                "emote-only": 7,
                "hype": 10,
                "meme": 10,
                "wholesome": 8,
                "angry": 6,
                "doomer": 5,
                "backseater": 7,
                "normie": 8,
                "lore-keeper": 4,
                "parser": 4,
                "incongruent": 5,
            }.get(archetype, 6)
            people.append(
                Persona(
                    handle=handle,
                    archetype=archetype,
                    style=_STYLES.get(archetype, "short chat"),
                    mood="neutral",
                    weight=active_weight,
                    chattiness=round(self.rng.uniform(0.65, 1.35), 2),
                    emotes=tuple(self.rng.sample(EMOTE_ARRAY, k=2)) if archetype != "emote-only" else EMOTE_ARRAY,
                    caps=archetype == "hype",
                )
            )
        self.personas = tuple(people)
        self.by_handle = {persona.handle: persona for persona in people}
        self.specials = SPECIAL_HANDLES

    def message(self, persona: Persona, scene: SceneState | None, rng: random.Random) -> str:
        objects = list(scene.objects) if scene is not None else []
        return crowd_line(persona.archetype, scene_objects=objects, rng=rng)

    def random_persona(self, rng: random.Random, *, archetype: str | None = None) -> Persona:
        pool = self.personas
        if archetype is not None:
            matching = [persona for persona in pool if persona.archetype == archetype]
            if matching:
                pool = tuple(matching)
        return rng.choice(pool)


@dataclass(slots=True)
class AuthorSelection:
    persona: Persona
    tier: str


class AuthorPool:
    """Weighted picker with a core/crowd split and recent-speaker dampening."""

    def __init__(
        self,
        *,
        core: tuple[Persona, ...] = CORE_ROSTER,
        crowd: CrowdRoster | None = None,
        core_share: float = 0.52,
        rng: random.Random | None = None,
    ) -> None:
        self.core = tuple(core)
        self.crowd = crowd or CrowdRoster()
        self.core_share = min(1.0, max(0.0, float(core_share)))
        self.rng = rng or random.Random()
        self._recent: list[str] = []

    def choose(
        self,
        *,
        core_available: bool = True,
        recent_handles: list[str] | None = None,
        force_tier: str | None = None,
    ) -> AuthorSelection:
        tier = force_tier
        if tier is None:
            tier = "core" if core_available and self.rng.random() < self.core_share else "crowd"
        pool = self.core if tier == "core" and core_available else self.crowd.personas
        recent = set(recent_handles or self._recent[-6:])
        weights: list[float] = []
        for persona in pool:
            weight = max(0.01, persona.weight * persona.chattiness)
            if persona.handle in recent:
                weight *= 0.20
            weights.append(weight)
        persona = self.rng.choices(list(pool), weights=weights, k=1)[0]
        self._recent.append(persona.handle)
        if len(self._recent) > 24:
            self._recent = self._recent[-24:]
        return AuthorSelection(persona=persona, tier="core" if persona in self.core else "crowd")

    def choose_crowd(self, *, recent_handles: list[str] | None = None) -> Persona:
        return self.choose(
            core_available=False,
            recent_handles=recent_handles,
            force_tier="crowd",
        ).persona

    def recent_handles(self, limit: int = 8) -> list[str]:
        return self._recent[-max(1, limit) :]
