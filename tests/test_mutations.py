from __future__ import annotations

import random
import re

from backend.chatter.personas import CORE_ROSTER
from backend.dynamics.emote_tables import EMOTES
from backend.dynamics.mutations import mutate


def test_hype_is_caps_and_short() -> None:
    hype = next(persona for persona in CORE_ROSTER if persona.archetype == "hype")
    text = mutate("this is a very long test line " * 4, hype, rng=random.Random(1))
    assert text == text.upper()
    assert len(text.split()) <= 12


def test_emote_only_does_not_leak_words() -> None:
    persona = next(persona for persona in CORE_ROSTER if persona.archetype == "emote-only")
    text = mutate("this should be removed", persona, rng=random.Random(2))
    assert text
    for token in text.split():
        cleaned = re.sub(r"[^A-Za-z0-9<>?!]", "", token)
        assert not cleaned or cleaned in EMOTES or cleaned.casefold() in {
            emote.casefold() for emote in EMOTES
        }


def test_mutations_are_seeded() -> None:
    persona = next(persona for persona in CORE_ROSTER if persona.archetype == "meme")
    first = [mutate("hello there friend", persona, rng=random.Random(9)) for _ in range(3)]
    second = [mutate("hello there friend", persona, rng=random.Random(9)) for _ in range(3)]
    assert first == second


def test_typos_are_possible_but_do_not_add_words() -> None:
    persona = next(persona for persona in CORE_ROSTER if persona.archetype == "normie")
    results = {
        mutate("something happened here", persona, rng=random.Random(seed))
        for seed in range(40)
    }
    assert len(results) >= 3
    assert all(len(text.split()) <= 12 for text in results)
