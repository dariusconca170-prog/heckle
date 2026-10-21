from __future__ import annotations

import random

from backend.chatter.personas import CORE_ROSTER
from backend.dynamics.roster import AuthorPool, CrowdRoster


def test_crowd_handles_are_unique_and_not_core() -> None:
    crowd = CrowdRoster(72, seed=4)
    handles = [persona.handle for persona in crowd.personas]
    assert len(handles) == len(set(handles))
    assert set(handles).isdisjoint({persona.handle for persona in CORE_ROSTER})
    assert {persona.archetype for persona in crowd.personas}


def test_author_pool_returns_members_of_the_pool() -> None:
    crowd = CrowdRoster(48, seed=5)
    pool = AuthorPool(crowd=crowd, core_share=1.0, rng=random.Random(1))
    chosen = {pool.choose().persona.handle for _ in range(20)}
    assert chosen
    assert chosen.issubset({persona.handle for persona in CORE_ROSTER})

    crowd_only = AuthorPool(crowd=crowd, core_share=0.0, rng=random.Random(1))
    chosen = {crowd_only.choose().persona.handle for _ in range(20)}
    assert chosen.issubset({persona.handle for persona in crowd.personas})


def test_crowd_message_is_short_and_grounded_in_an_object() -> None:
    crowd = CrowdRoster(32, seed=6)
    persona = crowd.personas[0]
    from backend.perception.scene import SceneState

    scene = SceneState(
        t=1.0,
        summary="a knight holds a sword",
        action="holding sword",
        objects=["sword"],
    )
    text = crowd.message(persona, scene, random.Random(3))
    assert text
    assert len(text.split()) <= 12
