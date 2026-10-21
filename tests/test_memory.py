from __future__ import annotations

from backend.chatter.memory import (
    MemoryEvent,
    MemoryMessage,
    extract_memories,
    prompt_memory,
)
from backend.llm.prompting import build_chatter_messages


def _messages(handle: str, texts: list[str], *, start: float = 0.0) -> list[MemoryMessage]:
    return [
        MemoryMessage(handle=handle, t=start + index, text=text)
        for index, text in enumerate(texts)
    ]


def test_catchphrase_needs_repetition() -> None:
    messages = _messages("xX_deadlift_Xx", ["W", "W", "nice one"])
    found = {
        candidate.kind: candidate
        for candidate in extract_memories(messages, [], max_per_viewer=10)
    }
    assert found["catchphrase"].value.casefold() == "w"
    assert found["catchphrase"].weight > 0.45


def test_opinion_uses_nearest_noun() -> None:
    messages = _messages("grimm_reaver", ["those trees are trash"])
    opinions = [
        candidate
        for candidate in extract_memories(messages, [])
        if candidate.kind == "opinion"
    ]
    assert opinions
    assert opinions[0].value == "hates trees"


def test_lore_keeper_remembers_salient_events() -> None:
    events = [
        MemoryEvent(t=12.0, kind="fail", intensity=0.9, detail="player falls in the cave"),
        MemoryEvent(t=40.0, kind="clutch", intensity=0.7, detail="player dodges the boss"),
        MemoryEvent(t=40.2, kind="topic_shift", intensity=0.5, detail="camera pans away"),
    ]
    lore = extract_memories([], events, lore_handles=["lore_dump"])
    assert lore
    assert all(candidate.handle == "lore_dump" for candidate in lore)
    values = [candidate.value for candidate in lore]
    assert values[0].startswith("fail:")
    assert not any("topic_shift" in value for value in values)


def test_silent_lore_keeper_gets_stream_fallback() -> None:
    lore = extract_memories([], [], title="cave run", lore_handles=["lore_dump"])
    assert len(lore) == 1
    assert lore[0].kind == "lore"
    assert lore[0].value == "cave run"


def test_memory_is_capped_per_viewer() -> None:
    messages = _messages("xX_deadlift_Xx", ["W", "W", "L", "L", "gg", "gg"])
    found = extract_memories(messages, [], max_per_viewer=2)
    assert len(found) == 2


def test_prompt_memory_respects_budget() -> None:
    messages = _messages("xX_deadlift_Xx", ["W", "W"])
    found = extract_memories(messages, [])
    described = prompt_memory(found, limit=3, max_chars=80)
    assert described
    assert sum(len(piece) for piece in described) <= 80 + 20


def test_chatter_prompt_includes_memory_and_rule() -> None:
    prompt = build_chatter_messages(
        scene={
            "summary": "player hides behind a car",
            "action": "hiding",
            "mood": "tension",
            "surprise": 0.6,
        },
        personas=[
            {
                "handle": "lore_dump",
                "archetype": "lore-keeper",
                "style": "short references to a previous stream",
                "mood": "knowing",
            },
            {
                "handle": "plain_jane",
                "archetype": "normie",
                "style": "plain language",
                "mood": "curious",
            },
        ],
        memory={"lore_dump": ['remembers fail: player falls in the cave']},
        n=2,
        prefix_handles=True,
    )
    user = prompt[1]["content"]
    assert 'memory: remembers fail: player falls in the cave' in user
    assert "old-stream detail" in user
    assert user.count("memory:") == 1
