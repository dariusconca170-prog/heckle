from __future__ import annotations

from backend.llm.prompting import build_chatter_messages, load_voice_personas


def test_chatter_prompt_contains_scene_and_personas() -> None:
    personas = load_voice_personas()[:3]
    messages = build_chatter_messages(
        scene={
            "summary": "player hides behind a car",
            "action": "hiding",
            "mood": "tense",
            "surprise": 0.7,
        },
        personas=personas,
        recent_events=["a door slammed"],
        recent_chat=["monkaS"],
        stream={"title": "late night", "category": "horror"},
        n=3,
    )
    assert messages[0]["role"] == "system"
    assert "NOT an assistant" in messages[0]["content"]
    user = messages[1]["content"]
    assert "player hides behind a car" in user
    assert personas[0]["handle"] in user
    assert "Output ONLY a JSON array of 3 strings" in user
    assert "not a description of the viewer" in user


def test_voice_personas_are_distinct() -> None:
    personas = load_voice_personas()
    assert len(personas) >= 8
    assert len({persona["handle"] for persona in personas}) == len(personas)
