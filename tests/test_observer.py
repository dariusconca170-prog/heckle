from __future__ import annotations

import time

import pytest

from backend.llm.client import Completion
from backend.perception.frames import Frame
from backend.perception.observer import ObservationError, Observer, scene_from_payload


class FakeClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    async def complete_vision(self, *args, **kwargs) -> Completion:
        self.calls += 1
        return Completion(
            text=self.text,
            model="fake",
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            elapsed_s=0.01,
        )


def test_scene_from_payload_coerces_common_model_shapes() -> None:
    scene = scene_from_payload(
        {
            "summary": "A chef stirs a pan while steam rises from the stove",
            "action": "stirs pan",
            "objects": "pan, stove, steam",
            "mood": "tension",
            "surprise": "0.7",
            "scene_change": "true",
            "confidence": 2,
            "on_screen_text": "  HEAT  ",
        },
        12.5,
    )
    assert scene.t == 12.5
    assert scene.mood == "tense"
    assert scene.surprise == 0.7
    assert scene.scene_change is True
    assert scene.confidence == 1.0
    assert scene.objects == ["pan", "stove", "steam"]
    assert scene.on_screen_text == "HEAT"


def test_scene_from_payload_caps_summary_and_falls_back() -> None:
    scene = scene_from_payload(
        {"summary": " ".join(f"word{i}" for i in range(50)), "action": "does a thing"},
        3.0,
    )
    assert len(scene.summary.split()) == 35
    missing = scene_from_payload({}, 4.0)
    assert missing.summary == "unclear"
    assert missing.action == "scene"
    assert missing.confidence <= 0.25


async def test_observer_runs_the_vision_call_and_parses_json() -> None:
    client = FakeClient(
        '```json\n{"summary":"player jumps over a gap","action":"jumps",'
        '"objects":["platform"],"mood":"funny","surprise":0.5,'
        '"scene_change":false,"on_screen_text":"","confidence":0.9}\n```'
    )
    observer = Observer(client)  # type: ignore[arg-type]
    observation = await observer.observe(Frame(t=5.0, data=b"jpeg"), 5.0)
    assert client.calls == 1
    assert observation.scene.summary == "player jumps over a gap"
    assert observation.scene.mood == "funny"
    assert observation.latency_s >= 0


async def test_observer_rejects_non_json() -> None:
    observer = Observer(FakeClient("I cannot help with that."))  # type: ignore[arg-type]
    with pytest.raises(ObservationError):
        await observer.observe(Frame(t=5.0, data=b"jpeg"), 5.0)


def test_scene_payload_is_not_mutated() -> None:
    payload = {"summary": "a dog runs", "action": "runs"}
    scene = scene_from_payload(payload, time.time())
    assert payload == {"summary": "a dog runs", "action": "runs"}
    assert scene.summary == "a dog runs"
