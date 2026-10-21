from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.perception.scene import SceneState


def test_scene_state_contract_round_trip() -> None:
    payload = {
        "t": 412.8,
        "summary": "First-person horror game; player hides behind a rusty car in the rain while two enemies patrol the street",
        "action": "hiding",
        "objects": ["car", "rain", "street", "flashlight beam"],
        "mood": "tense",
        "surprise": 0.72,
        "scene_change": True,
        "on_screen_text": "PRESS E TO HIDE",
        "confidence": 0.84,
        "speech_text": "",
        "speech_energy": 0.0,
        "source": "video_audio",
    }
    scene = SceneState(**payload)
    assert scene.model_dump() == payload
    assert SceneState.model_validate(payload) == scene


def test_scene_state_normalizes_and_dedupes_objects() -> None:
    scene = SceneState(
        t=0,
        summary="  a   cat  ",
        action="cat",
        objects=["cat", " Cat ", "", "toy"],
    )
    assert scene.summary == "a cat"
    assert scene.objects == ["cat", "toy"]


def test_scene_state_rejects_bad_values() -> None:
    with pytest.raises(ValidationError):
        SceneState(t=-1, summary="x", action="x")
    with pytest.raises(ValidationError):
        SceneState(t=0, summary="x", action="x", surprise=2)
    with pytest.raises(ValidationError):
        SceneState(t=0, summary="word " * 36, action="x")
    with pytest.raises(ValidationError):
        SceneState(t=0, summary="x", action="x", source="camera")
    with pytest.raises(ValidationError):
        SceneState(t=0, summary="x", action="x", extra="nope")
