from __future__ import annotations

from backend.perception.scene import SceneState
from backend.perception.timeline import SceneTimeline


def _scene(t: float, **overrides) -> SceneState:
    data = {
        "t": t,
        "summary": "player crouches behind a rusty car in the rain",
        "action": "hiding behind car",
        "objects": ["car", "rain"],
        "mood": "tense",
        "surprise": 0.4,
        "scene_change": False,
        "on_screen_text": "",
        "confidence": 0.8,
    }
    data.update(overrides)
    return SceneState(**data)


def test_timeline_keeps_window_and_reports_latency() -> None:
    timeline = SceneTimeline(window_s=90.0)
    timeline.add(_scene(10.0), latency_s=1.2, received_at=1000.0)
    timeline.add(_scene(20.0), latency_s=1.4, received_at=1001.0)
    snapshot = timeline.snapshot()
    assert snapshot["sample_count"] == 2
    assert snapshot["current"]["t"] == 20.0
    assert snapshot["latency_s"]["last"] == 1.4
    assert snapshot["latency_s"]["p50"] == 1.2
    assert snapshot["latency_s"]["p95"] == 1.4
    assert len(snapshot["timeline"]) == 2

    timeline.add(_scene(120.0), latency_s=1.0)
    snapshot = timeline.snapshot()
    assert snapshot["sample_count"] == 1
    assert snapshot["current"]["t"] == 120.0


def test_timeline_reset_and_backward_jump() -> None:
    timeline = SceneTimeline()
    timeline.add(_scene(20.0))
    timeline.add(_scene(10.0))
    assert timeline.latest is not None
    assert timeline.latest.t == 10.0
    assert timeline.snapshot()["sample_count"] == 1
    timeline.reset()
    assert timeline.snapshot()["current"] is None


def test_fail_clutch_and_jumpscare_events() -> None:
    timeline = SceneTimeline()
    timeline.add(_scene(10.0, action="player runs", summary="player runs down a corridor"))
    fail = timeline.add(
        _scene(
            13.0,
            action="player dies",
            summary="player dies to the boss and the screen goes dark",
            surprise=0.9,
            scene_change=True,
            confidence=0.92,
        )
    )
    kinds = {event.kind for event in fail}
    assert "fail" in kinds
    assert "jumpscare" in kinds

    clutch = timeline.add(
        _scene(
            20.0,
            action="player wins",
            summary="player clutches the final round",
            confidence=0.9,
        )
    )
    assert any(event.kind == "clutch" for event in clutch)


def test_topic_shift_is_throttled() -> None:
    timeline = SceneTimeline()
    timeline.add(_scene(10.0, summary="man chops onions on a wooden board", action="chopping onions"))
    first = timeline.add(
        _scene(14.0, summary="spaceship explodes above a ringed planet", action="ship explodes")
    )
    assert any(event.kind == "topic_shift" for event in first)
    second = timeline.add(
        _scene(16.0, summary="a cat sleeps on a warm radiator", action="cat sleeps")
    )
    assert not any(event.kind == "topic_shift" for event in second)
