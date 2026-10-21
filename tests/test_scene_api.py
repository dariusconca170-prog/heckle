from __future__ import annotations

from fastapi.testclient import TestClient

from backend.perception.scene import SceneState


def test_scene_endpoint_empty_and_after_observation(client: TestClient) -> None:
    empty = client.get("/api/scene")
    assert empty.status_code == 200
    body = empty.json()
    assert body["active"] is False
    assert body["current"] is None
    assert body["sample_count"] == 0

    started = client.post("/api/session", json={"media_id": "sample.mp4"})
    assert started.status_code == 201

    state = client.app.state.app_state
    state.timeline.add(
        SceneState(
            t=7.5,
            summary="player crouches behind a rusty car in the rain",
            action="hiding",
            objects=["car", "rain"],
            mood="tense",
            surprise=0.7,
            scene_change=True,
            confidence=0.9,
        ),
        generation=0,
        latency_s=1.2,
    )
    scene = client.get("/api/scene").json()
    assert scene["active"] is True
    assert scene["current"]["t"] == 7.5
    assert scene["current"]["summary"].startswith("player crouches")
    assert scene["sample_count"] == 1
    assert scene["service"]["status"] in {"idle", "waiting", "unavailable"}
