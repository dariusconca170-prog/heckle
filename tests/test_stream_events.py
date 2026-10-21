from __future__ import annotations

import random

from fastapi.testclient import TestClient

from backend import events
from backend.config import Settings
from backend.features.stream_events import StreamEventService
from backend.state import AppState


def _service(settings: Settings) -> tuple[AppState, StreamEventService, object]:
    state = AppState.build(settings)
    record = state.start_session(media_id="sample.mp4", viewer_count=1000)
    record.clock.set_duration(2400.0)
    record.clock.update(5.0, playing=True)
    service = StreamEventService(state, rng=random.Random(12), names=["raider_one"])
    service._begin_session(record)
    return state, service, record


def test_raid_fires_system_message_and_burst_cue(settings: Settings) -> None:
    state, service, _record = _service(settings)
    subscription = state.bus.subscribe([events.CHAT_MESSAGE, events.STREAM_EVENT], maxsize=64)
    moment = service.trigger("raid", t_play=120.0)
    assert moment is not None
    assert moment.kind == "raid"
    assert moment.amount > 0
    payloads = []
    while (event := subscription.get_nowait()) is not None:
        payloads.append(event)
    chat = [event for event in payloads if event.type == events.CHAT_MESSAGE]
    stream = [event for event in payloads if event.type == events.STREAM_EVENT]
    assert chat and chat[0].payload["kind"] == "system"
    assert chat[0].payload["handle"] == "RaidBot"
    assert "raided" in chat[0].payload["text"]
    assert stream and stream[0].payload["event"]["kind"] == "raid"
    assert stream[0].payload["viewer_count"] > 1000
    assert any(cue.get("kind") == "stream_event" for cue in state.chat_cues)
    assert service.metrics()["raid_boost"] > 1.0


def test_scheduled_events_advance_and_viewer_curve_is_bounded(settings: Settings) -> None:
    _state, service, _record = _service(settings)
    service._next_sub = 10.0
    service._next_raid = 10_000.0
    service._next_follow = 10_000.0
    assert service.tick(9.0) == 0
    assert service.tick(10.0) == 1
    assert service.tick(11.0) == 0
    assert service._next_sub > 10.0

    early = service.viewer_curve(0.0)
    late = service.viewer_curve(120.0)
    assert 0.4 * 1000 <= early <= 0.5 * 1000
    assert 0.96 * 1000 <= late <= 1.04 * 1000
    assert _state.live_viewers == service.metrics()["live_viewers"]


def test_stream_event_http_trigger(client: TestClient) -> None:
    start = client.post("/api/session", json={"media_id": "sample.mp4"})
    assert start.status_code == 201
    response = client.post("/api/stream-events/gift")
    assert response.status_code == 201, response.text
    body = response.json()["event"]
    assert body["kind"] == "gift"
    assert "gifted" in body["text"]
    assert client.post("/api/stream-events/nope").status_code == 422
