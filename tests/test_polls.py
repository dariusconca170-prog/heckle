from __future__ import annotations

import asyncio
import random

import pytest
from fastapi.testclient import TestClient

from backend import events
from backend.config import Settings
from backend.dynamics.roster import CrowdRoster
from backend.features.polls import PollError, PollService
from backend.state import AppState


def _poll_service(settings: Settings) -> tuple[AppState, PollService, object]:
    state = AppState.build(settings)
    record = state.start_session(media_id="sample.mp4", viewer_count=1200)
    record.clock.set_duration(300.0)
    record.clock.update(10.0, playing=True)
    service = PollService(state, personas=[], rng=random.Random(9))
    service.set_personas(list(CrowdRoster(count=40, seed=4).personas))
    return state, service, record


def test_poll_closes_on_playback_timer_and_announces(settings: Settings) -> None:
    state, service, _record = _poll_service(settings)
    subscription = state.bus.subscribe([events.POLL_OPENED, events.POLL_VOTE, events.POLL_RESULTS])
    poll = service.open("what next", ["push", "hold", "rotate"], duration_s=60.0)
    assert poll.status == "open"
    assert poll.opened_at == pytest.approx(10.0, abs=0.05)
    assert poll.closes_at == pytest.approx(70.0, abs=0.05)
    assert len(state.chat_cues) == 1

    for step in range(1, 63):
        service.tick(10.0 + step)
    assert poll.total > 0
    assert sum(poll.counts) == poll.total
    assert poll.status == "closed"
    payloads = []
    while (event := subscription.get_nowait()) is not None:
        payloads.append(event.type)
    assert events.POLL_OPENED in payloads
    assert events.POLL_VOTE in payloads
    assert events.POLL_RESULTS in payloads
    closed = [cue for cue in state.chat_cues if cue.get("kind") == "poll_close"]
    assert closed and closed[-1]["total_votes"] == poll.total


def test_paused_video_does_not_advance_poll(settings: Settings) -> None:
    state, service, record = _poll_service(settings)
    poll = service.open("pause test", ["a", "b"], duration_s=10.0)
    assert state.persistence.repo is not None
    before = state.persistence.repo.poll_tally(poll.id)
    # The service loop only calls tick while playing; no tick means no votes
    # and no close even if wall time passes.
    assert poll.status == "open"
    assert state.persistence.repo.poll_tally(poll.id) == before
    record.clock.update(10.0, playing=False)
    assert service.current() is poll


def test_votes_are_persisted_and_deduplicated(settings: Settings) -> None:
    state, service, _record = _poll_service(settings)
    poll = service.open("vote test", ["one", "two"], duration_s=30.0)
    service.add_vote("you", 1)
    with pytest.raises(PollError):
        service.add_vote("you", 0)
    assert poll.counts[1] == 1
    assert state.persistence.repo is not None
    rows = state.persistence.repo.polls(poll.stream_id)
    assert rows[0]["counts"] == [0, 1]
    assert rows[0]["total_votes"] == 1
    service.tick(11.0)
    service.close(reason="test")
    rows = state.persistence.repo.polls(poll.stream_id)
    assert rows[0]["status"] == "closed"


def test_poll_http_lifecycle(client: TestClient) -> None:
    start = client.post("/api/session", json={"media_id": "sample.mp4"})
    assert start.status_code == 201
    created = client.post(
        "/api/polls",
        json={
            "question": "what now",
            "options": ["push", "hold", "rotate", "farm"],
            "duration_s": 60,
        },
    )
    assert created.status_code == 201, created.text
    poll = created.json()["poll"]
    assert poll["status"] == "open"
    assert len(poll["options"]) == 4

    listed = client.get("/api/polls")
    assert listed.status_code == 200
    assert listed.json()["current"]["id"] == poll["id"]

    voted = client.post(
        f"/api/polls/{poll['id']}/vote", json={"handle": "you", "option_index": 2}
    )
    assert voted.status_code == 200, voted.text
    assert voted.json()["poll"]["counts"][2] == 1

    closed = client.post("/api/polls/close")
    assert closed.status_code == 200
    assert closed.json()["poll"]["status"] == "closed"
    assert client.post("/api/polls/close").status_code == 409


async def test_poll_loop_closes_without_clock_errors(settings: Settings) -> None:
    state, service, record = _poll_service(settings)
    poll = service.open("loop test", ["a", "b"], duration_s=5.0)
    await service.start()
    await asyncio.sleep(0.3)
    assert "initialized" not in service.metrics()["last_error"]
    record.clock.update(16.0, playing=True)
    for _ in range(20):
        await asyncio.sleep(0.05)
        if poll.status == "closed":
            break
    await service.stop()
    assert poll.status == "closed"
    assert service.metrics()["last_error"] == ""


def test_poll_requires_stream_and_two_options(client: TestClient) -> None:
    no_stream = client.post(
        "/api/polls", json={"question": "q", "options": ["a", "b"], "duration_s": 30}
    )
    assert no_stream.status_code == 409
    client.post("/api/session", json={"media_id": "sample.mp4"})
    bad = client.post(
        "/api/polls", json={"question": "q", "options": ["only"], "duration_s": 30}
    )
    assert bad.status_code == 422
