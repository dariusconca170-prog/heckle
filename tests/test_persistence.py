from __future__ import annotations

import time
from collections.abc import Callable

from fastapi.testclient import TestClient

from backend import events


def _wait_for(check: Callable[[], bool], *, timeout_s: float = 3.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if check():
            return
        time.sleep(0.02)
    raise AssertionError("condition was not met before the timeout")


def _publish_chat(
    state: object,
    *,
    handle: str,
    text: str,
    t: float,
    archetype: str = "normie",
    tier: str = "core",
    kind: str = "chat",
    release_offset: float = 0.2,
    batch_id: str = "batch1",
) -> None:
    state.bus.publish_nowait(  # type: ignore[attr-defined]
        events.CHAT_MESSAGE,
        {
            "id": f"{handle}-{t}-{text[:4]}",
            "handle": handle,
            "archetype": archetype,
            "text": text,
            "source": "model",
            "batch_id": batch_id,
            "t": t,
            "created_at": time.time(),
            "word_count": len(text.split()),
            "release_t": t + release_offset,
            "trigger_t": t,
            "late_s": 0.1,
            "tier": tier,
            "kind": kind,
        },
    )


def test_two_streams_carry_memory(client: TestClient) -> None:
    state = client.app.state.app_state
    first = client.post(
        "/api/session",
        json={"media_id": "sample.mp4", "title": "cave run"},
    )
    assert first.status_code == 201, first.text
    first_stream = first.json()["session"]["id"]

    state.bus.publish_nowait(
        events.SCENE_EVENT,
        {
            "event": {
                "t": 12.0,
                "kind": "fail",
                "intensity": 0.9,
                "detail": "player falls in the cave",
            }
        },
    )
    for index in range(3):
        _publish_chat(
            state,
            handle="lore_dump",
            text="just like last stream",
            t=12.0,
            archetype="lore-keeper",
            release_offset=0.2 + index * 0.1,
        )
    # A preview is not a released message; it must not enter history.
    _publish_chat(state, handle="lore_dump", text="preview line", t=13.0, kind="preview")

    history_url = f"/api/history/streams/{first_stream}/messages"

    def _first_stream_ready() -> bool:
        body = client.get(history_url).json()
        return body["total"] == 3 and state.persistence.metrics()["events_logged"] >= 1

    _wait_for(_first_stream_ready)

    ended = client.delete("/api/session")
    assert ended.status_code == 200
    assert ended.json()["ended"] == first_stream

    second = client.post(
        "/api/session",
        json={"media_id": "sample.mp4", "title": "cave run 2"},
    )
    assert second.status_code == 201, second.text

    streams = client.get("/api/history/streams").json()["streams"]
    assert len(streams) == 2
    assert {stream["id"] for stream in streams} == {first_stream, second.json()["session"]["id"]}
    assert any(stream["message_count"] == 3 for stream in streams)

    memory = client.get("/api/history/memory", params={"handle": "lore_dump"}).json()
    by_kind = {row["kind"]: row for row in memory["memory"]}
    assert "lore" in by_kind
    assert "cave" in by_kind["lore"]["value"]
    assert "catchphrase" in by_kind
    # The second stream's prompt block is loaded from the first stream.
    assert memory["loaded_prompt"]
    assert any("last stream" in piece for piece in memory["loaded_prompt"])
    assert state.memory_for("lore_dump")

    # The chatter view of the roster is all a prompt needs.
    chatter = client.app.state.chatter
    persona = chatter.roster.by_handle("lore_dump")
    assert persona is not None
    described = chatter._memory_map([persona])
    assert described["lore_dump"]


def test_checkpoint_writes_memory_mid_stream(client: TestClient) -> None:
    state = client.app.state.app_state
    started = client.post("/api/session", json={"media_id": "sample.mp4"})
    assert started.status_code == 201
    stream_id = started.json()["session"]["id"]

    for index in range(2):
        _publish_chat(
            state,
            handle="xX_deadlift_Xx",
            text="W",
            t=5.0,
            archetype="hype",
            release_offset=0.2 + index * 0.1,
        )
    _wait_for(lambda: client.get(f"/api/history/streams/{stream_id}/messages").json()["total"] == 2)

    state.persistence.checkpoint(300.0)
    rows = state.persistence.raw_memory("xX_deadlift_Xx")
    assert any(row["kind"] == "catchphrase" for row in rows)


def test_history_is_empty_when_stream_is_unknown(client: TestClient) -> None:
    assert client.get("/api/history/streams/nope").status_code == 404
    assert client.get("/api/history/streams/nope/messages").status_code == 404
    assert client.get("/api/history/streams").json()["streams"] == []
