from __future__ import annotations

from fastapi.testclient import TestClient


def _start(client: TestClient) -> dict:
    response = client.post(
        "/api/session",
        json={"media_id": "sample.mp4", "title": "test stream", "category": "testing"},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["frontend_built"] is False


def test_media_library_and_session(client: TestClient) -> None:
    listing = client.get("/api/media").json()
    assert listing["items"][0]["id"] == "sample.mp4"

    body = _start(client)
    assert body["session"]["media"]["id"] == "sample.mp4"
    assert body["clock"]["t_play"] == 0.0
    assert body["clock"]["t_read"] == 5.5
    media_url = body["session"]["media_url"]

    full = client.get(media_url)
    assert full.status_code == 200
    assert full.headers["accept-ranges"] == "bytes"
    assert len(full.content) == 1000

    partial = client.get(media_url, headers={"Range": "bytes=0-9"})
    assert partial.status_code == 206
    assert partial.content == b"0123456789"
    assert partial.headers["content-range"] == "bytes 0-9/1000"

    suffix = client.get(media_url, headers={"Range": "bytes=-10"})
    assert suffix.status_code == 206
    assert suffix.content == b"0123456789"

    unsatisfiable = client.get(media_url, headers={"Range": "bytes=2000-"})
    assert unsatisfiable.status_code == 416
    assert unsatisfiable.headers["content-range"] == "bytes */1000"

    head = client.head(media_url)
    assert head.status_code == 200
    assert head.headers["content-length"] == "1000"

    assert client.get("/api/session").status_code == 200
    assert client.get("/api/clock").json()["session_id"] == body["session"]["id"]

    assert client.delete("/api/session").json()["ended"] == body["session"]["id"]
    assert client.get("/api/session").status_code == 404


def test_media_token_required(client: TestClient) -> None:
    assert client.get("/api/media/not-a-token").status_code == 404


def test_session_rejects_traversal(client: TestClient) -> None:
    response = client.post("/api/session", json={"media_id": "../outside.mp4"})
    assert response.status_code == 404


def test_websocket_clock_and_seek(client: TestClient) -> None:
    body = _start(client)
    session_id = body["session"]["id"]
    with client.websocket_connect("/ws") as websocket:
        hello = websocket.receive_json()
        assert hello["type"] == "hello"
        assert hello["session"]["id"] == session_id

        websocket.send_json(
            {"type": "time", "t_play": 5.0, "playing": True, "rate": 1.0, "duration": 100.0}
        )
        update = _receive_event(websocket, "clock.update")
        assert update["payload"]["t_play"] == 5.0
        assert update["payload"]["t_read"] == 10.5
        assert update["payload"]["generation"] == 0

        websocket.send_json(
            {
                "type": "time",
                "t_play": 50.0,
                "playing": True,
                "rate": 1.0,
                "duration": 100.0,
                "seek": True,
            }
        )
        seek = _receive_event(websocket, "clock.seek")
        assert seek["payload"]["generation"] == 1
        assert seek["payload"]["t_play"] == 50.0

        websocket.send_json({"type": "ping", "t": 123})
        pong = websocket.receive_json()
        while pong["type"] != "pong":
            pong = websocket.receive_json()
        assert pong["t"] == 123


def _receive_event(websocket, event_type: str) -> dict:
    for _ in range(20):
        message = websocket.receive_json()
        if message.get("type") == "event" and message.get("event_type") == event_type:
            return message
    raise AssertionError(f"did not receive {event_type}")
