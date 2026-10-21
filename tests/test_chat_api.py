from __future__ import annotations

from fastapi.testclient import TestClient

from backend.chatter.guard import MAX_WORDS, classify


def test_chat_generate_preview_falls_back_without_model(client: TestClient) -> None:
    response = client.post("/api/chat/generate", json={"count": 10})
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["messages"]) == 10
    assert body["fallback"] is True
    for message in body["messages"]:
        assert message["word_count"] <= MAX_WORDS
        assert classify(message["text"]) == "ok"
        assert message["source"] == "template"
        assert message["handle"]


def test_chat_recent_and_personas(client: TestClient) -> None:
    start = client.post("/api/session", json={"media_id": "sample.mp4"})
    assert start.status_code == 201

    generated = client.post("/api/chat/generate", json={"count": 3})
    assert generated.status_code == 200

    recent = client.get("/api/chat/recent").json()["messages"]
    assert len(recent) == 3
    assert [message["batch_id"] for message in recent] == [generated.json()["batch_id"]] * 3

    personas = client.get("/api/personas").json()["personas"]
    assert len(personas) >= 8
    assert len({persona["handle"] for persona in personas}) == len(personas)
