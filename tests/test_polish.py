from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient


def test_overlay_is_transparent_and_chat_only(client: TestClient) -> None:
    for path in ("/overlay/chat", "/overlay"):
        response = client.get(path)
        assert response.status_code == 200, path
        body = response.text
        assert "background: transparent" in body
        assert "scrollbar-width: none" in body
        assert "/ws" in body
        assert "chat.message" in body
        assert "w.svg" in body
        assert "OBS" in body


def test_emote_assets_ship_with_the_frontend() -> None:
    root = Path(__file__).resolve().parents[1] / "frontend" / "public" / "emotes"
    for name in (
        "w.svg",
        "lul.svg",
        "kekw.svg",
        "pog.svg",
        "monkas.svg",
        "copium.svg",
        "ratio.svg",
        "heart.svg",
    ):
        path = root / name
        assert path.is_file(), name
        assert "<svg" in path.read_text(encoding="utf-8")


def test_settings_patch_and_preset_persist(client: TestClient) -> None:
    settings = client.app.state.app_state.settings
    initial = client.get("/api/settings")
    assert initial.status_code == 200
    payload = initial.json()["settings"]
    assert payload["content_modes"] == ["unhinged", "edgy", "neutral"]
    assert {preset["name"] for preset in payload["presets"]} == {"calm", "balanced", "chaos"}

    updated = client.patch(
        "/api/settings",
        json={
            "intensity": 1.4,
            "content_mode": "unhinged",
            "chat_target_buffer": 24,
            "lookahead_s": 6.0,
        },
    )
    assert updated.status_code == 200, updated.text
    body = updated.json()["settings"]
    assert body["intensity"] == 1.4
    assert body["content_mode"] == "unhinged"
    assert body["chat_target_buffer"] == 24
    assert settings.chat_target_buffer == 24
    assert client.app.state.app_state.lookahead_s == 6.0
    stored = (settings.data_dir / "settings.json").read_text(encoding="utf-8")
    assert '"unhinged"' in stored

    preset = client.post("/api/settings/preset/chaos")
    assert preset.status_code == 200, preset.text
    assert preset.json()["settings"]["intensity"] == 1.8
    assert preset.json()["settings"]["content_mode"] == "unhinged"

    invalid = client.patch("/api/settings", json={"content_mode": "maximum"})
    assert invalid.status_code == 422
    assert client.post("/api/settings/preset/nope").status_code == 422


def test_system_metrics_survive_without_a_gpu(client: TestClient) -> None:
    response = client.get("/api/system")
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body["gpu"]["available"], bool)
    assert "used_mb" in body["gpu"]
    assert body["chat"]["enabled"] in (True, False)
    assert "last_tok_s" in body["chat"]
    assert "ram" in body
    assert body["content_mode"] in {"unhinged", "edgy", "neutral"}


def _record(state, *, handle: str, text: str, t: float, reply_to: str | None = None) -> None:
    state.persistence.record_message(
        {
            "id": f"{handle}-{t}",
            "handle": handle,
            "archetype": "normie",
            "text": text,
            "kind": "chat",
            "tier": "core",
            "t": t,
            "release_t": t,
            "created_at": time.time(),
            "reply_to": reply_to,
            "mentions": ["@streamer"],
            "emotes": ["LUL"],
        }
    )


def test_chat_export_json_and_csv(client: TestClient) -> None:
    start = client.post("/api/session", json={"media_id": "sample.mp4"})
    assert start.status_code == 201
    stream_id = start.json()["session"]["id"]
    state = client.app.state.app_state
    _record(state, handle="cozybean", text="why would you push there", t=3.0)
    _record(state, handle="not_a_bot_92", text="cooldown was up", t=4.0, reply_to="cozybean-3.0")

    csv_response = client.get("/api/export/chat?format=csv")
    assert csv_response.status_code == 200
    assert csv_response.headers["content-type"].startswith("text/csv")
    assert "attachment" in csv_response.headers["content-disposition"]
    csv_text = csv_response.text
    assert csv_text.splitlines()[0].startswith("id,t_playback,handle")
    assert "cozybean" in csv_text
    assert "cooldown was up" in csv_text

    json_response = client.get(f"/api/history/streams/{stream_id}/export?format=json")
    assert json_response.status_code == 200
    rows = json_response.json()["messages"]
    assert len(rows) == 2
    assert rows[1]["reply_handle"] == "cozybean"
    assert rows[0]["emotes"] == ["LUL"]

    assert client.get("/api/export/chat?format=xml").status_code == 422
    assert client.get("/api/history/streams/nope/export").status_code == 404


def test_health_reports_new_features(client: TestClient) -> None:
    body = client.get("/health").json()
    assert "clips" in body
    assert "polls" in body
    assert "mic" in body
    assert "screen" in body
    assert "last_tok_s" in body["chat"]
