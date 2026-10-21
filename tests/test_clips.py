from __future__ import annotations

import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from backend import events
from backend.config import Settings
from backend.features.clips import (
    ClipCandidate,
    ClipService,
    MessageSample,
    build_reason,
    export_clip,
)
from backend.state import AppState


def _service(settings: Settings) -> tuple[AppState, ClipService, object]:
    state = AppState.build(settings)
    record = state.start_session(media_id="sample.mp4", viewer_count=900)
    record.clock.set_duration(180.0)
    record.clock.update(100.0, playing=True)
    service = ClipService(state)
    service._begin_session(record)
    return state, service, record


def test_rate_spike_creates_a_candidate_with_quoted_messages(settings: Settings) -> None:
    state, service, _record = _service(settings)
    for index in range(400):
        t = 0.5 + index * 0.75
        service.seed(
            [MessageSample(t=t, handle=f"viewer{index % 7}", text="lol", words=1)]
        )
    for index in range(70):
        service.seed(
            [
                MessageSample(
                    t=95.0 + index * 0.1,
                    handle="hype_guy",
                    text="LETS GO CLIP IT",
                    words=4,
                    emotes=("Pog",),
                )
            ]
        )
    stats = service.rate_stats(100.0)
    assert stats.rate > 3.0
    assert stats.z > 3.0
    candidate = service.evaluate(100.0, intensity=0.8)
    assert candidate is not None
    assert candidate.start_t == pytest.approx(85.0)
    assert candidate.end_t == pytest.approx(105.0)
    assert candidate.reason
    assert "hype_guy" in candidate.reason
    assert "LETS GO CLIP IT" in candidate.reason
    assert candidate.score > 0


def test_tick_persists_candidate_and_cues_chat(settings: Settings) -> None:
    state, service, record = _service(settings)
    for index in range(300):
        service.seed([MessageSample(t=index * 0.25, handle="a", text="lol", words=1)])
    for index in range(80):
        service.seed(
            [MessageSample(t=95.0 + index * 0.05, handle="b", text="W", words=1)]
        )
    subscription = state.bus.subscribe([events.CLIP_CANDIDATE])
    candidate = service.tick(100.0, intensity=0.9)
    assert candidate is not None
    assert candidate.id > 0
    assert state.persistence.repo is not None
    rows = state.persistence.repo.clips(record.id)
    assert rows and rows[0]["reason"] == candidate.reason
    event = subscription.get_nowait()
    assert event is not None and event.type == events.CLIP_CANDIDATE
    assert any(cue.get("kind") == "clip" for cue in state.chat_cues)


def test_no_candidate_without_a_spike(settings: Settings) -> None:
    _state, service, _record = _service(settings)
    for index in range(400):
        service.seed([MessageSample(t=index * 0.25, handle="a", text="lol", words=1)])
    assert service.evaluate(100.0) is None


def test_reason_prefers_engagement_and_dedupes() -> None:
    samples = [
        MessageSample(1.0, "quiet", "ok", 1),
        MessageSample(2.0, "loud", "LETS GOOO", 2, ("Pog", "Pog")),
        MessageSample(3.0, "loud", "LETS GOOO", 2, ("Pog", "Pog")),
        MessageSample(4.0, "mid", "clip it", 2),
    ]
    reason = build_reason(samples)
    assert reason.count("LETS GOOO") == 1
    assert reason.startswith('loud: "LETS GOOO"')
    assert "clip it" in reason


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="ffmpeg/ffprobe are required for clip export",
)
async def test_export_contains_the_requested_segment(tmp_path) -> None:
    source = tmp_path / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=15",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440",
            "-t",
            "12",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
    )
    candidate = ClipCandidate(
        stream_id="stream-test",
        start_t=3.0,
        end_t=8.0,
        score=12.0,
        peak_rate=4.0,
        z=4.5,
        reason="loud: \"clip it\"",
    )
    clips_dir = tmp_path / "clips"
    path, thumb = await export_clip(candidate, source, clips_dir)
    exported = clips_dir / path
    thumbnail = clips_dir / thumb
    assert exported.is_file() and exported.stat().st_size > 0
    assert thumbnail.is_file() and thumbnail.stat().st_size > 0
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(exported),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    duration = float(probe.stdout.strip())
    assert 4.5 <= duration <= 5.6


def test_clip_api_lists_and_serves_files(client: TestClient) -> None:
    start = client.post("/api/session", json={"media_id": "sample.mp4"})
    assert start.status_code == 201
    created = client.post("/api/clips", json={"start_t": 1.0, "end_t": 6.0})
    assert created.status_code == 201, created.text
    clip = created.json()["clip"]
    assert clip["duration_s"] == pytest.approx(5.0)
    listed = client.get("/api/clips")
    assert listed.status_code == 200
    assert any(item["id"] == clip["id"] for item in listed.json()["clips"])
    # The clip is waiting for its export window in a real stream, so the file
    # endpoint reports "still exporting" rather than a missing file.
    assert client.get(f"/api/clips/{clip['id']}/file").status_code == 409
    assert client.get("/api/clips/99999/file").status_code == 404
