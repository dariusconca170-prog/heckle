from __future__ import annotations

import math
import random
import struct

from fastapi.testclient import TestClient

from backend import events
from backend.chatter.service import ChatterService
from backend.config import Settings
from backend.dynamics.service import DynamicsService
from backend.features.mic import MicService, VadSegmenter, pcm_energy
from backend.perception.scene import SceneState
from backend.perception.service import PerceptionService
from backend.state import AppState


def _tone(duration_s: float, *, amplitude: int = 9000, rate: int = 16_000) -> bytes:
    count = int(duration_s * rate)
    samples = [
        int(amplitude * math.sin(2 * math.pi * 220 * index / rate)) for index in range(count)
    ]
    return struct.pack(f"<{len(samples)}h", *samples)


def _silence(duration_s: float, *, rate: int = 16_000) -> bytes:
    return b"\x00\x00" * int(duration_s * rate)


def test_vad_segments_speech_then_silence() -> None:
    segmenter = VadSegmenter(threshold=0.01)
    segments = []
    for block in (_silence(0.3), _tone(1.2), _silence(1.0)):
        segments.extend(segmenter.push(block))
    assert len(segments) == 1
    segment = segments[0]
    assert segment.duration_s >= 0.9
    assert segment.energy > 0.01
    assert len(segment.pcm) > 16_000


def test_vad_ignores_silence() -> None:
    segmenter = VadSegmenter(threshold=0.01)
    assert segmenter.push(_silence(2.0)) == []
    assert segmenter.flush() is None
    assert pcm_energy(b"") == 0.0


def test_inject_transcript_reaches_bus_and_cue(settings: Settings) -> None:
    state = AppState.build(settings)
    state.start_session(media_id="sample.mp4")
    mic = MicService(state)
    subscription = state.bus.subscribe([events.MIC_SPEECH])
    text = mic.inject_transcript("I just beat the boss", energy=0.7)
    assert text == "I just beat the boss"
    assert state.latest_speech is not None
    assert state.latest_speech["text"] == "I just beat the boss"
    event = subscription.get_nowait()
    assert event is not None and event.type == events.MIC_SPEECH
    assert event.payload["energy"] == 0.7
    assert any(cue.get("kind") == "speech" for cue in state.chat_cues)
    assert mic.metrics()["transcripts"] == 1
    assert mic.metrics()["last_text"] == "I just beat the boss"
    state.close()


async def test_heard_speech_gets_at_least_two_chat_reactions(settings: Settings) -> None:
    state = AppState.build(settings)
    record = state.start_session(media_id="sample.mp4")
    record.clock.set_duration(60.0)
    record.clock.update(12.0, playing=True)
    mic = MicService(state)
    chatter = ChatterService(state, rng=random.Random(3))
    dynamics = DynamicsService(state, chatter, rng=random.Random(4))
    dynamics._begin_session(record)
    mic.inject_transcript("I just beat the boss")
    dynamics._drain_cues(12.0)
    reactions = [item for item in dynamics._buffer if item.trigger_t == 12.0]
    assert len(reactions) >= 2
    assert all(item.release_t <= 12.0 + 5.5 for item in reactions)
    joined = " ".join(item.message.text for item in reactions).lower()
    assert "boss" in joined or "hear that" in joined
    state.close()


def test_perception_merges_mic_speech_into_scene(settings: Settings) -> None:
    state = AppState.build(settings)
    service = PerceptionService(state)
    state.latest_speech = {"text": "he said what", "energy": 0.66, "t": 8.0, "source": "mic"}
    scene = SceneState(t=10.0, summary="a player hides", action="hiding")
    merged = service._with_speech(scene)
    assert merged.speech_text == "he said what"
    assert merged.speech_energy == 0.66
    assert merged.source == "mic"
    # A transcript from the future is not applied to an older frame.
    state.latest_speech = {"text": "future", "energy": 0.5, "t": 20.0}
    assert service._with_speech(scene).speech_text == ""
    state.close()


def test_mic_http_transcript_and_status(client: TestClient) -> None:
    status = client.get("/api/mic")
    assert status.status_code == 200
    assert status.json()["enabled"] is False
    injected = client.post("/api/mic/transcript", json={"text": "chat am i wrong"})
    assert injected.status_code == 201, injected.text
    assert injected.json()["text"] == "chat am i wrong"
    state = client.app.state.app_state
    assert state.latest_speech is not None
    assert state.latest_speech["text"] == "chat am i wrong"
    assert client.post("/api/mic/stop").status_code == 200
