from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from backend.config import Settings
from backend.features.screen import ScreenService
from backend.perception.frames import FrameError
from backend.perception.observer import Observation
from backend.perception.scene import SceneState
from backend.perception.screen import ScreenFrameSource, is_blocked
from backend.state import AppState


def test_privacy_blocklist_is_case_insensitive() -> None:
    assert is_blocked("1Password - Vault", ("1password", "bank"))
    assert is_blocked("My Bank Account", ("1password", "bank"))
    assert not is_blocked("Visual Studio Code", ("1password", "bank"))
    assert not is_blocked("anything", ())
    assert not is_blocked("", ("bank",))


async def test_screen_source_refuses_blocked_window() -> None:
    source = ScreenFrameSource(
        capture=lambda: (b"\x00" * 12, 2, 2, "rgb24"),
        encoder=lambda raw, width, height, pix_fmt: b"jpeg-bytes",
        title_provider=lambda: "1Password - Vault",
        blocklist=("1password",),
    )
    with pytest.raises(FrameError) as error:
        await source.frame_at(1.0)
    assert "privacy" in str(error.value)
    assert source.last_blocked == "1Password - Vault"

    source._title_provider = lambda: "Notepad"
    frame = await source.frame_at(2.0)
    assert frame.data == b"jpeg-bytes"
    assert frame.mime == "image/jpeg"
    await source.close()


class _UnavailableSource:
    backend = "unavailable"
    setup_error = "no screen capture backend"
    last_title = ""
    last_blocked = ""

    def ensure_ready(self) -> str:
        return "unavailable"

    async def close(self) -> None:
        return None


class _FakeObserver:
    def __init__(self) -> None:
        self.calls = 0

    async def observe(self, frame, t: float) -> Observation:
        self.calls += 1
        return Observation(
            scene=SceneState(
                t=t,
                summary="a browser plays a video",
                action="watching a video",
                confidence=0.8,
            ),
            completion=None,  # type: ignore[arg-type]
            latency_s=0.01,
        )


async def test_screen_service_samples_and_pauses(settings: Settings) -> None:
    state = AppState.build(settings)
    record = state.start_session(media_id="sample.mp4", viewer_count=800)
    record.clock.set_duration(60.0)
    record.clock.update(5.0, playing=True)
    source = ScreenFrameSource(
        capture=lambda: (b"\x00" * 12, 2, 2, "rgb24"),
        encoder=lambda raw, width, height, pix_fmt: b"jpeg",
        title_provider=lambda: "Notepad",
    )
    observer = _FakeObserver()
    service = ScreenService(state, source=source, observer=observer)
    assert await service.sample_once(5.0) is True
    assert observer.calls == 1
    assert service.metrics()["samples"] == 1
    assert state.timeline.latest is not None
    assert state.timeline.latest.action == "watching a video"

    assert service.toggle() is True
    assert await service.sample_once(6.0) is False
    assert observer.calls == 1
    assert service.metrics()["paused"] is True
    assert service.toggle() is False
    await service.stop()
    state.close()


async def test_unavailable_screen_backend_skips_the_observer(settings: Settings) -> None:
    state = AppState.build(settings)
    record = state.start_session(media_id="sample.mp4")
    record.clock.set_duration(60.0)
    record.clock.update(5.0, playing=True)
    called = False

    def factory():
        nonlocal called
        called = True
        raise AssertionError("observer must not be built without a capture backend")

    service = ScreenService(state, source=_UnavailableSource(), observer_factory=factory)  # type: ignore[arg-type]
    await service.start()
    for _ in range(10):
        await asyncio.sleep(0.05)
        if service.metrics()["status"] == "unavailable":
            break
    metrics = service.metrics()
    await service.stop()
    assert metrics["status"] == "unavailable"
    assert "no screen capture backend" in metrics["last_error"]
    assert called is False
    state.close()


async def test_screen_loop_samples_a_playing_session(settings: Settings) -> None:
    state = AppState.build(settings)
    record = state.start_session(media_id="sample.mp4", viewer_count=800)
    record.clock.set_duration(60.0)
    record.clock.update(5.0, playing=True)
    source = ScreenFrameSource(
        capture=lambda: (b"\x00" * 12, 2, 2, "rgb24"),
        encoder=lambda raw, width, height, pix_fmt: b"jpeg",
        title_provider=lambda: "Notepad",
    )
    service = ScreenService(state, source=source, observer=_FakeObserver())
    await service.start()
    for _ in range(20):
        await asyncio.sleep(0.05)
        if service.metrics()["samples"] > 0:
            break
    metrics = service.metrics()
    await service.stop()
    assert metrics["samples"] > 0, metrics
    assert "initialized" not in metrics["last_error"]
    state.close()


def test_screen_api_reports_unavailable_or_live(client: TestClient) -> None:
    start = client.post("/api/session", json={"media_id": "sample.mp4"})
    assert start.status_code == 201
    started = client.post("/api/screen/start")
    assert started.status_code in (200, 503), started.text
    status = client.get("/api/screen")
    assert status.status_code == 200
    assert "blocklist" in status.json()
    toggled = client.post("/api/screen/toggle")
    assert toggled.status_code == 200
    assert client.post("/api/screen/stop").status_code == 200


def test_screen_start_requires_a_stream(client: TestClient) -> None:
    assert client.post("/api/screen/start").status_code == 409
