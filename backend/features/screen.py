"""Screen-share mode (M9, PLAN.md §10.6).

Perception reads the live desktop instead of a file. There is deliberately no
look-ahead: a screen frame is the present, so reactions land 3-5 s after the
moment (PLAN.md §7.7). A pause hotkey and a window-title blocklist are mandatory
because this captures the user's own desktop; both stop frames before the model
sees them.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import TYPE_CHECKING, Any

from .. import events
from ..llm.client import LLMClient
from ..perception.frames import FrameError
from ..perception.observer import ObservationError, Observer
from ..perception.screen import ScreenFrameSource

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..state import AppState

logger = logging.getLogger(__name__)


class ScreenService:
    """Sample the desktop at the playback clock's current time."""

    def __init__(
        self,
        state: AppState,
        *,
        source: ScreenFrameSource | None = None,
        observer: Observer | None = None,
        source_factory: Any | None = None,
        observer_factory: Any | None = None,
    ) -> None:
        self.state = state
        self.settings = state.settings
        self._source = source
        self._observer = observer
        self._source_factory = source_factory
        self._observer_factory = observer_factory
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self._paused = False
        self._status = "idle"
        self._last_error = ""
        self._samples = 0
        self._errors = 0
        self._last_latency_s: float | None = None
        self._last_t: float | None = None
        self._next_due = 0.0
        self._hotkey_handle: Any = None
        self._session_id: str | None = None
        self._client: LLMClient | None = None

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> dict[str, Any]:
        if self._task is not None and not self._task.done():
            return self.metrics()
        if self._source is None and self._source_factory is None:
            self._source = ScreenFrameSource(
                blocklist=tuple(self.settings.screen_blocklist),
                ffmpeg=self.settings.ffmpeg,
                max_edge=self.settings.frame_max_edge,
            )
        self._stopping = False
        self._paused = False
        self._status = "starting"
        self._task = asyncio.create_task(self._run(), name="screen-capture")
        self._register_hotkey()
        await asyncio.sleep(0)
        return self.metrics()

    async def stop(self) -> dict[str, Any]:
        self._stopping = True
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._unregister_hotkey()
        source = self._source
        self._source = None
        if source is not None:
            with contextlib.suppress(Exception):
                await source.close()
        if self._client is not None:
            with contextlib.suppress(Exception):
                await self._client.close()
            self._client = None
        self._status = "stopped"
        return self.metrics()

    def toggle(self) -> bool:
        """Hotkey kill switch: paused means no frames reach the model."""
        self._paused = not self._paused
        self._status = "paused" if self._paused else "live"
        if self._paused:
            self._last_error = ""
        return self._paused

    def metrics(self) -> dict[str, Any]:
        source = self._source
        return {
            "status": self._status,
            "paused": self._paused,
            "session_id": self._session_id,
            "backend": getattr(source, "backend", ""),
            "hotkey": self.settings.screen_hotkey,
            "samples": self._samples,
            "errors": self._errors,
            "last_t": round(self._last_t, 3) if self._last_t is not None else None,
            "last_latency_s": (
                round(self._last_latency_s, 3) if self._last_latency_s is not None else None
            ),
            "last_title": getattr(source, "last_title", ""),
            "blocked_title": getattr(source, "last_blocked", ""),
            "blocklist": list(self.settings.screen_blocklist),
            "last_error": self._last_error,
        }

    # -- loop --------------------------------------------------------------

    async def _run(self) -> None:
        try:
            source = self._source or await self._make_source()
            self._source = source
            if hasattr(source, "ensure_ready"):
                source.ensure_ready()
            if getattr(source, "backend", "") == "unavailable":
                self._status = "unavailable"
                self._last_error = getattr(source, "setup_error", "") or (
                    "no screen capture backend"
                )
                return
            self._observer = self._observer or await self._make_observer()
            self._status = "live"
            while not self._stopping:
                session = self.state.session
                if session is None:
                    self._status = "idle"
                    self._session_id = None
                    await asyncio.sleep(0.25)
                    continue
                self._session_id = session.id
                snapshot = session.clock.snapshot()
                if self._paused:
                    self._status = "paused"
                    await asyncio.sleep(0.2)
                    continue
                if not session.clock.initialized or not snapshot.playing:
                    self._status = "waiting"
                    await asyncio.sleep(0.2)
                    continue
                now = time.monotonic()
                if now < self._next_due:
                    await asyncio.sleep(min(0.25, self._next_due - now))
                    continue
                await self.sample_once(float(snapshot.t_play))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - last-resort loop guard
            logger.exception("screen loop failed")
            self._last_error = str(exc)[:400]
            self._status = "error"

    async def sample_once(self, t_play: float) -> bool:
        source = self._source
        observer = self._observer
        if source is None or observer is None or self._paused:
            return False
        started = time.monotonic()
        try:
            frame = await source.frame_at(t_play)
            observation = await observer.observe(frame, t_play)
        except (FrameError, ObservationError) as exc:
            self._errors += 1
            self._last_error = str(exc)[:400]
            self._status = "blocked" if "privacy" in self._last_error else "error"
            self._next_due = time.monotonic() + max(
                1.0, float(self.settings.perception_error_backoff_s)
            )
            return False
        latency = time.monotonic() - started
        self._last_latency_s = latency
        self._last_t = float(observation.scene.t)
        self._samples += 1
        self._errors = 0
        self._last_error = ""
        self._status = "live"
        derived = self.state.timeline.add(
            observation.scene,
            generation=self.state.session.clock.generation if self.state.session else 0,
            latency_s=latency,
        )
        snapshot = self.state.timeline.snapshot(limit=30, event_limit=20)
        with contextlib.suppress(RuntimeError):
            self.state.bus.publish_nowait(
                events.SCENE_UPDATE,
                {
                    "scene": observation.scene.model_dump(),
                    "generation": self.state.session.clock.generation if self.state.session else 0,
                    "latency_s": round(latency, 4),
                    "snapshot": snapshot,
                    "service": {**self.metrics(), "source": "screen"},
                    "derived_events": [event.as_dict() for event in derived],
                    "source": "screen",
                },
            )
        # Screen has no look-ahead, so sample a little faster than file
        # playback to keep the perceived reaction near the 3-5 s target.
        self._next_due = time.monotonic() + max(
            1.0, min(float(self.settings.perception_interval_s), 2.0)
        )
        return True

    async def _make_source(self) -> ScreenFrameSource:
        if self._source_factory is None:
            raise RuntimeError("no screen source configured")
        source = self._source_factory()
        if asyncio.iscoroutine(source):
            source = await source
        return source

    async def _make_observer(self) -> Observer:
        if self._observer_factory is not None:
            observer = self._observer_factory()
            if asyncio.iscoroutine(observer):
                observer = await observer
            return observer
        self._client = LLMClient(
            base_url=f"http://{self.settings.llama_host}:{self.settings.llama_port}",
            model="twitch-sim",
            timeout_s=self.settings.llm_timeout_s,
        )
        return Observer(self._client)

    # -- hotkey ------------------------------------------------------------

    def _register_hotkey(self) -> None:
        try:
            import keyboard  # type: ignore[import-not-found]
        except Exception as exc:
            self._last_error = f"hotkey unavailable (keyboard not installed): {exc}"[:400]
            return
        try:
            self._hotkey_handle = keyboard.add_hotkey(
                self.settings.screen_hotkey, self.toggle
            )
        except Exception as exc:
            self._last_error = f"hotkey unavailable: {exc}"[:400]

    def _unregister_hotkey(self) -> None:
        handle = self._hotkey_handle
        self._hotkey_handle = None
        if handle is None:
            return
        with contextlib.suppress(Exception):
            import keyboard  # type: ignore[import-not-found]

            keyboard.remove_hotkey(handle)


__all__ = ["ScreenService"]
