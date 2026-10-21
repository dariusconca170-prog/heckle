"""Perception loop: sample at t_read, observe, append to the timeline.

One request is in flight at a time. The loop uses the playback clock's
``t_read`` (``t_play + look-ahead``) so the scene it observes is the future the
viewer is about to see. It also enforces a model-time duty cap: chat will always
outrank perception once M2 lands, and that rule starts here.

Failures are log lines and a debug-panel status, never exceptions on the API.
If llama-server is not up yet, the loop backs off and retries; the player keeps
working.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .. import events
from ..llm.client import LLMClient, LLMError
from .frames import FileFrameSource, FrameError, FrameSource
from .observer import ObservationError, Observer
from .timeline import SceneTimeline

if TYPE_CHECKING:  # pragma: no cover - import cycle only for type checking
    from ..state import AppState, SessionRecord

logger = logging.getLogger(__name__)


class PerceptionService:
    def __init__(
        self,
        state: AppState,
        *,
        source_factory: Callable[[Any], FrameSource] | None = None,
        observer_factory: Callable[[], Observer] | None = None,
        timeline: SceneTimeline | None = None,
    ) -> None:
        self.state = state
        self.settings = state.settings
        self.timeline = timeline or state.timeline
        self._source_factory = source_factory
        self._observer_factory = observer_factory
        self._source: FrameSource | None = None
        self._observer: Observer | None = None
        self._client: LLMClient | None = None
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self._session_id: str | None = None
        self._generation: int | None = None
        self._last_t = -1.0
        self._next_due = 0.0
        self._boost_until = 0.0
        self._status = "idle"
        self._last_error = ""
        self._samples = 0
        self._errors = 0
        self._last_latency_s: float | None = None
        self._deferred = 0

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stopping = False
        self._task = asyncio.create_task(self._run(), name="perception-loop")

    async def stop(self) -> None:
        self._stopping = True
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await self._close_source()
        if self._client is not None:
            await self._client.close()
            self._client = None
        self._observer = None
        self._set_status("stopped")

    def metrics(self) -> dict[str, Any]:
        return {
            "status": self._status,
            "session_id": self._session_id,
            "samples": self._samples,
            "errors": self._errors,
            "last_error": self._last_error,
            "last_latency_s": (
                round(self._last_latency_s, 4) if self._last_latency_s is not None else None
            ),
            "last_t": round(self._last_t, 3) if self._last_t >= 0 else None,
            "next_sample_in_s": round(max(0.0, self._next_due - time.monotonic()), 2),
            "duty_cap": self.settings.perception_duty_cap,
            "deferred_for_chat": self._deferred,
        }

    async def _run(self) -> None:
        while not self._stopping:
            try:
                session = self.state.session
                if session is None:
                    await self._leave_session()
                    await asyncio.sleep(0.25)
                    continue
                if session.id != self._session_id:
                    await self._begin_session(session)

                clock = session.clock
                snapshot = clock.snapshot()
                if not clock.initialized:
                    self._set_status("waiting")
                    await asyncio.sleep(0.2)
                    continue
                if not snapshot.playing:
                    self._set_status("paused")
                    await asyncio.sleep(0.2)
                    continue

                # Cooperative GPU priority: chat is the product; a frame is
                # optional. Once chat has a scene, do not start a vision call
                # while a refill is in flight or the buffer is critical.
                if self.timeline.latest is not None and (
                    self.state.chat_generating
                    or self.state.chat_buffer_level < self.settings.chat_critical_buffer
                ):
                    self._deferred += 1
                    self._set_status("yielding")
                    await asyncio.sleep(0.2)
                    continue

                now = time.monotonic()
                if self._generation != snapshot.generation:
                    self._handle_discontinuity(snapshot.generation, snapshot.t_read)

                if (
                    snapshot.duration is not None
                    and snapshot.t_read >= snapshot.duration - 0.25
                ):
                    self._set_status("end")
                    await asyncio.sleep(0.5)
                    continue

                if now < self._next_due:
                    await asyncio.sleep(min(0.25, self._next_due - now))
                    continue

                await self._sample(session, snapshot)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - last-resort loop guard
                logger.exception("perception loop failed")
                self._record_error(str(exc))
                await asyncio.sleep(2.0)

    async def _begin_session(self, session: SessionRecord) -> None:
        await self._close_source()
        self.timeline.reset()
        self._session_id = session.id
        self._generation = session.clock.generation
        self._last_t = -1.0
        self._samples = 0
        self._errors = 0
        self._last_error = ""
        self._last_latency_s = None
        self._deferred = 0
        self._boost_until = 0.0
        self._next_due = time.monotonic() + 0.4
        self._source = await self._make_source(session)
        self._set_status("priming")

    async def _leave_session(self) -> None:
        if self._session_id is None and self._source is None:
            if self._status != "idle":
                self._set_status("idle")
            return
        await self._close_source()
        self.timeline.reset()
        self._session_id = None
        self._generation = None
        self._last_t = -1.0
        self._boost_until = 0.0
        self._set_status("idle")

    async def _make_source(self, session: SessionRecord) -> FrameSource:
        if self._source_factory is not None:
            source = self._source_factory(session.item.path)
            if asyncio.iscoroutine(source):  # pragma: no cover - custom async factory
                source = await source
            return source
        return FileFrameSource(
            session.item.path,
            ffmpeg=self.settings.ffmpeg,
            max_edge=self.settings.frame_max_edge,
            timeout_s=self.settings.frame_timeout_s,
        )

    def _model_name(self) -> str:
        try:
            from ..llm.profile import load_profile

            return load_profile(str(self.settings.model_profile)).alias
        except Exception:
            return "twitch-sim"

    async def _get_observer(self) -> Observer:
        if self._observer is not None:
            return self._observer
        if self._observer_factory is not None:
            self._observer = self._observer_factory()
            return self._observer
        self._client = LLMClient(
            base_url=f"http://{self.settings.llama_host}:{self.settings.llama_port}",
            model=self._model_name(),
            timeout_s=self.settings.llm_timeout_s,
        )
        self._observer = Observer(self._client)
        return self._observer

    async def _sample(self, session: SessionRecord, snapshot: Any) -> None:
        if self._source is None:
            return
        t_read = float(snapshot.t_read)
        if snapshot.duration is not None:
            t_read = min(t_read, max(0.0, float(snapshot.duration) - 0.5))
        if t_read <= self._last_t + 0.01:
            self._next_due = time.monotonic() + 0.5
            return

        started = time.monotonic()
        self._set_status("live")
        try:
            frame = await self._source.frame_at(t_read)
            observer = await self._get_observer()
            observation = await observer.observe(frame, t_read)
            observation.scene = self._with_speech(observation.scene)
        except (FrameError, ObservationError, LLMError) as exc:
            self._record_error(str(exc))
            self._next_due = time.monotonic() + max(
                1.0, float(self.settings.perception_error_backoff_s)
            )
            return
        finished = time.monotonic()

        # A seek/pause happened while the model was thinking. The observation
        # describes a timeline that no longer exists; drop it rather than show
        # a stale scene.
        if (
            self.state.session is None
            or self.state.session.id != session.id
            or session.clock.generation != snapshot.generation
        ):
            return

        self._last_t = observation.scene.t
        self._last_latency_s = observation.latency_s
        self._samples += 1
        self._errors = 0
        self._last_error = ""
        self._set_status("live")

        derived = self.timeline.add(
            observation.scene,
            generation=snapshot.generation,
            latency_s=observation.latency_s,
        )
        if observation.scene.scene_change:
            self._boost_until = finished + self.settings.perception_boost_s

        snapshot_payload = self.timeline.snapshot(limit=30, event_limit=20)
        self.state.bus.publish_nowait(
            events.SCENE_UPDATE,
            {
                "scene": observation.scene.model_dump(),
                "generation": snapshot.generation,
                "latency_s": round(observation.latency_s, 4),
                "snapshot": snapshot_payload,
                "service": self.metrics(),
                "derived_events": [event.as_dict() for event in derived],
            },
        )
        for event in derived:
            self.state.bus.publish_nowait(
                events.SCENE_EVENT,
                {
                    "event": event.as_dict(),
                    "scene": observation.scene.model_dump(),
                    "snapshot": snapshot_payload,
                },
            )
            if event.kind == "topic_shift":
                self.state.bus.publish_nowait(
                    events.TOPIC_SHIFT, {"event": event.as_dict()}
                )

        self._next_due = finished + self._next_gap(finished - started)

    def _with_speech(self, scene: Any) -> Any:
        """Fold the latest microphone transcript into the video observation.

        Mic input has no look-ahead, so only speech that is not ahead of the
        observed frame is merged. The scene contract already carries
        ``speech_text``/``speech_energy``/``source`` (PLAN.md §5.2).
        """
        speech = getattr(self.state, "latest_speech", None)
        if not isinstance(speech, dict):
            return scene
        text = " ".join(str(speech.get("text") or "").split())
        if not text:
            return scene
        try:
            speech_t = float(speech.get("t") or 0.0)
            energy = float(speech.get("energy") or 0.0)
        except (TypeError, ValueError):
            return scene
        if speech_t > float(scene.t) + 1.0 or float(scene.t) - speech_t > 10.0:
            return scene
        return scene.model_copy(
            update={
                "speech_text": text[:300],
                "speech_energy": min(1.0, max(0.0, energy)),
                "source": "mic",
            }
        )

    def _next_gap(self, elapsed_s: float) -> float:
        elapsed = max(0.0, float(elapsed_s))
        cap = min(max(float(self.settings.perception_duty_cap), 0.05), 1.0)
        # Keep perception below its share of model time. With a 1.5 s vision
        # call this is ~2.8 s of idle between calls, slightly above the 3 s
        # baseline in PLAN.md §5.1; the hard cap wins.
        cap_gap = elapsed * (1.0 / cap - 1.0)
        gap = max(float(self.settings.perception_interval_s) - elapsed, cap_gap)
        if time.monotonic() < self._boost_until:
            gap = min(gap, float(self.settings.perception_scene_change_interval_s))
        return max(0.05, gap)

    def _handle_discontinuity(self, generation: int, t_read: float) -> None:
        latest = self.timeline.latest
        self._generation = generation
        self._last_t = -1.0
        self._boost_until = 0.0
        self._next_due = time.monotonic() + 0.4
        if latest is not None and (
            t_read < latest.t - 1.0 or abs(t_read - latest.t) > 4.0
        ):
            self.timeline.reset()
        self._set_status("priming")

    def _record_error(self, message: str) -> None:
        self._errors += 1
        self._last_error = message[:500]
        was_error = self._status == "error"
        self._set_status("error")
        if was_error:
            self._publish_status()
        logger.warning("perception error: %s", self._last_error)

    def _set_status(self, status: str) -> None:
        if status == self._status:
            return
        self._status = status
        self._publish_status()

    def _publish_status(self) -> None:
        # The bus closes during shutdown; a status event is not worth keeping
        # the lifecycle alive for.
        with contextlib.suppress(RuntimeError):
            self.state.bus.publish_nowait(
                events.SYSTEM_STATUS,
                {
                    "perception": {
                        **self.metrics(),
                        "error": self._last_error or None,
                        "lookahead_s": self.state.lookahead_s,
                    }
                },
            )

    async def _close_source(self) -> None:
        source = self._source
        self._source = None
        if source is not None:
            with contextlib.suppress(Exception):
                await source.close()
