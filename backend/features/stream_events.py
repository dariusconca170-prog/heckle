"""Subs, gift subs, raids, follows and the viewer-count curve (M6).

These are synthetic stream events driven entirely by playback time. They render
as system messages, wake chat up briefly (the dynamics engine applies the
burst), and move the viewer counter along a slow ramp with noise and a spike
after a raid. No model call is involved, so an event never competes with chat
generation for the GPU.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import secrets
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .. import events

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..state import AppState

logger = logging.getLogger(__name__)

EVENT_KINDS = ("sub", "gift", "raid", "follow")
SYSTEM_HANDLES = {
    "sub": "SubBot",
    "gift": "GiftBot",
    "raid": "RaidBot",
    "follow": "FollowBot",
}


@dataclass(slots=True)
class StreamMoment:
    """One scheduled non-chat event."""

    kind: str
    handle: str
    amount: int = 1
    text: str = ""
    t: float = 0.0


class StreamEventService:
    """Playback-time event generator and viewer counter."""

    def __init__(
        self,
        state: AppState,
        *,
        rng: random.Random | None = None,
        names: list[str] | None = None,
        sub_interval_s: tuple[float, float] = (240.0, 480.0),
        raid_interval_s: tuple[float, float] = (900.0, 1800.0),
        follow_interval_s: tuple[float, float] = (90.0, 300.0),
    ) -> None:
        self.state = state
        self.rng = rng or random.Random()
        self.names = list(names or [])
        self.sub_interval_s = sub_interval_s
        self.raid_interval_s = raid_interval_s
        self.follow_interval_s = follow_interval_s
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self._session_id: str | None = None
        self._base_viewers = 0
        self._live_viewers = 0
        self._raid_boost = 0.0
        self._last_t: float | None = None
        self._next_publish = 0.0
        self._next_sub = 0.0
        self._next_raid = 0.0
        self._next_follow = 0.0
        self._status = "idle"
        self._last_error = ""
        self._events = 0

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stopping = False
        self._task = asyncio.create_task(self._run(), name="stream-events")
        self._status = "live"

    async def stop(self) -> None:
        self._stopping = True
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._status = "stopped"

    def set_names(self, names: list[str]) -> None:
        self.names = list(names)

    def metrics(self) -> dict[str, Any]:
        return {
            "status": self._status,
            "session_id": self._session_id,
            "base_viewers": self._base_viewers,
            "live_viewers": self._live_viewers,
            "raid_boost": round(1.0 + self._raid_boost, 3),
            "events": self._events,
            "last_error": self._last_error,
        }

    # -- loop --------------------------------------------------------------

    async def _run(self) -> None:
        while not self._stopping:
            try:
                session = self.state.session
                if session is None:
                    self._reset_session()
                    await asyncio.sleep(0.3)
                    continue
                if session.id != self._session_id:
                    self._begin_session(session)
                if session.clock.initialized and session.clock.snapshot().playing:
                    self.tick(session.clock.snapshot().t_play)
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - last-resort loop guard
                logger.exception("stream event loop failed")
                self._last_error = str(exc)[:400]
                await asyncio.sleep(1.0)

    def _begin_session(self, session: Any) -> None:
        self._session_id = session.id
        self._base_viewers = max(10, int(session.viewer_count))
        self._live_viewers = int(self._base_viewers * 0.5)
        self._raid_boost = 0.0
        self._last_t = None
        self._next_publish = 0.0
        self._next_sub = self.rng.uniform(*self.sub_interval_s)
        self._next_raid = self.rng.uniform(*self.raid_interval_s)
        self._next_follow = self.rng.uniform(*self.follow_interval_s)

    def _reset_session(self) -> None:
        self._session_id = None
        self._base_viewers = 0
        self._live_viewers = 0
        self._raid_boost = 0.0
        self._last_t = None
        self._status = "idle"

    # -- public tick -------------------------------------------------------

    def tick(self, t_play: float) -> int:
        """Advance the curve/event clocks. Returns the number of events fired."""
        if self._session_id is None:
            return 0
        t_play = max(0.0, float(t_play))
        dt = 0.5 if self._last_t is None else max(0.0, min(t_play - self._last_t, 2.0))
        if self._last_t is not None and t_play < self._last_t - 1.0:
            # A seek re-anchors the schedule instead of replaying old events.
            self._next_sub = t_play + self.rng.uniform(*self.sub_interval_s)
            self._next_raid = t_play + self.rng.uniform(*self.raid_interval_s)
            self._next_follow = t_play + self.rng.uniform(*self.follow_interval_s)
        self._last_t = t_play
        self._raid_boost = max(0.0, self._raid_boost - 0.006 * dt)
        self._live_viewers = self.viewer_curve(t_play)
        self.state.live_viewers = self._live_viewers

        fired = 0
        if t_play >= self._next_sub:
            self._fire("sub", t_play)
            self._next_sub = t_play + self.rng.uniform(*self.sub_interval_s)
            fired += 1
        if t_play >= self._next_raid:
            self._fire("raid", t_play)
            self._next_raid = t_play + self.rng.uniform(*self.raid_interval_s)
            fired += 1
        if t_play >= self._next_follow:
            self._fire("follow", t_play)
            self._next_follow = t_play + self.rng.uniform(*self.follow_interval_s)
            fired += 1

        if t_play >= self._next_publish:
            self._next_publish = t_play + 2.0
            self._publish_viewers(t_play)
        return fired

    def viewer_curve(self, t_play: float) -> int:
        """Slow ramp, plateau, ±3 % noise, and a decaying raid spike."""
        elapsed = max(0.0, float(t_play))
        ramp = 0.45 + 0.55 * min(1.0, elapsed / 75.0)
        wobble = 1.0 + self.rng.uniform(-0.03, 0.03)
        value = self._base_viewers * ramp * wobble * (1.0 + self._raid_boost)
        return max(1, int(round(value)))

    def trigger(self, kind: str, *, t_play: float | None = None) -> StreamMoment | None:
        """Manually fire an event (API/debug)."""
        if self._session_id is None and self.state.session is not None:
            self._begin_session(self.state.session)
        if self._session_id is None:
            return None
        t = self._t_play() if t_play is None else float(t_play)
        return self._fire(kind, t)

    # -- events ------------------------------------------------------------

    def _fire(self, kind: str, t_play: float) -> StreamMoment:
        handle = self._name()
        if kind == "sub":
            moment = StreamMoment(kind, handle, 1, f"{handle} subscribed with Prime!", t_play)
        elif kind == "gift":
            amount = self.rng.choice((1, 3, 5, 10))
            moment = StreamMoment(
                kind,
                handle,
                amount,
                f"{handle} gifted {amount} subs to the community!",
                t_play,
            )
        elif kind == "raid":
            amount = self.rng.randint(12, 420)
            moment = StreamMoment(
                kind,
                handle,
                amount,
                f"{handle} raided with {amount} viewers!",
                t_play,
            )
            self._raid_boost = min(0.35, self._raid_boost + 0.10 + amount / 4000.0)
            self._live_viewers = self.viewer_curve(t_play)
            self.state.live_viewers = self._live_viewers
        else:
            moment = StreamMoment(kind, handle, 1, f"{handle} followed", t_play)

        self._events += 1
        payload = {
            "event": {
                "t": round(t_play, 3),
                "kind": moment.kind,
                "detail": moment.text,
                "intensity": 0.9 if kind == "raid" else 0.6,
                "handle": moment.handle,
                "amount": moment.amount,
            },
            "viewer_count": self._live_viewers,
            "system_text": moment.text,
        }
        self._publish(events.STREAM_EVENT, payload)
        self._publish_system_message(moment)
        self._queue(
            {
                "kind": "stream_event",
                "event_kind": moment.kind,
                "handle": moment.handle,
                "amount": moment.amount,
                "text": moment.text,
                "boost": 0.5 if kind != "raid" else 0.9,
            }
        )
        return moment

    def _publish_system_message(self, moment: StreamMoment) -> None:
        bot = SYSTEM_HANDLES.get(moment.kind, "StreamBot")
        now = time.time()
        self._publish(
            events.CHAT_MESSAGE,
            {
                "id": secrets.token_hex(6),
                "handle": bot,
                "archetype": "system",
                "text": moment.text,
                "source": "system",
                "batch_id": f"{moment.kind}-{secrets.token_hex(3)}",
                "t": round(moment.t, 3),
                "created_at": now,
                "word_count": len(moment.text.split()),
                "kind": "system",
                "tier": "system",
                "release_t": round(moment.t, 3),
                "trigger_t": round(moment.t, 3),
                "late_s": 0.0,
                "mentions": [],
                "emotes": [],
            },
        )

    def _publish_viewers(self, t_play: float) -> None:
        self._publish(
            events.STREAM_EVENT,
            {
                "event": {
                    "t": round(t_play, 3),
                    "kind": "viewers",
                    "detail": "",
                    "intensity": 0.0,
                },
                "viewer_count": self._live_viewers,
                "raid_boost": round(1.0 + self._raid_boost, 3),
            },
        )

    def _name(self) -> str:
        if self.names:
            return self.rng.choice(self.names)
        return f"viewer_{self.rng.randrange(1000, 9999)}"

    def _publish(self, type_: str, payload: dict[str, Any]) -> None:
        with contextlib.suppress(RuntimeError):
            self.state.bus.publish_nowait(type_, payload)

    def _queue(self, cue: dict[str, Any]) -> None:
        cues = getattr(self.state, "chat_cues", None)
        if cues is not None:
            cues.append(cue)

    def _t_play(self) -> float:
        try:
            return float(self.state.clock_snapshot().t_play)
        except Exception:  # pragma: no cover - defensive
            return 0.0


def system_ack(kind: str, rng: random.Random) -> str:
    """Short in-character acknowledgement of a stream event."""
    if kind == "raid":
        return rng.choice(("o7", "W raid", "welcome raiders", "raid W", "o7 o7 o7"))
    if kind == "gift":
        return rng.choice(("gigachad", "W", "gifting goes hard", "actual gigachad"))
    if kind == "follow":
        return rng.choice(("W", "welcome", "followed too", "o7"))
    return rng.choice(("W", "gigachad", "o7", "sub W", "LETSGO"))


__all__ = ["EVENT_KINDS", "StreamEventService", "StreamMoment", "system_ack"]
