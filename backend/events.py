"""In-process async event bus and the event-type vocabulary.

Modules communicate through this bus; they do not import each other's
internals (PLAN.md §3.1). Event type strings are part of the contract and should
only be extended, never renamed. Slow subscribers lose their oldest events
rather than stalling the producer — every feature must degrade gracefully.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

# --- Event type vocabulary -------------------------------------------------

CLOCK_UPDATE = "clock.update"
CLOCK_SEEK = "clock.seek"
CLOCK_PAUSE = "clock.pause"
CLOCK_PLAY = "clock.play"

SESSION_STARTED = "session.started"
SESSION_ENDED = "session.ended"

SCENE_UPDATE = "scene.update"
SCENE_EVENT = "scene.event"
TOPIC_SHIFT = "topic.shift"

CHAT_MESSAGE = "chat.message"
CHAT_BUFFER_LEVEL = "chat.buffer_level"
CHAT_CATCHING_UP = "chat.catching_up"

STREAM_EVENT = "stream.event"  # sub / gift / raid / follow / viewer_count

POLL_OPENED = "poll.opened"
POLL_VOTE = "poll.vote"
POLL_CLOSED = "poll.closed"
POLL_RESULTS = "poll.results"

CLIP_CANDIDATE = "clip.candidate"
CLIP_CREATED = "clip.created"

MIC_SPEECH = "mic.speech"  # transcript from the microphone (M8)

SYSTEM_STATUS = "system.status"


@dataclass(slots=True, frozen=True)
class Event:
    """An immutable event. ``seq`` is assigned by the bus on publish."""

    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    seq: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"type": self.type, "payload": self.payload, "ts": self.ts, "seq": self.seq}


@dataclass(slots=True, frozen=True)
class SceneEvent:
    """A derived moment in the scene timeline (PLAN.md §5.4)."""

    t: float
    kind: str
    intensity: float = 0.5
    detail: str = ""

    KINDS = frozenset(
        {"big_change", "jumpscare", "laugh", "fail", "clutch", "dead_air", "topic_shift"}
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "t": round(self.t, 3),
            "kind": self.kind,
            "intensity": self.intensity,
            "detail": self.detail,
        }


def _matches(pattern: str, event_type: str) -> bool:
    if pattern == "*":
        return True
    if pattern.endswith(".*"):
        return event_type.startswith(pattern[:-1])
    return pattern == event_type


class Subscription:
    """One consumer's bounded queue."""

    __slots__ = ("_queue", "_types", "_bus", "dropped")

    def __init__(self, bus: EventBus, types: tuple[str, ...] | None, maxsize: int) -> None:
        self._bus = bus
        self._types = types
        self._queue: asyncio.Queue[Event | None] = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0

    def wants(self, event_type: str) -> bool:
        return self._types is None or any(_matches(p, event_type) for p in self._types)

    def _offer(self, event: Event) -> None:
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
            self.dropped += 1
            self._queue.put_nowait(event)

    async def get(self) -> Event | None:
        """Return the next event, or ``None`` when the bus closes."""
        return await self._queue.get()

    def get_nowait(self) -> Event | None:
        """Return the next queued event without waiting, or ``None``."""
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def close(self) -> None:
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
            self._queue.put_nowait(None)

    def __enter__(self) -> Subscription:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._bus.unsubscribe(self)


class EventBus:
    """Fan-out bus with per-subscriber backpressure."""

    def __init__(self) -> None:
        self._subs: list[Subscription] = []
        self._seq = itertools.count(1)
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def subscribe(self, types: Iterable[str] | None = None, *, maxsize: int = 512) -> Subscription:
        if self._closed:
            raise RuntimeError("event bus is closed")
        patterns = tuple(types) if types is not None else None
        sub = Subscription(self, patterns, maxsize)
        self._subs.append(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        if sub in self._subs:
            self._subs.remove(sub)
        sub.close()

    def publish_nowait(self, type_: str, payload: dict[str, Any] | None = None) -> Event:
        if self._closed:
            raise RuntimeError("event bus is closed")
        event = Event(type=type_, payload=payload or {}, seq=next(self._seq))
        for sub in tuple(self._subs):
            if sub.wants(type_):
                sub._offer(event)
        return event

    async def publish(self, type_: str, payload: dict[str, Any] | None = None) -> Event:
        return self.publish_nowait(type_, payload)

    def close(self) -> None:
        self._closed = True
        for sub in tuple(self._subs):
            sub.close()
        self._subs.clear()
