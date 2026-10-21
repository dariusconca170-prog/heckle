"""The 90 s scene timeline and the moments derived from it (PLAN.md §5.4).

This is deliberately a small, boring ring buffer. Chat, clips and polls all
read the same timeline, so they cannot disagree about what just happened. The
buffer is pruned by playback time (not wall time), which keeps it correct when
the stream is paused or a seek re-primes the source.
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from ..events import SceneEvent
from .scene import SceneState

_WORD = re.compile(r"[^\W_]+(?:['-][^\W_]+)*", re.UNICODE)
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "while",
        "with",
    }
)
_FAIL = ("dies", "death", "dead", "fails", "failed", "failing", "game over", "loses", "lost")
_CLUTCH = (
    "wins",
    "won",
    "winning",
    "clutch",
    "clutches",
    "clutching",
    "success",
    "succeeds",
    "clears",
    "scores",
    "scoring",
    "survives",
    "victory",
)
_FAIL_RE = re.compile(r"\b(?:" + "|".join(re.escape(token) for token in _FAIL) + r")\b")
_CLUTCH_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(token) for token in _CLUTCH) + r")\b"
)


@dataclass(slots=True, frozen=True)
class TimelineEntry:
    scene: SceneState
    received_at: float
    generation: int
    latency_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene.model_dump(),
            "received_at": round(self.received_at, 3),
            "generation": self.generation,
            "latency_s": round(self.latency_s, 4),
        }


class SceneTimeline:
    """A lock-protected ring buffer of observations for one session."""

    def __init__(self, *, window_s: float = 90.0, max_entries: int = 512) -> None:
        self.window_s = max(1.0, float(window_s))
        self.max_entries = max(8, int(max_entries))
        self._entries: deque[TimelineEntry] = deque(maxlen=self.max_entries)
        self._events: deque[SceneEvent] = deque(maxlen=128)
        self._lock = threading.RLock()
        self._last_event: SceneEvent | None = None

    def reset(self) -> None:
        with self._lock:
            self._entries.clear()
            self._events.clear()
            self._last_event = None

    @property
    def latest(self) -> SceneState | None:
        with self._lock:
            return self._entries[-1].scene if self._entries else None

    @property
    def latest_entry(self) -> TimelineEntry | None:
        with self._lock:
            return self._entries[-1] if self._entries else None

    def at(self, t: float, *, tolerance: float = 0.75) -> SceneState | None:
        """Return the newest scene observed at or before ``t``.

        Timelines are sampled every few seconds, so a small tolerance keeps a
        trigger at 6.2 s from skipping a perfectly good observation at 6.0 s.
        """
        cutoff = max(0.0, float(t)) + max(0.0, float(tolerance))
        with self._lock:
            for entry in reversed(self._entries):
                if entry.scene.t <= cutoff:
                    return entry.scene
            return self._entries[0].scene if self._entries else None

    def add(
        self,
        scene: SceneState,
        *,
        generation: int = 0,
        latency_s: float = 0.0,
        received_at: float | None = None,
    ) -> list[SceneEvent]:
        with self._lock:
            previous = self._entries[-1].scene if self._entries else None
            if previous is not None and scene.t < previous.t - 1.0:
                # A backward jump is a new timeline. The service normally resets
                # first; this guard keeps a late response from poisoning history.
                self._entries.clear()
                self._events.clear()
                self._last_event = None
                previous = None

            cutoff = scene.t - self.window_s
            while self._entries and self._entries[0].scene.t < cutoff:
                self._entries.popleft()

            entry = TimelineEntry(
                scene=scene,
                received_at=time.time() if received_at is None else received_at,
                generation=generation,
                latency_s=max(0.0, latency_s),
            )
            self._entries.append(entry)
            events = self._derive_events(previous, scene)
            for event in events:
                self._events.append(event)
            if events:
                self._last_event = events[-1]
            return events

    def entries(self, *, limit: int | None = None) -> list[TimelineEntry]:
        with self._lock:
            items = list(self._entries)
        if limit is not None:
            items = items[-max(0, limit) :]
        return items

    def events(self, *, limit: int | None = None) -> list[SceneEvent]:
        with self._lock:
            items = list(self._events)
        if limit is not None:
            items = items[-max(0, limit) :]
        return items

    def snapshot(self, *, limit: int = 30, event_limit: int = 20) -> dict[str, Any]:
        with self._lock:
            entries = list(self._entries)
            events = list(self._events)
            current = entries[-1].scene if entries else None
            updated_at = entries[-1].received_at if entries else None
        latencies = sorted(entry.latency_s for entry in entries if entry.latency_s > 0)
        recent_latencies = latencies[-60:]
        return {
            "current": current.model_dump() if current else None,
            "timeline": [
                entry.as_dict() for entry in entries[-max(1, limit) :]
            ],
            "events": [event.as_dict() for event in events[-max(0, event_limit) :]],
            "sample_count": len(entries),
            "event_count": len(events),
            "window_s": self.window_s,
            "latency_s": {
                "last": round(latencies[-1], 4) if latencies else None,
                "p50": _percentile(recent_latencies, 0.50),
                "p95": _percentile(recent_latencies, 0.95),
            },
            "updated_at": round(updated_at, 3) if updated_at else None,
        }

    def _derive_events(
        self, previous: SceneState | None, scene: SceneState
    ) -> list[SceneEvent]:
        events: list[SceneEvent] = []
        text = f"{scene.action} {scene.summary}".casefold()

        if _FAIL_RE.search(text):
            events.append(
                SceneEvent(
                    t=scene.t,
                    kind="fail",
                    intensity=scene.confidence,
                    detail=scene.summary[:160],
                )
            )
        elif _CLUTCH_RE.search(text):
            events.append(
                SceneEvent(
                    t=scene.t,
                    kind="clutch",
                    intensity=scene.confidence,
                    detail=scene.summary[:160],
                )
            )

        if scene.scene_change and scene.surprise >= 0.85:
            events.append(
                SceneEvent(t=scene.t, kind="jumpscare", intensity=scene.surprise, detail=scene.summary[:160])
            )
        elif scene.scene_change and scene.surprise >= 0.55:
            events.append(
                SceneEvent(t=scene.t, kind="big_change", intensity=scene.surprise, detail=scene.summary[:160])
            )

        if previous is not None and self._is_topic_shift(previous, scene):
            events.append(
                SceneEvent(t=scene.t, kind="topic_shift", intensity=0.5, detail=scene.summary[:160])
            )

        return events

    def _is_topic_shift(self, previous: SceneState, scene: SceneState) -> bool:
        if scene.t - previous.t > 20.0:
            return False
        if scene.confidence < 0.35 or previous.confidence < 0.35:
            return False
        if 0.0 < scene.t - previous.t < 1.0:
            return False
        previous_words = _keywords(previous.summary)
        current_words = _keywords(scene.summary)
        if len(previous_words) < 2 or len(current_words) < 2:
            return False
        overlap = len(previous_words & current_words) / max(
            1, len(previous_words | current_words)
        )
        if overlap >= 0.12:
            return False
        recent_shift = (
            self._last_event is not None
            and self._last_event.kind == "topic_shift"
            and scene.t - self._last_event.t < 12.0
        )
        return not recent_shift


def _keywords(text: str) -> set[str]:
    return {
        word.casefold()
        for word in _WORD.findall(text)
        if len(word) > 1 and word.casefold() not in _STOPWORDS
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * percentile))))
    return round(ordered[index], 4)
