"""The deterministic Hawkes clock for chat arrivals (PLAN.md §4).

The LLM never chooses when a message appears. This module produces trigger
times in **playback seconds**. The scheduler adds per-viewer perception delay
and releases the resulting message against the playhead, so pausing or seeking
the video pauses or clears the schedule too.

The implementation uses the standard exponential-kernel form:

    lambda(t) = mu(t) + sum(alpha * exp(-beta * (t - t_i)))

with a hard rate clamp and a global "catch your breath" cooldown.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class HawkesConfig:
    base_per_chatter: float = 0.015
    alpha: float = 0.35
    beta: float = 0.6
    max_rate: float = 8.0
    cooldown_after_s: float = 3.0
    cooldown_s: float = 2.0


class HawkesProcess:
    """Stateful point process; advance it with playback time, never wall time."""

    def __init__(
        self,
        *,
        config: HawkesConfig | None = None,
        active_chatters: int = 40,
        intensity: float = 0.3,
        t: float = 0.0,
    ) -> None:
        self.config = config or HawkesConfig()
        self.t = max(0.0, float(t))
        self.active_chatters = max(0, int(active_chatters))
        self.intensity = max(0.05, float(intensity))
        self._excitation = 0.0
        self._high_since: float | None = None
        self._cooldown_until = -math.inf
        self.cooldown_events = 0

    @property
    def rate(self) -> float:
        if self.t < self._cooldown_until:
            return 0.0
        mu = self.active_chatters * self.config.base_per_chatter * self.intensity
        return min(self.config.max_rate, max(0.0, mu + self._excitation))

    @property
    def excitation(self) -> float:
        return self._excitation

    @property
    def cooling_down(self) -> bool:
        return self.t < self._cooldown_until

    def set_context(self, *, active_chatters: int | None = None, intensity: float | None = None) -> None:
        if active_chatters is not None:
            self.active_chatters = max(0, int(active_chatters))
        if intensity is not None:
            self.intensity = min(5.0, max(0.05, float(intensity)))

    def reset(self, t: float = 0.0, *, intensity: float = 0.3) -> None:
        self.t = max(0.0, float(t))
        self.intensity = max(0.05, float(intensity))
        self._excitation = 0.0
        self._high_since = None
        self._cooldown_until = -math.inf

    def advance_to(self, t: float) -> None:
        if t <= self.t:
            return
        dt = t - self.t
        self._excitation *= math.exp(-self.config.beta * dt)
        if self._excitation < 1e-5:
            self._excitation = 0.0
        self.t = t
        if self._high_since is not None and t - self._high_since >= self.config.cooldown_after_s:
            self._start_cooldown()

    def excite(self, amount: float) -> None:
        self._excitation = min(
            self.config.max_rate * 2.0,
            max(0.0, self._excitation + max(0.0, float(amount))),
        )

    def record_event(self) -> None:
        self.excite(self.config.alpha)
        if self.rate >= self.config.max_rate:
            if self._high_since is None:
                self._high_since = self.t
            elif self.t - self._high_since >= self.config.cooldown_after_s:
                self._start_cooldown()
        else:
            self._high_since = None

    def next_events(
        self,
        count: int,
        *,
        rng: random.Random,
        max_span_s: float = 10.0,
    ) -> list[float]:
        """Advance the process and return up to ``count`` trigger times.

        The state is mutated: callers own these trigger times and should not
        call again for the same horizon. An empty list means the process stayed
        quiet for the span (also a valid chat outcome).
        """
        if count <= 0:
            return []
        end = self.t + max(0.0, float(max_span_s))
        events: list[float] = []
        guard = 0
        while len(events) < count and self.t < end and guard < count * 16 + 32:
            guard += 1
            rate = self.rate
            if rate <= 0.001:
                self.advance_to(end)
                break
            gap = rng.expovariate(rate)
            if self.t + gap > end:
                self.advance_to(end)
                break
            self.advance_to(self.t + gap)
            if self.cooling_down:
                continue
            events.append(self.t)
            self.record_event()
        return events

    def _start_cooldown(self) -> None:
        self._cooldown_until = self.t + self.config.cooldown_s
        self._high_since = None
        self._excitation = 0.0
        self.cooldown_events += 1

    def snapshot(self) -> dict[str, float | int | bool]:
        return {
            "t": round(self.t, 3),
            "rate": round(self.rate, 3),
            "excitation": round(self._excitation, 3),
            "intensity": round(self.intensity, 3),
            "active_chatters": self.active_chatters,
            "cooling_down": self.cooling_down,
            "cooldown_events": self.cooldown_events,
        }


def scene_intensity(
    *,
    surprise: float = 0.0,
    speech_energy: float = 0.0,
    scene_change: bool = False,
    event_boost: float = 0.0,
) -> float:
    """Map scene signals onto the 0.3..5.0 intensity scale in PLAN.md §4."""
    value = (
        0.3
        + 4.2 * _unit(surprise)
        + 2.0 * _unit(speech_energy)
        + (0.45 if scene_change else 0.0)
        + max(0.0, float(event_boost))
    )
    return min(5.0, max(0.3, value))


def active_chatters_for_viewers(viewers: int) -> int:
    """A small, stable active-chatter fraction for the UI's viewer count."""
    return max(8, min(80, round(max(0, int(viewers)) * 0.03)))


def _unit(value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return min(1.0, max(0.0, number))
