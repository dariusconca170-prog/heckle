"""The single authoritative playback clock.

Two clocks exist in the app and must never be mixed:

* ``t_play`` — where the viewer actually is. The browser is the source of truth,
  throttled to 4 Hz over the WebSocket.
* ``t_read`` — where perception reads from, ``t_play + lookahead``. File playback
  knows the future, so the model can pre-generate reactions and the release
  queue can reveal them against ``t_play`` like a human noticing a moment late.

A seek, pause or resume is a *discontinuity*: ``generation`` increments and every
consumer must flush queued work generated against the old timeline.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ClockSnapshot:
    t_play: float
    t_read: float
    playing: bool
    rate: float
    duration: float | None
    generation: int
    lookahead_s: float
    wall_time: float

    def as_dict(self) -> dict[str, object]:
        return {
            "t_play": round(self.t_play, 3),
            "t_read": round(self.t_read, 3),
            "playing": self.playing,
            "rate": self.rate,
            "duration": self.duration,
            "generation": self.generation,
            "lookahead_s": self.lookahead_s,
            "wall_time": round(self.wall_time, 3),
        }


@dataclass(slots=True, frozen=True)
class ClockUpdate:
    snapshot: ClockSnapshot
    seek: bool
    discontinuity: bool
    paused: bool
    resumed: bool


class PlaybackClock:
    """Holds ``t_play``/``t_read`` and detects timeline discontinuities."""

    def __init__(self, *, lookahead_s: float = 5.5, seek_threshold_s: float = 1.25) -> None:
        if lookahead_s < 0:
            raise ValueError("lookahead_s must be >= 0")
        if seek_threshold_s <= 0:
            raise ValueError("seek_threshold_s must be > 0")
        self.lookahead_s = float(lookahead_s)
        self.seek_threshold_s = float(seek_threshold_s)
        self._t_play = 0.0
        self._playing = False
        self._rate = 1.0
        self._duration: float | None = None
        self._generation = 0
        self._updated_monotonic = time.monotonic()
        self._wall_time = time.time()
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def duration(self) -> float | None:
        return self._duration

    def set_duration(self, duration: float | None) -> None:
        if duration is not None and (not math.isfinite(duration) or duration <= 0):
            duration = None
        self._duration = duration
        if duration is not None:
            self._t_play = min(self._t_play, duration)

    def reset(self, *, duration: float | None = None) -> None:
        self._t_play = 0.0
        self._playing = False
        self._rate = 1.0
        self._duration = duration
        self._updated_monotonic = time.monotonic()
        self._wall_time = time.time()
        self._initialized = False
        self._generation += 1

    def _clamp(self, t: float) -> float:
        t = max(0.0, t)
        if self._duration is not None:
            t = min(t, self._duration)
        return t

    def projected_playback(self, now: float | None = None) -> float:
        """Best estimate of ``t_play`` between browser updates."""
        now = time.monotonic() if now is None else now
        t = self._t_play
        if self._playing:
            t += max(0.0, now - self._updated_monotonic) * self._rate
        return self._clamp(t)

    def snapshot(self, now: float | None = None) -> ClockSnapshot:
        now = time.monotonic() if now is None else now
        t_play = self.projected_playback(now)
        t_read = self._clamp(t_play + self.lookahead_s)
        return ClockSnapshot(
            t_play=t_play,
            t_read=t_read,
            playing=self._playing,
            rate=self._rate,
            duration=self._duration,
            generation=self._generation,
            lookahead_s=self.lookahead_s,
            wall_time=self._wall_time,
        )

    def update(
        self,
        t_play: float,
        *,
        playing: bool | None = None,
        rate: float | None = None,
        seek: bool = False,
        now: float | None = None,
    ) -> ClockUpdate:
        """Apply a browser clock report and classify it.

        A normal 4 Hz report is accepted as authoritative. A jump larger than
        ``seek_threshold_s`` away from the projected position is a seek and
        bumps ``generation``.
        """
        if not math.isfinite(t_play):
            raise ValueError("t_play must be finite")
        now = time.monotonic() if now is None else now
        new_playing = self._playing if playing is None else bool(playing)
        new_rate = self._rate if rate is None else float(rate)
        if not math.isfinite(new_rate) or new_rate <= 0:
            new_rate = 1.0
        new_rate = min(max(new_rate, 0.25), 4.0)
        new_t = self._clamp(float(t_play))

        first = not self._initialized
        expected = self._clamp(self.projected_playback(now)) if not first else new_t
        is_seek = bool(seek) or (not first and abs(new_t - expected) > self.seek_threshold_s)
        was_playing = self._playing
        paused = (not first) and was_playing and not new_playing
        resumed = (not first) and (not was_playing) and new_playing
        discontinuity = is_seek or paused or resumed
        if discontinuity:
            self._generation += 1

        self._t_play = new_t
        self._playing = new_playing
        self._rate = new_rate
        self._updated_monotonic = now
        self._wall_time = time.time()
        self._initialized = True

        return ClockUpdate(
            snapshot=self.snapshot(now),
            seek=is_seek,
            discontinuity=discontinuity,
            paused=paused,
            resumed=resumed,
        )

    def pause(self) -> ClockUpdate:
        return self.update(self.projected_playback(), playing=False)

    def resume(self) -> ClockUpdate:
        return self.update(self.projected_playback(), playing=True)
