"""Polls (M6, PLAN.md §10.1).

A poll is a playback-time object: it opens at ``t_play``, closes after
``duration_s`` seconds of *playback*, and pauses when the video pauses. Votes
come from the same personas that chat with, weighted by archetype bias, a small
first-option edge and a herd effect toward the current leader. The service never
blocks on the model: poll chat lines are cues the dynamics engine schedules.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .. import events
from ..chatter.personas import Persona

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..state import AppState

logger = logging.getLogger(__name__)

MIN_OPTIONS = 2
MAX_OPTIONS = 10
MAX_QUESTION_CHARS = 140
MAX_OPTION_CHARS = 60


@dataclass(slots=True)
class Poll:
    id: int
    stream_id: str
    question: str
    options: list[str]
    opened_at: float
    duration_s: float
    allow_multiple: bool = False
    status: str = "open"
    counts: list[int] = field(default_factory=list)
    voters: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts)

    @property
    def closes_at(self) -> float:
        return self.opened_at + self.duration_s

    def winner(self) -> int | None:
        if not self.counts or self.total == 0:
            return None
        return max(range(len(self.counts)), key=lambda index: self.counts[index])

    def as_dict(self, *, t_play: float | None = None) -> dict[str, Any]:
        total = self.total
        percentages = [
            round((count / total) * 100, 1) if total else 0.0 for count in self.counts
        ]
        remaining = None
        elapsed = None
        if t_play is not None:
            elapsed = max(0.0, min(self.duration_s, t_play - self.opened_at))
            remaining = max(0.0, self.duration_s - elapsed)
        return {
            "id": self.id,
            "stream_id": self.stream_id,
            "question": self.question,
            "options": list(self.options),
            "counts": list(self.counts),
            "percentages": percentages,
            "total_votes": total,
            "opened_at": round(self.opened_at, 3),
            "closes_at": round(self.closes_at, 3),
            "duration_s": round(self.duration_s, 3),
            "elapsed_s": round(elapsed, 3) if elapsed is not None else None,
            "remaining_s": round(remaining, 3) if remaining is not None else None,
            "allow_multiple": self.allow_multiple,
            "status": self.status,
            "winner": self.winner(),
        }


class PollError(ValueError):
    """A poll request the API should reject with a 4xx."""


_ARCHETYPE_BIAS: dict[str, tuple[float, ...]] = {
    "hype": (1.5, 0.9, 0.8, 0.7),
    "wholesome": (1.0, 1.2, 1.0, 1.0),
    "meme": (1.1, 1.3, 0.9, 1.2),
    "angry": (1.0, 0.8, 1.3, 1.1),
    "doomer": (0.8, 1.0, 1.4, 1.0),
    "backseater": (1.2, 1.4, 0.9, 1.1),
    "lurker": (1.0, 1.0, 1.0, 1.0),
    "normie": (1.0, 1.1, 1.0, 1.0),
    "emote-only": (1.2, 1.0, 0.9, 1.0),
    "lore-keeper": (0.9, 1.0, 1.1, 1.3),
    "parser": (1.0, 1.0, 1.2, 1.1),
    "incongruent": (1.1, 0.9, 1.2, 1.0),
}


class PollService:
    """Timer + voter simulation for one open poll at a time."""

    def __init__(
        self,
        state: AppState,
        *,
        personas: list[Persona] | None = None,
        rng: random.Random | None = None,
        vote_rate: float = 0.55,
    ) -> None:
        self.state = state
        self.rng = rng or random.Random()
        self.vote_rate = min(1.0, max(0.0, float(vote_rate)))
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self._poll: Poll | None = None
        self._personas = list(personas or [])
        self._voters: list[Persona] = []
        self._last_t: float | None = None
        self._status = "idle"
        self._last_error = ""
        self._closed = 0

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stopping = False
        self._task = asyncio.create_task(self._run(), name="poll-service")

    async def stop(self) -> None:
        self._stopping = True
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._status = "stopped"

    def set_personas(self, personas: list[Persona]) -> None:
        self._personas = list(personas)

    def metrics(self) -> dict[str, Any]:
        return {
            "status": self._status,
            "current": self._poll.as_dict(t_play=self._t_play()) if self._poll else None,
            "closed": self._closed,
            "voters": len(self._voters),
            "last_error": self._last_error,
        }

    # -- API ---------------------------------------------------------------

    def open(
        self,
        question: str,
        options: list[str],
        *,
        duration_s: float = 60.0,
        allow_multiple: bool = False,
    ) -> Poll:
        session = self.state.session
        if session is None:
            raise PollError("start a stream before creating a poll")
        clean_question = " ".join(str(question).split())[:MAX_QUESTION_CHARS]
        clean_options = [
            " ".join(str(option).split())[:MAX_OPTION_CHARS]
            for option in options
            if str(option).strip()
        ]
        if not clean_question:
            raise PollError("question is required")
        if len(clean_options) < MIN_OPTIONS:
            raise PollError("a poll needs at least two options")
        if len(clean_options) > MAX_OPTIONS:
            raise PollError(f"a poll can have at most {MAX_OPTIONS} options")
        if not 5.0 <= float(duration_s) <= 3600.0:
            raise PollError("duration must be between 5 and 3600 seconds")
        if self._poll is not None and self._poll.status == "open":
            self.close(reason="replaced")

        self._prepare_voters()
        t_play = self._t_play()
        poll_id = 0
        repo = self.state.persistence.repo
        if repo is not None:
            try:
                poll_id = repo.create_poll(
                    session.id,
                    question=clean_question,
                    options=clean_options,
                    opened_at=t_play,
                    closes_at=t_play + float(duration_s),
                )
            except Exception as exc:  # pragma: no cover - disk failures
                self._last_error = str(exc)[:400]
        if not poll_id:
            poll_id = abs(hash((session.id, clean_question, time.time()))) % 2_000_000_000
        self._poll = Poll(
            id=poll_id,
            stream_id=session.id,
            question=clean_question,
            options=clean_options,
            opened_at=t_play,
            duration_s=float(duration_s),
            allow_multiple=bool(allow_multiple),
            counts=[0] * len(clean_options),
        )
        self._last_t = t_play
        self._status = "open"
        self._publish(
            events.POLL_OPENED,
            {
                "poll": self._poll.as_dict(t_play=t_play),
                "voters": len(self._voters),
            },
        )
        self._queue(
            {
                "kind": "poll_open",
                "question": clean_question,
                "options": list(clean_options),
            }
        )
        return self._poll

    def close(self, *, reason: str = "timer") -> Poll | None:
        poll = self._poll
        if poll is None or poll.status != "open":
            return None
        poll.status = "closed"
        t_play = self._t_play()
        repo = self.state.persistence.repo
        if repo is not None:
            try:
                repo.close_poll(poll.id, status="closed")
            except Exception as exc:  # pragma: no cover - disk failures
                self._last_error = str(exc)[:400]
        data = poll.as_dict(t_play=t_play)
        data["reason"] = reason
        self._closed += 1
        self._status = "closed"
        self._publish(events.POLL_CLOSED, {"poll": data, "reason": reason})
        self._publish(events.POLL_RESULTS, {"poll": data})
        winner = poll.winner()
        self._queue(
            {
                "kind": "poll_close",
                "question": poll.question,
                "options": list(poll.options),
                "counts": list(poll.counts),
                "winner": winner,
                "winner_option": poll.options[winner] if winner is not None else "",
                "total_votes": poll.total,
            }
        )
        return poll

    def current(self) -> Poll | None:
        return self._poll

    def snapshot(self) -> dict[str, Any] | None:
        return self._poll.as_dict(t_play=self._t_play()) if self._poll else None

    def history(self, *, limit: int = 20) -> list[dict[str, Any]]:
        session = self.state.session
        repo = self.state.persistence.repo
        if session is None or repo is None:
            return [self._poll.as_dict(t_play=self._t_play())] if self._poll else []
        try:
            return repo.polls(session.id, limit=limit)
        except Exception as exc:  # pragma: no cover - disk failures
            self._last_error = str(exc)[:400]
            return []

    def add_vote(self, handle: str, option_index: int) -> dict[str, Any]:
        """A vote from the human operator; chat votes come from ``tick``."""
        poll = self._poll
        if poll is None or poll.status != "open":
            raise PollError("no open poll")
        if not 0 <= int(option_index) < len(poll.options):
            raise PollError("option index out of range")
        if handle in poll.voters:
            raise PollError("this viewer already voted")
        self._record_vote(poll, handle, int(option_index))
        self._publish_vote(poll)
        return poll.as_dict(t_play=self._t_play())

    # -- loop --------------------------------------------------------------

    async def _run(self) -> None:
        while not self._stopping:
            try:
                session = self.state.session
                if session is None:
                    if self._poll is not None and self._poll.status == "open":
                        self.close(reason="stream_ended")
                    self._last_t = None
                    await asyncio.sleep(0.25)
                    continue
                if not session.clock.initialized:
                    await asyncio.sleep(0.25)
                    continue
                clock = session.clock.snapshot()
                if clock.playing:
                    self.tick(clock.t_play)
                await asyncio.sleep(0.25)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - last-resort loop guard
                logger.exception("poll loop failed")
                self._last_error = str(exc)[:400]
                await asyncio.sleep(1.0)

    def tick(self, t_play: float) -> Poll | None:
        """Advance the playback-time poll clock and simulate votes."""
        poll = self._poll
        if poll is None or poll.status != "open":
            return poll
        t_play = max(0.0, float(t_play))
        if self._last_t is None:
            self._last_t = t_play
        dt = max(0.0, min(t_play - self._last_t, 2.0))
        self._last_t = t_play
        elapsed = t_play - poll.opened_at
        if elapsed < 0:
            # Seeking backwards rewound the poll but not its results; keep the
            # poll open and let the timer run from zero again.
            elapsed = 0.0
        if elapsed >= poll.duration_s:
            self.close(reason="timer")
            return self._poll
        if dt > 0:
            self._simulate_votes(poll, dt=dt, remaining=max(1.0, poll.duration_s - elapsed))
        return poll

    # -- voting ------------------------------------------------------------

    def _prepare_voters(self) -> None:
        if self._voters:
            return
        pool = list(self._personas)
        if not pool:
            return
        self.rng.shuffle(pool)
        limit = min(48, max(8, len(pool)))
        self._voters = pool[:limit]

    def _simulate_votes(self, poll: Poll, *, dt: float, remaining: float) -> None:
        if not self._voters:
            return
        added = 0
        for persona in self._voters:
            if persona.handle in poll.voters:
                continue
            engagement = min(1.0, max(0.15, float(persona.chattiness))) * self.vote_rate
            probability = engagement * dt / max(remaining, 1.0)
            if self.rng.random() >= probability:
                continue
            self._record_vote(poll, persona.handle, self._pick_option(poll, persona))
            added += 1
            if added >= 3:
                break
        if added:
            self._publish_vote(poll)

    def _pick_option(self, poll: Poll, persona: Persona) -> int:
        bias = _ARCHETYPE_BIAS.get(persona.archetype, (1.0, 1.0, 1.0, 1.0))
        leader = poll.winner()
        weights: list[float] = []
        for index in range(len(poll.options)):
            weight = bias[index % len(bias)]
            if index == 0:
                weight *= 1.15
            if leader is not None and index == leader and poll.total:
                weight *= 1.0 + 0.45 * (poll.counts[index] / max(1, poll.total))
            weights.append(weight)
        # A few voters misread the options.
        if self.rng.random() < 0.05:
            return self.rng.randrange(len(poll.options))
        return self.rng.choices(range(len(poll.options)), weights=weights, k=1)[0]

    def _record_vote(self, poll: Poll, handle: str, option_index: int) -> None:
        poll.voters[handle] = option_index
        poll.counts[option_index] += 1
        repo = self.state.persistence.repo
        if repo is None:
            return
        try:
            repo.add_poll_vote(
                poll.id,
                handle=handle,
                option_index=option_index,
                t_playback=self._t_play(),
            )
        except Exception as exc:  # pragma: no cover - disk failures
            self._last_error = str(exc)[:400]

    def _publish_vote(self, poll: Poll) -> None:
        self._publish(
            events.POLL_VOTE,
            {
                "poll_id": poll.id,
                "counts": list(poll.counts),
                "total_votes": poll.total,
                "percentages": poll.as_dict(t_play=self._t_play())["percentages"],
            },
        )

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


def poll_line(kind: str, poll: dict[str, Any], rng: random.Random) -> str:
    """In-character chat line for a poll event (template tier, no model)."""
    options = [str(option) for option in poll.get("options") or []]
    if kind == "vote" and options:
        index = rng.randrange(len(options))
        return rng.choice(
            (
                f"voted {index + 1}",
                f"option {index + 1} easy",
                f"{options[index]} obviously",
                f"who voted {index + 1}",
            )
        )
    if kind == "closed":
        winner = poll.get("winner")
        if winner is None:
            return rng.choice(("rigged", "poll was rigged", "no one voted"))
        option = str(poll.get("winner_option") or (options[winner] if options else winner + 1))
        return rng.choice(
            (
                f"{option} won easy",
                "told you",
                "rigged",
                "who voted the other one",
                f"{option} was obvious",
                "ez",
            )
        )
    return "poll?"


__all__ = ["Poll", "PollError", "PollService", "poll_line"]
