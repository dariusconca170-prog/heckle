"""Playback-time chat scheduler (M3).

The LLM fills text; this service owns every timing decision:

* Hawkes process -> trigger times
* weighted author selection -> persona
* lognormal perception delay -> release time
* pre-generation buffer -> no model call in the release path
* deterministic mutations -> casing, typos, emotes, copy-pasta

No wall-clock timer decides when chat appears. The only wall clock here is the
loop tick that checks the playback clock.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import random
import secrets
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .. import events
from ..chatter.guard import MAX_WORDS, template_for, word_count
from ..chatter.personas import Persona
from ..chatter.service import ChatMessage, ChatterService
from ..features.polls import poll_line
from ..features.stream_events import system_ack
from ..perception.scene import SceneState
from .emote_tables import EMOTES, crowd_line
from .hawkes import (
    HawkesProcess,
    active_chatters_for_viewers,
    scene_intensity,
)
from .mutations import copy_pasta_count, mutate
from .roster import AuthorPool, CrowdRoster
from .threads import ParentRef, ReplyPlanner, build_argument

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..state import AppState, SessionRecord

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Planned:
    """One message slot before its text exists."""

    trigger: float
    release: float
    persona: Persona
    tier: str
    reply: ParentRef | None = None
    mention: str | None = None


@dataclass(slots=True)
class ScheduledMessage:
    message: ChatMessage
    release_t: float
    trigger_t: float
    tier: str
    ready: bool = False


class DynamicsService:
    def __init__(
        self,
        state: AppState,
        chatter: ChatterService,
        *,
        rng: random.Random | None = None,
        crowd: CrowdRoster | None = None,
        core_share: float = 0.52,
    ) -> None:
        self.state = state
        self.chatter = chatter
        self.settings = state.settings
        self.rng = rng or random.Random()
        self.crowd = crowd or CrowdRoster()
        self.authors = AuthorPool(crowd=self.crowd, core_share=core_share, rng=self.rng)
        self.threads = ReplyPlanner(self.rng)
        self.hawkes = HawkesProcess(
            active_chatters=active_chatters_for_viewers(1200),
            intensity=0.5,
        )

        self._task: asyncio.Task[None] | None = None
        self._refill_task: asyncio.Task[None] | None = None
        self._stopping = False
        self._session_id: str | None = None
        self._generation: int | None = None
        self._last_t = -1.0
        self._buffer: list[ScheduledMessage] = []
        self._seen_events: set[str] = set()
        self._first_done = False
        self._event_boost = 0.0
        self._last_event_t = -1.0
        self._boost_t = 0.0
        self._status = "idle"
        self._last_error = ""
        self._next_refill = 0.0
        self._last_status_publish = 0.0
        self._catching_up = False
        self._released = 0
        self._generated = 0
        self._templates = 0
        self._copies = 0
        self._dropped = 0
        self._late = 0
        self._active_chatters = active_chatters_for_viewers(1200)
        self._copy_window: dict[str, float] = {}
        self._primed = False
        self._lore_beat_done = False
        self._lore_beat_pending: str | None = None
        self._recent_refs: deque[ParentRef] = deque(maxlen=40)
        self._replies = 0
        self._arguments = 0
        self._mentions = 0
        self._next_argument_at = 25.0
        self._poll_context: dict[str, Any] | None = None
        self._poll_context_until = 0.0

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stopping = False
        self._task = asyncio.create_task(self._run(), name="chat-dynamics")

    async def stop(self) -> None:
        self._stopping = True
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        refill = self._refill_task
        self._refill_task = None
        if refill is not None:
            refill.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await refill
        self._buffer.clear()
        self.state.chat_buffer_level = 0
        self.state.chat_generating = False
        self._status = "stopped"

    def metrics(self) -> dict[str, Any]:
        scene = self.state.timeline.latest
        return {
            "status": self._status,
            "session_id": self._session_id,
            "buffer_level": len(self._buffer),
            "target_buffer": int(self.settings.chat_target_buffer),
            "low_buffer": int(self.settings.chat_low_buffer),
            "critical_buffer": int(self.settings.chat_critical_buffer),
            "rate": round(self.hawkes.rate, 3),
            "intensity": round(self.hawkes.intensity, 3),
            "excitation": round(self.hawkes.excitation, 3),
            "cooling_down": self.hawkes.cooling_down,
            "active_chatters": self._active_chatters,
            "released": self._released,
            "generated": self._generated,
            "templates": self._templates,
            "copies": self._copies,
            "dropped": self._dropped,
            "late_released": self._late,
            "replies": self._replies,
            "reply_rate": round(self._replies / self._released, 3) if self._released else 0.0,
            "arguments": self._arguments,
            "mentions": self._mentions,
            "last_error": self._last_error,
            "scene_t": round(scene.t, 3) if scene is not None else None,
            "next_refill_in_s": round(max(0.0, self._next_refill - time.monotonic()), 2),
        }

    async def _run(self) -> None:
        while not self._stopping:
            try:
                session = self.state.session
                if session is None:
                    await self._leave_session()
                    await asyncio.sleep(0.2)
                    continue
                if session.id != self._session_id:
                    self._begin_session(session)
                snapshot = session.clock.snapshot()
                if not session.clock.initialized:
                    self._set_status("waiting")
                    await asyncio.sleep(0.2)
                    continue
                if self._generation != snapshot.generation:
                    self._handle_seek(snapshot.generation, snapshot.t_play)
                if not snapshot.playing:
                    self._set_status("paused")
                    await self._publish_status(force=False)
                    await asyncio.sleep(0.15)
                    continue
                if snapshot.duration is not None and snapshot.t_play >= snapshot.duration - 0.05:
                    self._set_status("end")
                    await self._publish_status(force=False)
                    await asyncio.sleep(0.5)
                    continue

                t_play = snapshot.t_play
                self._last_t = max(self._last_t, t_play)
                self._apply_scene_events(t_play)
                self._drain_cues(t_play)
                self._decay_event_boost(t_play)
                self._update_context()
                self.hawkes.advance_to(t_play)
                self._release_due(t_play)
                self._inject_first(t_play)
                self._bootstrap_chat(t_play)
                self._maybe_argument(t_play)
                self._maybe_refill(t_play)
                await self._publish_status(force=self._status != "live")
                self._set_status("live")
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - last-resort loop guard
                logger.exception("chat dynamics loop failed")
                self._last_error = str(exc)[:400]
                self._set_status("error")
                await asyncio.sleep(1.0)

    def _begin_session(self, session: SessionRecord) -> None:
        self._session_id = session.id
        self._generation = session.clock.generation
        self._buffer.clear()
        self._seen_events.clear()
        self._first_done = False
        self._event_boost = 0.0
        self._last_event_t = -1.0
        self._boost_t = 0.0
        self._last_t = -1.0
        self._released = 0
        self._generated = 0
        self._templates = 0
        self._copies = 0
        self._dropped = 0
        self._late = 0
        self._copy_window.clear()
        self._primed = False
        self._lore_beat_done = False
        self._lore_beat_pending = None
        self._catching_up = False
        self._next_refill = 0.0
        self._recent_refs.clear()
        self._replies = 0
        self._arguments = 0
        self._mentions = 0
        self._next_argument_at = self.rng.uniform(25.0, 60.0)
        self._poll_context = None
        self._poll_context_until = 0.0
        self._active_chatters = active_chatters_for_viewers(session.viewer_count)
        self.hawkes.reset(0.0)
        self.hawkes.set_context(active_chatters=self._active_chatters, intensity=0.5)
        self.state.chat_buffer_level = 0
        self.state.chat_generating = False
        self._set_status("priming")

    async def _leave_session(self) -> None:
        if self._session_id is None:
            if self._status != "idle":
                self._set_status("idle")
            return
        self._session_id = None
        self._generation = None
        self._buffer.clear()
        self._seen_events.clear()
        self._recent_refs.clear()
        self._first_done = False
        self._primed = False
        self._catching_up = False
        self._poll_context = None
        self.state.chat_buffer_level = 0
        self.state.chat_generating = False
        self._set_status("idle")
        await self._publish_status(force=True)

    def _handle_seek(self, generation: int, t_play: float) -> None:
        self._generation = generation
        self._buffer.clear()
        self._seen_events.clear()
        self._last_t = t_play
        self._event_boost = 0.0
        self._boost_t = t_play
        self._copy_window.clear()
        self._primed = True
        self._lore_beat_done = False
        self._lore_beat_pending = None
        self._catching_up = False
        self._recent_refs.clear()
        self._next_argument_at = t_play + self.rng.uniform(25.0, 60.0)
        self._poll_context = None
        self._poll_context_until = 0.0
        self.state.chat_buffer_level = 0
        self.hawkes.reset(t_play)
        self._next_refill = time.monotonic() + 0.5
        self._set_status("seeking")

    def _apply_scene_events(self, t_play: float) -> None:
        for event in self.state.timeline.events(limit=40):
            key = f"{event.t:.2f}:{event.kind}"
            if key in self._seen_events:
                continue
            if event.t > t_play + 0.35:
                continue
            self._seen_events.add(key)
            self._event_boost = min(2.0, self._event_boost + max(0.0, event.intensity))
            self._last_event_t = max(self._last_event_t, event.t)
            self._schedule_burst(t_play, event.kind, event.intensity)

    def _decay_event_boost(self, t_play: float) -> None:
        dt = max(0.0, min(1.0, t_play - self._boost_t))
        self._boost_t = t_play
        if self._event_boost <= 0.0:
            return
        self._event_boost *= math.exp(-0.35 * dt)

    def _update_context(self) -> None:
        if self.state.live_viewers > 0:
            self._active_chatters = active_chatters_for_viewers(self.state.live_viewers)
        scene = self.state.timeline.latest
        intensity = scene_intensity(
            surprise=scene.surprise if scene else 0.2,
            speech_energy=scene.speech_energy if scene else 0.0,
            scene_change=scene.scene_change if scene else False,
            event_boost=self._event_boost,
        )
        runtime = getattr(self.state, "runtime", None)
        if runtime is not None:
            intensity *= float(runtime.intensity())
        self.hawkes.set_context(
            active_chatters=self._active_chatters,
            intensity=min(5.0, max(0.05, intensity)),
        )

    def _drain_cues(self, t_play: float) -> None:
        """Turn feature cues into chat lines. Timing still lives here."""
        while self.state.chat_cues:
            cue = self.state.chat_cues.popleft()
            kind = str(cue.get("kind", ""))
            if kind == "poll_open":
                self._poll_context = cue
                self._poll_context_until = t_play + 600.0
                self._schedule_cue_lines(
                    t_play,
                    [
                        poll_line("vote", cue, self.rng),
                        poll_line("vote", cue, self.rng),
                    ],
                    spread=(1.0, 5.0),
                )
            elif kind == "poll_close":
                self._poll_context = cue
                self._poll_context_until = t_play + 60.0
                lines = [poll_line("closed", cue, self.rng) for _ in range(3)]
                if cue.get("winner_option"):
                    lines.append(f"{cue['winner_option']} was never losing")
                self._schedule_cue_lines(t_play, lines, spread=(0.8, 5.5))
                self._event_boost = min(2.0, self._event_boost + 0.35)
            elif kind == "stream_event":
                event_kind = str(cue.get("event_kind", "sub"))
                self._event_boost = min(
                    2.0, self._event_boost + float(cue.get("boost", 0.5))
                )
                count = 4 if event_kind == "raid" else 2
                lines = [system_ack(event_kind, self.rng) for _ in range(count)]
                self._schedule_cue_lines(t_play, lines, spread=(0.6, 4.0))
            elif kind == "clip":
                self._schedule_cue_lines(
                    t_play,
                    ["CLIP IT", "clip that", "someone clip that"][:2],
                    spread=(0.5, 2.5),
                )
            elif kind == "speech":
                text = " ".join(str(cue.get("text", "")).split())
                if text:
                    phrase = " ".join(text.split()[:5])
                    lines = [
                        f"wait he said {phrase}",
                        "chat did you hear that",
                    ]
                    if self.rng.random() < 0.6:
                        lines.append(f"{phrase} LMAO")
                    self._schedule_cue_lines(t_play, lines, spread=(0.8, 5.0))
        if self._poll_context is not None and t_play >= self._poll_context_until:
            self._poll_context = None

    def _schedule_cue_lines(
        self,
        t_play: float,
        lines: list[str],
        *,
        spread: tuple[float, float],
    ) -> None:
        if len(self._buffer) >= 44:
            return
        for text in lines:
            if len(self._buffer) >= 44:
                return
            persona = self.authors.choose_crowd(recent_handles=self.authors.recent_handles())
            message = self._make_message(
                persona,
                text,
                source="template",
                t=t_play,
                text_override=text,
                humanize=False,
            )
            self._insert_message(
                message,
                release_t=t_play + self.rng.uniform(*spread),
                trigger_t=t_play,
                tier="crowd",
                ready=True,
            )
            self._templates += 1

    def _release_due(self, t_play: float) -> None:
        while self._buffer and self._buffer[0].release_t <= t_play + 0.001:
            scheduled = self._buffer.pop(0)
            late_s = t_play - scheduled.release_t
            if late_s > 5.0:
                self._dropped += 1
                continue
            if late_s > 0.3:
                self._late += 1
            if not scheduled.ready or not scheduled.message.text:
                self._fill_template(scheduled)
            message = scheduled.message
            message.created_at = time.time()
            self.chatter.remember(message)
            self._recent_refs.append(
                ParentRef(
                    id=message.id,
                    handle=message.handle,
                    text=message.text,
                    release_t=scheduled.release_t,
                    archetype=message.archetype,
                )
            )
            if message.reply_to:
                self._replies += 1
            if message.mentions:
                self._mentions += 1
            data = message.as_dict()
            data["release_t"] = round(scheduled.release_t, 3)
            data["trigger_t"] = round(scheduled.trigger_t, 3)
            data["late_s"] = round(late_s, 3)
            data["tier"] = scheduled.tier
            data["kind"] = "chat"
            with contextlib.suppress(RuntimeError):
                self.state.bus.publish_nowait(events.CHAT_MESSAGE, data)
            self._released += 1
            if late_s <= 5.0:
                self._maybe_copy_pasta(scheduled, t_play)

    def _inject_first(self, t_play: float) -> None:
        if self._first_done:
            return
        if t_play > 4.0:
            self._first_done = True
            return
        author = self.authors.choose_crowd()
        release = max(0.45, t_play + 0.2)
        first = self._make_message(
            author,
            "first",
            source="template",
            t=max(0.0, release - 0.2),
            text_override="first",
            humanize=False,
        )
        first.batch_id = "first"
        self._insert_message(
            first,
            release_t=release,
            trigger_t=max(0.0, release - 0.2),
            tier="crowd",
            ready=True,
        )
        flame = self.authors.choose_crowd()
        flame_message = self._make_message(
            flame,
            "no one asked",
            source="template",
            t=release,
            text_override="no one asked",
            humanize=False,
        )
        flame_message.batch_id = "first"
        self._insert_message(
            flame_message,
            release_t=release + 1.1,
            trigger_t=release,
            tier="crowd",
            ready=True,
        )
        self._first_done = True
        self._templates += 2


    def _schedule_burst(self, t_play: float, kind: str, intensity: float) -> None:
        """Inject a crowd reaction immediately for a derived scene event.

        Hawkes supplies ambient timing; a big scene event is a separate trigger.
        Keeping this path crowd-only means a burst never waits on the GPU.
        """
        if intensity < 0.6 or len(self._buffer) >= 44:
            return
        scene = self._scene_at(t_play)
        count = min(6, max(2, int(round(2 + intensity * 4))))
        for _ in range(count):
            persona = self.authors.choose_crowd(recent_handles=self.authors.recent_handles())
            text = crowd_line(persona.archetype, scene_objects=list(scene.objects), rng=self.rng)
            if kind in {"jumpscare", "big_change"} and self.rng.random() < 0.45:
                text = crowd_line("emote-only", scene_objects=list(scene.objects), rng=self.rng)
            message = self._make_message(
                persona,
                text,
                source="crowd",
                t=t_play,
                text_override=text,
            )
            self._insert_message(
                message,
                release_t=t_play + self.rng.uniform(0.35, 2.4),
                trigger_t=t_play,
                tier="crowd",
                ready=True,
            )
            self._generated += 1
            self._templates += 1


    def _bootstrap_chat(self, t_play: float) -> None:
        """Put a few deterministic crowd lines in flight at stream start.

        The first model batch can be several seconds cold; first-time viewers
        should not stare at an empty chat while it warms up. This is still the
        dynamics service deciding timing, not the model.
        """
        if self._primed or t_play < 0.8:
            return
        self._primed = True
        scene = self._scene_at(t_play)
        for index in range(3):
            persona = self.authors.choose_crowd(recent_handles=self.authors.recent_handles())
            text = crowd_line(persona.archetype, scene_objects=list(scene.objects), rng=self.rng)
            message = self._make_message(
                persona,
                text,
                source="crowd",
                t=t_play,
                text_override=text,
            )
            self._insert_message(
                message,
                release_t=t_play + 0.9 + index * 1.0 + self.rng.uniform(-0.1, 0.1),
                trigger_t=t_play,
                tier="crowd",
                ready=True,
            )
            self._generated += 1

    def _maybe_refill(self, t_play: float) -> None:
        if self._refill_task is not None and not self._refill_task.done():
            return
        if self._refill_task is not None:
            exc = self._refill_task.exception()
            if exc is not None:
                self._last_error = str(exc)[:400]
                self._set_status("error")
            self._refill_task = None
        if self.state.timeline.latest is None and t_play < 1.5:
            return
        level = len(self._buffer)
        if level >= self.settings.chat_target_buffer:
            return
        if level >= self.settings.chat_low_buffer and time.monotonic() < self._next_refill:
            return
        self._next_refill = time.monotonic() + 1.0
        self._refill_task = asyncio.create_task(
            self._refill(t_play, self._session_id, self._generation),
            name="chat-refill",
        )

    async def _refill(
        self,
        t_play: float,
        session_id: str | None,
        generation: int | None,
    ) -> None:
        try:
            count = min(
                int(self.settings.chat_batch_size),
                int(self.settings.chat_target_buffer) - len(self._buffer),
            )
            if count <= 0:
                return
            triggers = self.hawkes.next_events(
                count,
                rng=self.rng,
                max_span_s=6.0,
            )
            if not triggers:
                return

            recent = self.authors.recent_handles()
            planned: list[Planned] = []
            used_core: set[str] = set()
            for trigger in triggers:
                selection = self.authors.choose(
                    core_available=self.chatter.enabled,
                    recent_handles=recent,
                )
                if selection.tier == "core" and selection.persona.handle in used_core:
                    selection = self.authors.choose(
                        core_available=False,
                        recent_handles=recent,
                        force_tier="crowd",
                    )
                if selection.tier == "core":
                    used_core.add(selection.persona.handle)
                delay = min(9.0, max(0.6, self.rng.lognormvariate(1.7, 0.55)))
                release = trigger + delay + self.rng.uniform(-0.15, 0.15)
                release = max(release, t_play + 0.35)
                planned.append(
                    Planned(
                        trigger=trigger,
                        release=release,
                        persona=selection.persona,
                        tier=selection.tier,
                    )
                )

            # Threads are decided after every slot has a release time, so a
            # reply can only target a message that is already visible by then.
            for item in planned:
                item.reply = self.threads.maybe_reply(
                    item.persona.handle, item.release, list(self._recent_refs)
                )
                if item.reply is None:
                    item.mention = self.threads.maybe_mention(self._channel_name())

            scene = self._scene_at(
                sorted(item.trigger for item in planned)[len(planned) // 2]
            )
            lore_handles = self._plan_lore_beat(planned, t_play)
            core = [item for item in planned if item.tier == "core"]
            crowd = [item for item in planned if item.tier != "core"]

            # Fill the crowd tier immediately. These messages never touch the
            # GPU, so chat keeps moving even if a model call is slow or queued
            # behind a vision request. The core batch is inserted when it
            # returns; releases are re-based so it does not arrive stale.
            for item in crowd:
                text = crowd_line(
                    item.persona.archetype,
                    scene_objects=list(scene.objects),
                    rng=self.rng,
                )
                scheduled = self._make_message(
                    item.persona,
                    text,
                    source="crowd",
                    t=item.trigger,
                    text_override=text,
                    reply=item.reply,
                    mention=item.mention,
                )
                scheduled.source = "crowd"
                self._insert_message(
                    scheduled,
                    release_t=item.release,
                    trigger_t=item.trigger,
                    tier="crowd",
                    ready=True,
                )
                self._generated += 1

            if not core:
                self._last_error = ""
                return

            result = None
            if self.chatter.enabled:
                self.state.chat_generating = True
                try:
                    reply_targets = {
                        item.persona.handle: {
                            "handle": item.reply.handle,
                            "text": item.reply.quote,
                        }
                        for item in core
                        if item.reply is not None
                    }
                    result = await self.chatter.generate(
                        scene,
                        personas=[item.persona for item in core],
                        publish=False,
                        lore_handles=lore_handles,
                        reply_targets=reply_targets,
                        poll=(
                            self._poll_context
                            if self._poll_context is not None
                            and t_play < self._poll_context_until
                            else None
                        ),
                    )
                except Exception as exc:  # pragma: no cover - generation must degrade
                    self._last_error = str(exc)[:400]
                    result = None
                finally:
                    self.state.chat_generating = False
            else:
                self.state.chat_generating = False

            if (
                self.state.session is None
                or self.state.session.id != session_id
                or self._generation != generation
            ):
                # The beat stays pending: a seek flushed this batch, so the
                # next refill must request it again.
                return

            current_t = self.state.clock_snapshot().t_play
            earliest_release = min((item.release for item in core), default=current_t)
            shift = max(0.0, current_t + 0.25 - earliest_release)
            if shift:
                for item in core:
                    item.release += shift

            by_handle = {message.handle: message for message in result.messages} if result else {}
            for item in core:
                persona = item.persona
                generated = by_handle.get(persona.handle)
                if generated is None or not generated.text:
                    scheduled = self._make_message(
                        persona,
                        "",
                        source="template",
                        t=item.trigger,
                        text_override=self._template_text(persona, humanize=False),
                        reply=item.reply,
                        mention=item.mention,
                    )
                    scheduled.source = "template"
                    self._templates += 1
                else:
                    scheduled = generated
                    scheduled.t = item.trigger
                    scheduled.text = mutate(scheduled.text, persona, rng=self.rng)
                    scheduled.word_count = word_count(scheduled.text)
                    scheduled.source = "model" if scheduled.source == "model" else "template"
                    if scheduled.source == "template":
                        self._templates += 1
                    self._attach_thread(scheduled, persona, item.reply, item.mention)
                self._insert_message(
                    scheduled,
                    release_t=item.release,
                    trigger_t=item.trigger,
                    tier=item.tier,
                    ready=True,
                )
                self._generated += 1
                if persona.handle in lore_handles:
                    self._lore_beat_done = True
                    self._lore_beat_pending = None
            self._last_error = ""
        except Exception as exc:  # pragma: no cover - refill must never kill loop
            self._last_error = str(exc)[:400]
            logger.warning("chat refill failed: %s", self._last_error)
        finally:
            self.state.chat_generating = False

    def _plan_lore_beat(
        self, planned: list[Planned], t_play: float
    ) -> set[str]:
        """Once per session, make the lore-keeper reference an old stream.

        Without this, referencing memory is left to chance; the M4 acceptance
        criterion needs one guaranteed opportunity. The viewer still speaks in
        their normal slot, the model still writes the line, and the fallback in
        the chatter service only fires if the model ignores the request.
        """
        if self._lore_beat_done or t_play < 10.0 or not self.chatter.enabled:
            return set()
        keepers = [
            persona
            for persona in self.chatter.roster.personas
            if persona.archetype == "lore-keeper" and self.chatter.has_memory(persona.handle)
        ]
        if not keepers:
            return set()
        keeper = next(
            (persona for persona in keepers if persona.handle == self._lore_beat_pending),
            None,
        )
        if keeper is None:
            keeper = self.rng.choice(keepers)
            self._lore_beat_pending = keeper.handle
        chosen_index = next(
            (
                index
                for index, item in enumerate(planned)
                if item.tier == "core" and item.persona.handle == keeper.handle
            ),
            None,
        )
        if chosen_index is None:
            core_indexes = [index for index, item in enumerate(planned) if item.tier == "core"]
            if not core_indexes:
                return set()
            chosen_index = core_indexes[-1]
            planned[chosen_index].persona = keeper
        return {keeper.handle}

    def _scene_at(self, t: float) -> SceneState:
        scene = self.state.timeline.at(t)
        if scene is not None:
            return scene
        latest = self.state.timeline.latest
        if latest is not None and latest.t <= t + 0.75:
            return latest
        return SceneState(
            t=max(0.0, t),
            summary="the stream is starting; the scene is not clear yet",
            action="stream starting",
            mood="neutral",
            surprise=0.2,
            confidence=0.15,
        )

    def _make_message(
        self,
        persona: Persona,
        text: str,
        *,
        source: str,
        t: float,
        text_override: str | None = None,
        humanize: bool = True,
        reply: ParentRef | None = None,
        mention: str | None = None,
        kind: str = "chat",
    ) -> ChatMessage:
        value = text_override if text_override is not None else text
        if humanize:
            value = mutate(value, persona, rng=self.rng)
        if mention and reply is None:
            value = _with_mention(value, mention)
        return ChatMessage(
            id=secrets.token_hex(6),
            handle=persona.handle,
            archetype=persona.archetype,
            text=value,
            source=source,
            batch_id=secrets.token_hex(4),
            t=max(0.0, t),
            created_at=time.time(),
            word_count=word_count(value),
            kind=kind,
            reply_to=reply.id if reply is not None else None,
            reply_to_handle=reply.handle if reply is not None else "",
            reply_to_text=reply.quote if reply is not None else "",
            mentions=(mention,) if mention and reply is None else (),
            emotes=_emote_tokens(value),
        )

    def _attach_thread(
        self,
        message: ChatMessage,
        persona: Persona,
        reply: ParentRef | None,
        mention: str | None,
    ) -> None:
        if reply is not None:
            message.reply_to = reply.id
            message.reply_to_handle = reply.handle
            message.reply_to_text = reply.quote
        elif mention:
            message.text = _with_mention(message.text, mention)
            message.word_count = word_count(message.text)
            message.mentions = (mention,)
        message.emotes = _emote_tokens(message.text)
        if message.kind == "chat" and persona.archetype == "emote-only" and not message.emotes:
            message.emotes = ("LUL",)

    def _channel_name(self) -> str:
        session = self.state.session
        title = (session.title if session else "").strip()
        name = getattr(self.settings, "channel_name", "") or title or "streamer"
        cleaned = "".join(ch for ch in name if ch.isalnum() or ch == "_")
        return cleaned[:24] or "streamer"

    def _persona_by_archetype(self, archetype: str) -> Persona | None:
        for persona in self.chatter.roster.personas:
            if persona.archetype == archetype:
                return persona
        matches = [p for p in self.crowd.personas if p.archetype == archetype]
        return self.rng.choice(matches) if matches else None

    def _maybe_argument(self, t_play: float) -> None:
        """Guarantee one visible multi-message argument per ~10 minutes."""
        if t_play < self._next_argument_at:
            return
        self._next_argument_at = t_play + self.rng.uniform(75.0, 150.0)
        if len(self._buffer) > 34:
            return
        lines = build_argument(self.rng, lookup=self._persona_by_archetype)
        if not lines:
            return
        refs: list[ParentRef] = []
        cursor = t_play + 0.4
        for line in lines:
            cursor += line.delay_s
            parent = (
                refs[line.reply_index]
                if line.reply_index is not None and 0 <= line.reply_index < len(refs)
                else None
            )
            message = self._make_message(
                line.persona,
                line.text,
                source="template",
                t=cursor,
                text_override=line.text,
                humanize=False,
                reply=parent,
            )
            refs.append(
                ParentRef(
                    id=message.id,
                    handle=message.handle,
                    text=message.text,
                    release_t=cursor,
                    archetype=message.archetype,
                )
            )
            self._insert_message(
                message,
                release_t=cursor,
                trigger_t=t_play,
                tier="core" if line.persona in self.chatter.roster.personas else "crowd",
                ready=True,
            )
            self._templates += 1
        self._arguments += 1

    def _template_text(self, persona: Persona, *, humanize: bool = True) -> str:
        text = template_for(persona.archetype, self.rng.randrange(0, 12))
        if humanize:
            text = mutate(text, persona, rng=self.rng, allow_typos=False)
        return text

    def _fill_template(self, scheduled: ScheduledMessage) -> None:
        persona = self._persona_for(scheduled.message.handle) or self.crowd.random_persona(self.rng)
        scheduled.message.text = self._template_text(persona)
        scheduled.message.word_count = word_count(scheduled.message.text)
        scheduled.message.source = "template"
        scheduled.ready = True
        self._templates += 1

    def _persona_for(self, handle: str) -> Persona | None:
        found = self.chatter.roster.by_handle(handle)
        if found is not None:
            return found
        return self.crowd.by_handle.get(handle)

    def _insert_message(
        self,
        message: ChatMessage,
        *,
        release_t: float,
        trigger_t: float,
        tier: str,
        ready: bool = False,
    ) -> None:
        if len(self._buffer) >= 48:
            return
        self._buffer.append(
            ScheduledMessage(
                message=message,
                release_t=max(0.0, release_t),
                trigger_t=max(0.0, trigger_t),
                tier=tier,
                ready=ready,
            )
        )
        self._buffer.sort(key=lambda item: item.release_t)

    def _maybe_copy_pasta(self, scheduled: ScheduledMessage, t_play: float) -> None:
        if scheduled.message.source == "copypasta":
            return
        if scheduled.message.batch_id == "first":
            return
        text = scheduled.message.text.strip()
        if not text or len(text) > 40 or scheduled.message.archetype == "lurker":
            return
        if not self._copy_pasta_worthy(text):
            return
        key = text.casefold()
        last = self._copy_window.get(key)
        if last is not None and t_play - last < 30.0:
            return
        if self.rng.random() > 0.08 or len(self._buffer) >= 36:
            return
        self._copy_window[key] = t_play
        for old_key, old_t in list(self._copy_window.items()):
            if t_play - old_t > 90.0:
                self._copy_window.pop(old_key, None)
        count = copy_pasta_count(self.rng)
        recent = self.authors.recent_handles()
        for _ in range(count):
            persona = self.authors.choose_crowd(recent_handles=recent)
            value = text
            if persona.caps:
                value = value.upper()
            elif self.rng.random() < 0.6:
                value = value[0].lower() + value[1:] if value and value[0].isalpha() else value
            message = ChatMessage(
                id=secrets.token_hex(6),
                handle=persona.handle,
                archetype=persona.archetype,
                text=value,
                source="copypasta",
                batch_id=scheduled.message.batch_id,
                t=scheduled.message.t,
                created_at=time.time(),
                word_count=word_count(value),
            )
            self._insert_message(
                message,
                release_t=t_play + self.rng.uniform(1.0, 3.0),
                trigger_t=t_play,
                tier="crowd",
                ready=True,
            )
            self._copies += 1


    @staticmethod
    def _copy_pasta_worthy(text: str) -> bool:
        if len(text) > 18:
            return False
        lowered = text.casefold()
        short_phrases = {
            "w", "l", "gg", "bro", "no way", "lets go", "let's go", "lul",
            "kekw", "pog", "copium", "ratio", "it's over", "it's joever",
            "we're so back", "we are so back",
        }
        if lowered.strip(" !?.") in short_phrases:
            return True
        if text.isupper():
            return True
        tokens = {token.casefold() for token in text.split()}
        return any(emote.casefold() in tokens for emote in EMOTES)

    async def _publish_status(self, *, force: bool) -> None:
        now = time.monotonic()
        if not force and now - self._last_status_publish < 0.5:
            return
        self._last_status_publish = now
        level = len(self._buffer)
        self.state.chat_buffer_level = level
        payload = self.metrics()
        with contextlib.suppress(RuntimeError):
            self.state.bus.publish_nowait(events.CHAT_BUFFER_LEVEL, payload)
        if level < self.settings.chat_critical_buffer and not self._catching_up:
            self._catching_up = True
            with contextlib.suppress(RuntimeError):
                self.state.bus.publish_nowait(
                    events.CHAT_CATCHING_UP,
                    {"level": level, "target": int(self.settings.chat_target_buffer)},
                )
        elif level >= self.settings.chat_low_buffer:
            self._catching_up = False

    def _set_status(self, status: str) -> None:
        self._status = status


def _with_mention(text: str, mention: str) -> str:
    if not mention:
        return text
    value = f"{mention} {text}".strip()
    words = value.split()
    if len(words) > MAX_WORDS:
        value = " ".join(words[:MAX_WORDS])
    return value


def _emote_tokens(text: str) -> tuple[str, ...]:
    cleaned = [token.strip(".,!?;:") for token in str(text).split()]
    return tuple(token for token in cleaned if token and token in EMOTES)


__all__ = ["DynamicsService", "ScheduledMessage"]
