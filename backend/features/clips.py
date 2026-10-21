"""Clip detection (M7, PLAN.md §10.4).

Chat spikes are the signal: a 10 s rolling message rate is scored against a
5-minute baseline. ``z > 3`` opens a candidate covering the preceding 15 s and
the next 5 s, which is then exported with ffmpeg and given a thumbnail. The
reason string quotes the three highest-engagement messages from the window, so
it is evidence from this stream, not generated copy.

Everything here is playback time. If the GPU is busy, clipping still works:
it only reads chat samples and calls ffmpeg.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import re
import secrets
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import events

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..state import AppState

logger = logging.getLogger(__name__)

_SAFE_STREAM = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass(slots=True)
class MessageSample:
    t: float
    handle: str
    text: str
    words: int
    emotes: tuple[str, ...] = ()

    @property
    def engagement(self) -> float:
        caps = 1.5 if self.text.isupper() and len(self.text) > 2 else 0.0
        return 1.0 + 0.25 * self.words + 0.8 * len(self.emotes) + caps


@dataclass(slots=True)
class RateStats:
    rate: float
    mean: float
    std: float
    z: float


@dataclass(slots=True)
class ClipCandidate:
    stream_id: str
    start_t: float
    end_t: float
    score: float
    peak_rate: float
    z: float
    reason: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    id: int = 0
    path: str = ""
    thumb_path: str = ""
    status: str = "candidate"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "stream_id": self.stream_id,
            "start_t": round(self.start_t, 3),
            "end_t": round(self.end_t, 3),
            "duration_s": round(max(0.0, self.end_t - self.start_t), 3),
            "score": round(self.score, 3),
            "peak_rate": round(self.peak_rate, 3),
            "z": round(self.z, 2),
            "reason": self.reason,
            "messages": self.messages,
            "path": self.path,
            "thumb_path": self.thumb_path,
            "status": self.status,
        }


class ClipService:
    """Playback-time spike detector with a failure-tolerant exporter."""

    def __init__(
        self,
        state: AppState,
        *,
        ffmpeg: str | None = None,
        ffprobe: str | None = None,
    ) -> None:
        self.state = state
        self.settings = state.settings
        self.ffmpeg = ffmpeg or self.settings.ffmpeg
        self.ffprobe = ffprobe or "ffprobe"
        self._task: asyncio.Task[None] | None = None
        self._subscription: events.Subscription | None = None
        self._stopping = False
        self._session_id: str | None = None
        self._messages: deque[MessageSample] = deque(maxlen=6000)
        self._pending: list[ClipCandidate] = []
        self._last_candidate_t = -10_000.0
        self._exports: set[asyncio.Task[None]] = set()
        self._status = "idle"
        self._last_error = ""
        self._created = 0
        self._exported = 0
        self._failed = 0
        self._failed_ids: set[int] = set()

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._subscription = self.state.bus.subscribe([events.CHAT_MESSAGE], maxsize=4096)
        self._stopping = False
        self._status = "live"
        self._task = asyncio.create_task(self._run(), name="clip-service")

    async def stop(self) -> None:
        self._stopping = True
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        for export in list(self._exports):
            export.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await export
        self._exports.clear()
        if self._subscription is not None:
            self.state.bus.unsubscribe(self._subscription)
            self._subscription = None
        self._status = "stopped"

    def metrics(self) -> dict[str, Any]:
        return {
            "status": self._status,
            "session_id": self._session_id,
            "messages_seen": len(self._messages),
            "pending": len(self._pending),
            "clips": self._created,
            "exported": self._exported,
            "failed": self._failed,
            "last_error": self._last_error,
            "z_threshold": self.settings.clip_z_threshold,
        }

    # -- loop --------------------------------------------------------------

    async def _run(self) -> None:
        subscription = self._subscription
        if subscription is None:
            return
        while not self._stopping:
            try:
                drained = 0
                while (event := subscription.get_nowait()) is not None:
                    if event.type == events.CHAT_MESSAGE:
                        self.record(event.payload)
                    drained += 1
                    if drained >= 512:
                        break
                session = self.state.session
                if session is None:
                    self._reset_session()
                else:
                    if session.id != self._session_id:
                        self._begin_session(session)
                    snapshot = session.clock.snapshot()
                    if snapshot.playing:
                        self.tick(snapshot.t_play, intensity=float(
                            getattr(self.state.timeline.latest, "surprise", 0.2) or 0.2
                        ))
                    self._advance_exports(snapshot.t_play, snapshot.duration)
                await asyncio.sleep(0.2 if drained else 0.4)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - last-resort loop guard
                logger.exception("clip loop failed")
                self._last_error = str(exc)[:400]
                await asyncio.sleep(1.0)

    def _begin_session(self, session: Any) -> None:
        self._session_id = session.id
        self._messages.clear()
        self._pending.clear()
        self._last_candidate_t = -10_000.0
        self._status = "live"

    def _reset_session(self) -> None:
        self._session_id = None
        self._messages.clear()
        self._pending.clear()
        self._status = "idle"

    # -- detection ---------------------------------------------------------

    def record(self, payload: dict[str, Any]) -> bool:
        if str(payload.get("kind", "chat")) != "chat":
            return False
        text = " ".join(str(payload.get("text") or "").split())
        if not text:
            return False
        t = _as_float(payload.get("release_t"), _as_float(payload.get("t"), 0.0))
        self._messages.append(
            MessageSample(
                t=t,
                handle=str(payload.get("handle") or "viewer"),
                text=text,
                words=int(payload.get("word_count") or len(text.split())),
                emotes=tuple(str(item) for item in payload.get("emotes") or []),
            )
        )
        return True

    def seed(self, samples: list[MessageSample]) -> None:
        """Test/debug hook for a deterministic baseline."""
        self._messages.extend(samples)

    def rate_stats(self, t_play: float) -> RateStats:
        t = max(0.0, float(t_play))
        window = max(2.0, float(self.settings.clip_window_s))
        baseline = max(30.0, float(self.settings.clip_baseline_s))
        recent = [sample for sample in self._messages if t - window < sample.t <= t]
        rate = len(recent) / window
        base_start = max(0.0, t - baseline)
        base_end = max(base_start + 1.0, t - window)
        buckets = Counter(
            int(sample.t)
            for sample in self._messages
            if base_start <= sample.t < base_end
        )
        span = max(1.0, base_end - base_start)
        total = sum(buckets.values())
        mean = total / span
        sum_squares = sum(count * count for count in buckets.values())
        variance = max(0.0, sum_squares / span - mean * mean)
        std = math.sqrt(variance)
        z = (rate - mean) / max(std, 0.35)
        return RateStats(rate=rate, mean=mean, std=std, z=z)

    def evaluate(self, t_play: float, *, intensity: float = 0.2) -> ClipCandidate | None:
        if self._session_id is None or t_play < 45.0:
            return None
        if t_play - self._last_candidate_t < float(self.settings.clip_min_gap_s):
            return None
        stats = self.rate_stats(t_play)
        if stats.z <= float(self.settings.clip_z_threshold):
            return None
        start = max(0.0, t_play - float(self.settings.clip_preroll_s))
        end = t_play + float(self.settings.clip_postroll_s)
        samples = [
            sample
            for sample in self._messages
            if start <= sample.t <= t_play
        ]
        reason = build_reason(samples)
        duration = max(1.0, end - start)
        score = stats.rate * (1.0 + max(0.0, intensity)) * duration
        return ClipCandidate(
            stream_id=self._session_id,
            start_t=start,
            end_t=end,
            score=score,
            peak_rate=stats.rate,
            z=stats.z,
            reason=reason,
            messages=[
                {"handle": sample.handle, "text": sample.text, "t": round(sample.t, 2)}
                for sample in samples[-8:]
            ],
        )

    def tick(self, t_play: float, *, intensity: float = 0.2) -> ClipCandidate | None:
        candidate = self.evaluate(t_play, intensity=intensity)
        if candidate is None:
            return None
        self._last_candidate_t = t_play
        repo = self.state.persistence.repo
        if repo is not None:
            try:
                candidate.id = repo.add_clip(
                    candidate.stream_id,
                    start_t=candidate.start_t,
                    end_t=candidate.end_t,
                    reason=candidate.reason,
                    score=candidate.score,
                )
            except Exception as exc:  # pragma: no cover - disk failures
                self._last_error = str(exc)[:400]
        self._pending.append(candidate)
        self._created += 1
        self._publish(events.CLIP_CANDIDATE, {"clip": candidate.as_dict()})
        self._queue({"kind": "clip", "start_t": candidate.start_t, "end_t": candidate.end_t})
        return candidate

    def manual_clip(self, start_t: float, end_t: float, *, reason: str = "") -> ClipCandidate:
        if self._session_id is None and self.state.session is not None:
            self._begin_session(self.state.session)
        if self._session_id is None:
            raise ValueError("start a stream first")
        start = max(0.0, float(start_t))
        end = max(start + 1.0, float(end_t))
        samples = [sample for sample in self._messages if start <= sample.t <= end]
        candidate = ClipCandidate(
            stream_id=self._session_id,
            start_t=start,
            end_t=end,
            score=sum(sample.engagement for sample in samples),
            peak_rate=len(samples) / max(1.0, end - start),
            z=0.0,
            reason=reason.strip() or build_reason(samples),
            messages=[
                {"handle": sample.handle, "text": sample.text, "t": round(sample.t, 2)}
                for sample in samples[-8:]
            ],
        )
        repo = self.state.persistence.repo
        if repo is not None:
            try:
                candidate.id = repo.add_clip(
                    candidate.stream_id,
                    start_t=candidate.start_t,
                    end_t=candidate.end_t,
                    reason=candidate.reason,
                    score=candidate.score,
                )
            except Exception as exc:  # pragma: no cover - disk failures
                self._last_error = str(exc)[:400]
        self._pending.append(candidate)
        self._created += 1
        self._publish(events.CLIP_CANDIDATE, {"clip": candidate.as_dict()})
        return candidate

    def _advance_exports(self, t_play: float, duration: float | None) -> None:
        ready = []
        for candidate in list(self._pending):
            if t_play + 0.5 >= candidate.end_t:
                ready.append(candidate)
                self._pending.remove(candidate)
            elif duration is not None and candidate.end_t > duration - 0.25:
                candidate.end_t = max(candidate.start_t + 1.0, float(duration))
                ready.append(candidate)
                self._pending.remove(candidate)
        for candidate in ready:
            self._start_export(candidate)

    def _start_export(self, candidate: ClipCandidate) -> None:
        if len(self._exports) >= 2:
            self._pending.append(candidate)
            return
        task = asyncio.create_task(self._export(candidate), name="clip-export")
        self._exports.add(task)
        task.add_done_callback(self._exports.discard)

    async def _export(self, candidate: ClipCandidate) -> None:
        session = self.state.session
        if session is None:
            return
        try:
            path, thumb = await export_clip(
                candidate,
                Path(session.item.path),
                self.settings.clips_dir,
                ffmpeg=self.ffmpeg,
            )
        except Exception as exc:
            self._failed += 1
            if candidate.id:
                self._failed_ids.add(candidate.id)
            self._last_error = str(exc)[:400]
            candidate.status = "failed"
            self._publish(
                events.CLIP_CREATED,
                {"clip": candidate.as_dict(), "exported": False, "error": self._last_error},
            )
            return
        candidate.path = path
        candidate.thumb_path = thumb
        candidate.status = "exported"
        self._exported += 1
        repo = self.state.persistence.repo
        if repo is not None and candidate.id:
            try:
                repo.update_clip_export(candidate.id, path=path, thumb_path=thumb)
            except Exception as exc:  # pragma: no cover - disk failures
                self._last_error = str(exc)[:400]
        self._publish(
            events.CLIP_CREATED,
            {"clip": candidate.as_dict(), "exported": True},
        )

    # -- reads -------------------------------------------------------------

    def clips(self, *, limit: int = 50) -> list[dict[str, Any]]:
        if self._session_id is None:
            return []
        repo = self.state.persistence.repo
        if repo is None:
            return [candidate.as_dict() for candidate in self._pending]
        try:
            rows = repo.clips(self._session_id, limit=limit)
            for row in rows:
                clip_id = int(row.get("id") or 0)
                if row.get("path"):
                    row["status"] = "exported"
                elif clip_id in self._failed_ids:
                    row["status"] = "failed"
                else:
                    row["status"] = "exporting"
                row.setdefault("messages", [])
            return rows
        except Exception as exc:  # pragma: no cover - disk failures
            self._last_error = str(exc)[:400]
            return []

    def resolve(self, relative: str) -> Path | None:
        """Resolve a stored clip path inside the clips directory only."""
        if not relative:
            return None
        root = Path(self.settings.clips_dir).resolve()
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        return candidate if candidate.is_file() else None

    def _publish(self, type_: str, payload: dict[str, Any]) -> None:
        with contextlib.suppress(RuntimeError):
            self.state.bus.publish_nowait(type_, payload)

    def _queue(self, cue: dict[str, Any]) -> None:
        cues = getattr(self.state, "chat_cues", None)
        if cues is not None:
            cues.append(cue)


def build_reason(samples: list[MessageSample]) -> str:
    """Quote the three highest-engagement messages in the window."""
    if not samples:
        return "chat went quiet, then exploded"
    ranked = sorted(samples, key=lambda sample: sample.engagement, reverse=True)
    picked: list[MessageSample] = []
    seen: set[str] = set()
    for sample in ranked:
        key = sample.text.casefold()
        if key in seen:
            continue
        seen.add(key)
        picked.append(sample)
        if len(picked) == 3:
            break
    return " · ".join(f'{sample.handle}: "{sample.text}"' for sample in picked)


async def export_clip(
    candidate: ClipCandidate,
    source: Path,
    clips_dir: Path,
    *,
    ffmpeg: str = "ffmpeg",
    timeout_s: float = 120.0,
) -> tuple[str, str]:
    """Cut the segment, scale to 720p and grab a midpoint thumbnail."""
    if not source.is_file():
        raise RuntimeError(f"source video is missing: {source}")
    stream_dir = Path(clips_dir) / _SAFE_STREAM.sub("_", candidate.stream_id or "stream")
    stream_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{int(candidate.start_t)}-{secrets.token_hex(3)}"
    output = stream_dir / f"{stem}.mp4"
    thumb = stream_dir / f"{stem}.jpg"
    duration = max(1.0, candidate.end_t - candidate.start_t)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-ss",
        f"{candidate.start_t:.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.3f}",
        "-vf",
        "scale=-2:720",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "22",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(output),
    ]
    await _run(command, timeout_s=timeout_s)
    thumb_command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-ss",
        f"{duration / 2:.3f}",
        "-i",
        str(output),
        "-frames:v",
        "1",
        "-vf",
        "scale=-2:360",
        "-q:v",
        "4",
        str(thumb),
    ]
    await _run(thumb_command, timeout_s=timeout_s)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("ffmpeg produced no clip file")
    relative = output.relative_to(Path(clips_dir)).as_posix()
    thumb_relative = thumb.relative_to(Path(clips_dir)).as_posix() if thumb.is_file() else ""
    return relative, thumb_relative


async def _run(command: list[str], *, timeout_s: float) -> None:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
    except TimeoutError as exc:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        with contextlib.suppress(Exception):
            await process.wait()
        raise RuntimeError("ffmpeg timed out exporting the clip") from exc
    if process.returncode != 0:
        detail = stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(detail[-400:] or "ffmpeg failed")


def _as_float(value: object, default: float) -> float:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


__all__ = [
    "ClipCandidate",
    "ClipService",
    "MessageSample",
    "RateStats",
    "build_reason",
    "export_clip",
]
