"""Persistence store: session lifecycle, message logging and memory.

This object is owned by ``AppState`` so that starting/ending a session is
synchronous with the rest of the lifecycle. The event-driven writer lives in
``service.py``. Splitting them this way means the second stream always sees
the first stream's memory even if the bus task has not drained yet.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from ..chatter.memory import (
    MemoryCandidate,
    MemoryEvent,
    MemoryMessage,
    extract_memories,
    prompt_memory,
)
from ..db.repo import MemoryRow, Repository, RepositoryError

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..config import Settings

logger = logging.getLogger(__name__)


class PersistenceStore:
    """Thin, failure-tolerant facade over :class:`Repository`."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.last_error = ""
        self._repo: Repository | None = None
        self._stream_id: str | None = None
        self._title = ""
        self._last_checkpoint_t = 0.0
        self._memory: dict[str, tuple[str, ...]] = {}
        self._candidates: dict[str, tuple[MemoryCandidate, ...]] = {}
        self._memory_handles: tuple[str, ...] = ()
        self._lore_handles: tuple[str, ...] = ()
        self._messages_logged = 0
        self._events_logged = 0
        self._memories_saved = 0
        self._checkpoints = 0
        self._live_ids: OrderedDict[str, int] = OrderedDict()
        if not settings.persistence_enabled:
            return
        try:
            self._repo = Repository(settings.db_path)
        except RepositoryError as exc:
            self._repo = None
            self.last_error = str(exc)[:400]
            logger.warning("persistence disabled: %s", self.last_error)

    # -- introspection -----------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._repo is not None

    @property
    def repo(self) -> Repository | None:
        return self._repo

    def set_roster(self, handles: tuple[str, ...], lore_handles: tuple[str, ...]) -> None:
        self._memory_handles = tuple(dict.fromkeys(handles))
        self._lore_handles = tuple(dict.fromkeys(lore_handles))

    def metrics(self) -> dict[str, Any]:
        stats: dict[str, int] = {}
        if self._repo is not None:
            try:
                stats = self._repo.stats()
            except Exception as exc:  # pragma: no cover - diagnostics only
                stats = {}
                self._record_error(exc)
        return {
            "enabled": self.enabled,
            "db_path": str(self.settings.db_path),
            "stream_id": self._stream_id,
            "messages_logged": self._messages_logged,
            "events_logged": self._events_logged,
            "memory_rows_saved": self._memories_saved,
            "memory_viewers": len(self._memory),
            "checkpoints": self._checkpoints,
            "checkpoint_interval_s": round(float(self.settings.memory_checkpoint_s), 1),
            "last_error": self.last_error,
            "totals": stats,
        }

    # -- session lifecycle -------------------------------------------------

    def begin_session(
        self,
        stream_id: str,
        *,
        title: str = "",
        category: str = "",
        source_path: str = "",
        meta: dict[str, Any] | None = None,
    ) -> None:
        self._stream_id = stream_id
        self._title = title
        self._last_checkpoint_t = 0.0
        self._messages_logged = 0
        self._events_logged = 0
        self._checkpoints = 0
        self._live_ids.clear()
        if self._repo is None:
            return
        try:
            self._repo.create_stream(
                stream_id,
                title=title,
                category=category,
                source_path=source_path,
                meta=meta,
            )
            self._refresh_memory()
        except Exception as exc:  # pragma: no cover - disk failures
            self._record_error(exc)

    def end_session(self, stream_id: str) -> None:
        if self._repo is not None and self._stream_id == stream_id:
            try:
                self._extract(stream_id)
                self._repo.finish_stream(
                    stream_id,
                    meta={
                        "messages": self._messages_logged,
                        "events": self._events_logged,
                        "memories": self._memories_saved,
                    },
                )
                self._refresh_memory()
            except Exception as exc:  # pragma: no cover - disk failures
                self._record_error(exc)
        self._stream_id = None

    # -- event ingestion ---------------------------------------------------

    def record_message(self, payload: dict[str, Any]) -> None:
        if self._repo is None or self._stream_id is None:
            return
        kind = str(payload.get("kind", "chat"))
        if kind not in {"chat", "system"}:
            return
        handle = str(payload.get("handle") or "").strip()
        text = str(payload.get("text") or "").strip()
        if not handle or not text:
            return
        release_t = _as_float(payload.get("release_t"), _as_float(payload.get("t"), 0.0))
        scene_t = _as_float(payload.get("t"), release_t)
        live_id = str(payload.get("id") or "")
        parent_live = payload.get("reply_to")
        reply_to = self._live_ids.get(str(parent_live)) if parent_live else None
        try:
            row_id = self._repo.add_message(
                self._stream_id,
                t_playback=release_t,
                handle=handle,
                text=text,
                archetype=str(payload.get("archetype") or "normie"),
                tier=str(payload.get("tier") or "crowd"),
                kind=kind,
                created_at=_as_float(payload.get("created_at"), time.time()),
                reply_to=reply_to,
                mentions=[str(item) for item in payload.get("mentions") or []],
                emotes=[str(item) for item in payload.get("emotes") or []],
                scene_ref={"scene_t": round(scene_t, 3)},
            )
        except Exception as exc:  # pragma: no cover - disk failures
            self._record_error(exc)
            return
        if live_id:
            self._live_ids[live_id] = row_id
            while len(self._live_ids) > 600:
                self._live_ids.popitem(last=False)
        self._messages_logged += 1

    def record_event(self, payload: dict[str, Any]) -> None:
        if self._repo is None or self._stream_id is None:
            return
        event = payload.get("event")
        if not isinstance(event, dict):
            return
        kind = str(event.get("kind") or "").strip()
        if not kind:
            return
        try:
            self._repo.add_event(
                self._stream_id,
                t_playback=_as_float(event.get("t"), 0.0),
                type_=kind,
                payload=event,
            )
        except Exception as exc:  # pragma: no cover - disk failures
            self._record_error(exc)
            return
        self._events_logged += 1

    def checkpoint(self, t_play: float) -> None:
        """Persist memory every few minutes of playback (crash safety)."""
        if self._repo is None or self._stream_id is None:
            return
        interval = max(30.0, float(self.settings.memory_checkpoint_s))
        t = max(0.0, float(t_play))
        if t < self._last_checkpoint_t:
            # A seek rewound the clock; the next interval starts from here.
            self._last_checkpoint_t = t
            return
        if t - self._last_checkpoint_t < interval:
            return
        self._last_checkpoint_t = t
        self._checkpoints += 1
        try:
            self._extract(self._stream_id)
        except Exception as exc:  # pragma: no cover - disk failures
            self._record_error(exc)

    # -- memory reads ------------------------------------------------------

    def refresh_memory(self) -> None:
        """Reload prompt memory from the database (used after a manual write)."""
        self._refresh_memory()

    def memory_for(self, handle: str) -> tuple[str, ...]:
        return self._memory.get(handle, ())

    def memory_candidates(self, handle: str) -> tuple[MemoryCandidate, ...]:
        return self._candidates.get(handle, ())

    def raw_memory(self, handle: str, *, limit: int = 20) -> list[dict[str, object]]:
        if self._repo is None:
            return []
        try:
            rows = self._repo.memory_for(handle, limit=limit)
        except Exception as exc:  # pragma: no cover - disk failures
            self._record_error(exc)
            return []
        return [row.as_dict() for row in rows]

    def all_memory(self, *, per_viewer: int = 6) -> dict[str, list[dict[str, object]]]:
        if self._repo is None:
            return {}
        try:
            loaded = self._repo.load_memory(None, per_viewer=per_viewer)
        except Exception as exc:  # pragma: no cover - disk failures
            self._record_error(exc)
            return {}
        return {
            handle: [row.as_dict() for row in rows]
            for handle, rows in sorted(loaded.items())
        }

    def close(self) -> None:
        if self._repo is not None:
            self._repo.close()
            self._repo = None

    # -- internals ---------------------------------------------------------

    def _extract(self, stream_id: str) -> int:
        repo = self._repo
        if repo is None:
            return 0
        transcript = repo.transcript(stream_id)
        messages = [
            MemoryMessage(
                handle=str(row["handle"]),
                t=float(row["t_playback"]),
                text=str(row["text"]),
                archetype=str(row["archetype"]),
            )
            for row in transcript
            if str(row["tier"]) != "system"
        ]
        events: list[MemoryEvent] = []
        for row in repo.events(stream_id):
            payload = row.get("payload") if isinstance(row, dict) else {}
            payload = payload if isinstance(payload, dict) else {}
            events.append(
                MemoryEvent(
                    t=float(row["t_playback"]),
                    kind=str(row["type"]),
                    intensity=_as_float(payload.get("intensity"), 0.5),
                    detail=str(payload.get("detail") or ""),
                )
            )
        candidates = extract_memories(
            messages,
            events,
            title=self._title,
            lore_handles=self._lore_handles,
            max_per_viewer=self.settings.memory_per_viewer,
            max_total=self.settings.memory_max_total,
        )
        rows = [
            MemoryRow(
                handle=candidate.handle,
                kind=candidate.kind,
                key=candidate.key,
                value=candidate.value,
                weight=candidate.weight,
                stream_id=stream_id,
            )
            for candidate in candidates
        ]
        saved = repo.save_memory(rows)
        repo.prune_memory(per_viewer=self.settings.memory_per_viewer_cap)
        self._memories_saved = saved
        return saved

    def _refresh_memory(self) -> None:
        repo = self._repo
        if repo is None:
            self._memory = {}
            return
        loaded = repo.load_memory(
            self._memory_handles or None,
            per_viewer=self.settings.memory_top_n,
        )
        memory: dict[str, tuple[str, ...]] = {}
        candidates: dict[str, tuple[MemoryCandidate, ...]] = {}
        for handle, rows in loaded.items():
            parsed = tuple(
                MemoryCandidate(
                    handle=row.handle,
                    kind=row.kind,
                    key=row.key,
                    value=row.value,
                    weight=row.weight,
                )
                for row in rows
            )
            candidates[handle] = parsed
            described = prompt_memory(
                list(parsed),
                limit=self.settings.memory_prompt_limit,
            )
            if described:
                memory[handle] = tuple(described)
        self._memory = memory
        self._candidates = candidates

    def _record_error(self, exc: BaseException) -> None:
        message = str(exc)[:400]
        if message != self.last_error:
            logger.warning("persistence error: %s", message)
        self.last_error = message


def _as_float(value: object, default: float) -> float:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


__all__ = ["PersistenceStore"]
