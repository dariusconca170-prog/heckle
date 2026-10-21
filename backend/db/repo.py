"""SQLite persistence for streams, messages, events and persona memory.

The repository is deliberately synchronous and boring. Chat releases at ~2
messages/second, so one INSERT per message is cheap enough; batching would add
a failure mode (losing the last seconds of a stream) without buying anything.

Every write is guarded by one lock and one connection. WAL keeps readers (the
history API) from blocking the dynamics loop, and ``synchronous=NORMAL`` is the
right durability trade for a local simulator. If the database cannot be opened,
the caller disables persistence and chat keeps working from templates — rule 9
in AGENTS.md: degrade, never raise.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


class RepositoryError(RuntimeError):
    """Raised when the database cannot be opened or initialised."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True, frozen=True)
class MemoryRow:
    """One structured persona memory row (PLAN.md §8)."""

    handle: str
    kind: str
    key: str
    value: str
    weight: float = 0.5
    stream_id: str | None = None
    updated_at: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "handle": self.handle,
            "kind": self.kind,
            "key": self.key,
            "value": self.value,
            "weight": round(float(self.weight), 3),
            "stream_id": self.stream_id,
            "updated_at": self.updated_at,
        }


class Repository:
    """One connection, one lock, prepared schema."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._lock = threading.RLock()
            self._conn: sqlite3.Connection | None = sqlite3.connect(
                self.path, check_same_thread=False, timeout=5.0
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA busy_timeout=2000")
            schema = SCHEMA_PATH.read_text(encoding="utf-8")
            with self._lock, self._conn:
                self._conn.executescript(schema)
        except (OSError, sqlite3.Error) as exc:
            self._conn = None
            raise RepositoryError(f"cannot open database {self.path}: {exc}") from exc
        self._viewers: dict[str, int] = {}

    # -- lifecycle ---------------------------------------------------------

    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RepositoryError("database is closed")
        return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> Repository:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- schema introspection ---------------------------------------------

    def table_columns(self, table: str) -> list[str]:
        with self._lock:
            rows = self.connection.execute(f"PRAGMA table_info({table})").fetchall()
        return [str(row["name"]) for row in rows]

    # -- streams -----------------------------------------------------------

    def create_stream(
        self,
        stream_id: str,
        *,
        title: str = "",
        category: str = "",
        source_path: str = "",
        started_at: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        with self._lock, self.connection as conn:
            conn.execute(
                """
                INSERT INTO streams (id, title, category, source_path, started_at, meta_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    category = excluded.category,
                    source_path = excluded.source_path,
                    meta_json = excluded.meta_json
                """,
                (
                    stream_id,
                    title,
                    category,
                    source_path,
                    started_at or utc_now(),
                    json.dumps(meta or {}, ensure_ascii=False),
                ),
            )

    def finish_stream(
        self,
        stream_id: str,
        *,
        ended_at: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        stamp = ended_at or utc_now()
        with self._lock, self.connection as conn:
            if meta is None:
                conn.execute(
                    "UPDATE streams SET ended_at = ? WHERE id = ?", (stamp, stream_id)
                )
            else:
                conn.execute(
                    """
                    UPDATE streams
                       SET ended_at = ?,
                           meta_json = json_patch(
                               CASE WHEN json_valid(meta_json) THEN meta_json ELSE '{}' END,
                               ?
                           )
                     WHERE id = ?
                    """,
                    (stamp, json.dumps(meta, ensure_ascii=False), stream_id),
                )

    def stream(self, stream_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.connection.execute(
                """
                SELECT s.*,
                       (SELECT COUNT(*) FROM messages m WHERE m.stream_id = s.id)
                           AS message_count
                  FROM streams s
                 WHERE s.id = ?
                """,
                (stream_id,),
            ).fetchone()
        return _stream_dict(row) if row is not None else None

    def streams(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT s.*,
                       (SELECT COUNT(*) FROM messages m WHERE m.stream_id = s.id)
                           AS message_count
                  FROM streams s
                 ORDER BY s.started_at DESC
                 LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        return [_stream_dict(row) for row in rows]

    # -- viewers -----------------------------------------------------------

    def ensure_viewer(
        self,
        handle: str,
        *,
        archetype: str = "normie",
        chattiness: float = 1.0,
        style: dict[str, Any] | None = None,
    ) -> int:
        cached = self._viewers.get(handle)
        if cached is not None:
            return cached
        with self._lock, self.connection as conn:
            conn.execute(
                """
                INSERT INTO viewers (handle, archetype, created_at, chattiness, style_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(handle) DO NOTHING
                """,
                (
                    handle,
                    archetype,
                    utc_now(),
                    float(chattiness),
                    json.dumps(style or {}, ensure_ascii=False),
                ),
            )
            row = conn.execute("SELECT id FROM viewers WHERE handle = ?", (handle,)).fetchone()
        if row is None:  # pragma: no cover - would mean a rolled-back insert
            raise RepositoryError(f"viewer {handle!r} could not be stored")
        viewer_id = int(row["id"])
        self._viewers[handle] = viewer_id
        return viewer_id

    def viewer(self, handle: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM viewers WHERE handle = ?", (handle,)
            ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["style_json"] = _json_or(data.get("style_json"), {})
        return data

    def viewer_count(self) -> int:
        with self._lock:
            row = self.connection.execute("SELECT COUNT(*) AS n FROM viewers").fetchone()
        return int(row["n"]) if row is not None else 0

    # -- messages and events ----------------------------------------------

    def add_message(
        self,
        stream_id: str,
        *,
        t_playback: float,
        handle: str,
        text: str,
        archetype: str = "normie",
        tier: str = "crowd",
        kind: str = "chat",
        created_at: float,
        reply_to: int | None = None,
        mentions: Sequence[str] | None = None,
        emotes: Sequence[str] | None = None,
        scene_ref: dict[str, Any] | None = None,
    ) -> int:
        author_id = self.ensure_viewer(handle, archetype=archetype)
        with self._lock, self.connection as conn:
            cursor = conn.execute(
                """
                INSERT INTO messages (
                    stream_id, t_playback, author_id, tier, text, reply_to,
                    mentions_json, emotes_json, kind, scene_ref_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stream_id,
                    float(t_playback),
                    author_id,
                    tier,
                    text,
                    reply_to,
                    json.dumps(list(mentions or []), ensure_ascii=False),
                    json.dumps(list(emotes or []), ensure_ascii=False),
                    kind,
                    json.dumps(scene_ref, ensure_ascii=False) if scene_ref else None,
                    float(created_at),
                ),
            )
        return int(cursor.lastrowid or 0)

    def messages(
        self,
        stream_id: str,
        *,
        limit: int = 200,
        offset: int = 0,
        newest_first: bool = False,
    ) -> list[dict[str, Any]]:
        order = "DESC" if newest_first else "ASC"
        with self._lock:
            rows = self.connection.execute(
                f"""
                SELECT m.id, m.stream_id, m.t_playback, m.text, m.tier, m.kind,
                       m.reply_to, m.mentions_json, m.emotes_json,
                       m.scene_ref_json, m.created_at,
                       v.handle, v.archetype,
                       pv.handle AS reply_handle, p.text AS reply_text
                  FROM messages m
                  JOIN viewers v ON v.id = m.author_id
                  LEFT JOIN messages p ON p.id = m.reply_to
                  LEFT JOIN viewers pv ON pv.id = p.author_id
                 WHERE m.stream_id = ?
                 ORDER BY m.t_playback {order}, m.id {order}
                 LIMIT ? OFFSET ?
                """,
                (stream_id, max(1, int(limit)), max(0, int(offset))),
            ).fetchall()
        return [_message_dict(row) for row in rows]

    def message_count(self, stream_id: str) -> int:
        with self._lock:
            row = self.connection.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE stream_id = ?", (stream_id,)
            ).fetchone()
        return int(row["n"]) if row is not None else 0

    def transcript(self, stream_id: str) -> list[dict[str, Any]]:
        """Compact rows used for memory extraction (no UI fields)."""
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT m.t_playback, m.text, m.tier, v.handle, v.archetype
                  FROM messages m
                  JOIN viewers v ON v.id = m.author_id
                 WHERE m.stream_id = ?
                 ORDER BY m.t_playback, m.id
                """,
                (stream_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def add_event(
        self,
        stream_id: str,
        *,
        t_playback: float,
        type_: str,
        payload: dict[str, Any] | None = None,
    ) -> int:
        with self._lock, self.connection as conn:
            cursor = conn.execute(
                """
                INSERT INTO events (stream_id, t_playback, type, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    stream_id,
                    float(t_playback),
                    type_,
                    json.dumps(payload or {}, ensure_ascii=False),
                ),
            )
        return int(cursor.lastrowid or 0)

    def events(self, stream_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        query = (
            "SELECT id, t_playback, type, payload_json FROM events "
            "WHERE stream_id = ? ORDER BY t_playback, id"
        )
        params: list[Any] = [stream_id]
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(1, int(limit)))
        with self._lock:
            rows = self.connection.execute(query, params).fetchall()
        return [
            {
                "id": int(row["id"]),
                "t_playback": float(row["t_playback"]),
                "type": str(row["type"]),
                "payload": _json_or(row["payload_json"], {}),
            }
            for row in rows
        ]

    # -- polls -------------------------------------------------------------

    def create_poll(
        self,
        stream_id: str,
        *,
        question: str,
        options: Sequence[str],
        opened_at: float,
        closes_at: float,
        allow_multiple: bool = False,
    ) -> int:
        payload = {
            "allow_multiple": bool(allow_multiple),
            "labels": list(options),
        }
        with self._lock, self.connection as conn:
            cursor = conn.execute(
                """
                INSERT INTO polls (stream_id, question, options_json, opened_at, closes_at, status)
                VALUES (?, ?, ?, ?, ?, 'open')
                """,
                (
                    stream_id,
                    question,
                    json.dumps(payload, ensure_ascii=False),
                    float(opened_at),
                    float(closes_at),
                ),
            )
        return int(cursor.lastrowid or 0)

    def close_poll(self, poll_id: int, *, status: str = "closed") -> None:
        with self._lock, self.connection as conn:
            conn.execute("UPDATE polls SET status = ? WHERE id = ?", (status, int(poll_id)))

    def add_poll_vote(
        self,
        poll_id: int,
        *,
        handle: str,
        option_index: int,
        t_playback: float,
    ) -> None:
        viewer_id = self.ensure_viewer(handle)
        with self._lock, self.connection as conn:
            conn.execute(
                """
                INSERT INTO poll_votes (poll_id, viewer_id, option_index, t_playback)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(poll_id, viewer_id) DO NOTHING
                """,
                (int(poll_id), viewer_id, int(option_index), float(t_playback)),
            )

    def poll_tally(self, poll_id: int) -> list[int]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT option_index, COUNT(*) AS n
                  FROM poll_votes
                 WHERE poll_id = ?
                 GROUP BY option_index
                 ORDER BY option_index
                """,
                (int(poll_id),),
            ).fetchall()
        counts: dict[int, int] = {int(row["option_index"]): int(row["n"]) for row in rows}
        width = (max(counts) + 1) if counts else 0
        return [counts.get(index, 0) for index in range(width)]

    def polls(self, stream_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT id, stream_id, question, options_json, opened_at, closes_at, status
                  FROM polls
                 WHERE stream_id = ?
                 ORDER BY opened_at DESC
                 LIMIT ?
                """,
                (stream_id, max(1, int(limit))),
            ).fetchall()
        return [self._poll_dict(row) for row in rows]

    def poll(self, poll_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self.connection.execute(
                """
                SELECT id, stream_id, question, options_json, opened_at, closes_at, status
                  FROM polls WHERE id = ?
                """,
                (int(poll_id),),
            ).fetchone()
        return self._poll_dict(row) if row is not None else None

    def _poll_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        raw = _json_or(row["options_json"], {})
        if isinstance(raw, dict):
            options = [str(item) for item in raw.get("labels", [])]
            allow_multiple = bool(raw.get("allow_multiple", False))
        else:  # legacy/plain arrays
            options = [str(item) for item in raw]
            allow_multiple = False
        counts = self.poll_tally(int(row["id"]))
        counts = counts + [0] * max(0, len(options) - len(counts))
        total = sum(counts)
        return {
            "id": int(row["id"]),
            "stream_id": str(row["stream_id"]),
            "question": str(row["question"]),
            "options": options,
            "counts": counts,
            "percentages": [round((c / total) * 100, 1) if total else 0.0 for c in counts],
            "total_votes": total,
            "opened_at": round(float(row["opened_at"]), 3),
            "closes_at": round(float(row["closes_at"]), 3) if row["closes_at"] is not None else None,
            "allow_multiple": allow_multiple,
            "status": str(row["status"]),
        }

    # -- clips -------------------------------------------------------------

    def add_clip(
        self,
        stream_id: str,
        *,
        start_t: float,
        end_t: float,
        path: str = "",
        thumb_path: str | None = None,
        reason: str = "",
        score: float = 0.0,
    ) -> int:
        with self._lock, self.connection as conn:
            cursor = conn.execute(
                """
                INSERT INTO clips (stream_id, start_t, end_t, path, thumb_path, reason, score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stream_id,
                    float(start_t),
                    float(end_t),
                    path,
                    thumb_path,
                    reason,
                    float(score),
                ),
            )
        return int(cursor.lastrowid or 0)

    def update_clip_export(self, clip_id: int, *, path: str, thumb_path: str | None) -> None:
        with self._lock, self.connection as conn:
            conn.execute(
                "UPDATE clips SET path = ?, thumb_path = ? WHERE id = ?",
                (path, thumb_path, int(clip_id)),
            )

    def clips(self, stream_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT id, stream_id, start_t, end_t, path, thumb_path, reason, score
                  FROM clips
                 WHERE stream_id = ?
                 ORDER BY score DESC, start_t DESC
                 LIMIT ?
                """,
                (stream_id, max(1, int(limit))),
            ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "stream_id": str(row["stream_id"]),
                "start_t": round(float(row["start_t"]), 3),
                "end_t": round(float(row["end_t"]), 3),
                "path": str(row["path"] or ""),
                "thumb_path": str(row["thumb_path"] or ""),
                "reason": str(row["reason"] or ""),
                "score": round(float(row["score"]), 3),
            }
            for row in rows
        ]

    # -- memory ------------------------------------------------------------

    def save_memory(self, rows: Iterable[MemoryRow], *, now: str | None = None) -> int:
        stamp = now or utc_now()
        saved = 0
        with self._lock, self.connection as conn:
            for row in rows:
                if not row.handle or not row.value:
                    continue
                viewer_id = self.ensure_viewer(row.handle)
                # Keep the strongest evidence for a memory; a weak later
                # observation must not erase a strong earlier one.
                conn.execute(
                    """
                    INSERT INTO viewer_memory (
                        viewer_id, stream_id, kind, key, value, weight, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(viewer_id, kind, key) DO UPDATE SET
                        value = excluded.value,
                        weight = MAX(viewer_memory.weight, excluded.weight),
                        stream_id = excluded.stream_id,
                        updated_at = excluded.updated_at
                    """,
                    (
                        viewer_id,
                        row.stream_id,
                        row.kind,
                        row.key,
                        row.value,
                        float(row.weight),
                        stamp,
                    ),
                )
                saved += 1
        return saved

    def load_memory(
        self,
        handles: Sequence[str] | None = None,
        *,
        per_viewer: int = 6,
    ) -> dict[str, list[MemoryRow]]:
        params: list[Any] = []
        where = ""
        if handles is not None:
            unique = list(dict.fromkeys(handles))
            if not unique:
                return {}
            placeholders = ", ".join("?" for _ in unique)
            where = f"WHERE v.handle IN ({placeholders})"
            params.extend(unique)
        with self._lock:
            rows = self.connection.execute(
                f"""
                SELECT v.handle, m.stream_id, m.kind, m.key, m.value, m.weight,
                       m.updated_at
                  FROM viewer_memory m
                  JOIN viewers v ON v.id = m.viewer_id
                 {where}
                 ORDER BY m.weight DESC, m.updated_at DESC
                """,
                params,
            ).fetchall()
        limit = max(1, int(per_viewer))
        grouped: dict[str, list[MemoryRow]] = {}
        for row in rows:
            bucket = grouped.setdefault(str(row["handle"]), [])
            if len(bucket) >= limit:
                continue
            bucket.append(
                MemoryRow(
                    handle=str(row["handle"]),
                    kind=str(row["kind"]),
                    key=str(row["key"]),
                    value=str(row["value"]),
                    weight=float(row["weight"]),
                    stream_id=row["stream_id"],
                    updated_at=str(row["updated_at"]),
                )
            )
        return grouped

    def memory_for(self, handle: str, *, limit: int = 6) -> list[MemoryRow]:
        return self.load_memory([handle], per_viewer=limit).get(handle, [])

    def prune_memory(self, *, per_viewer: int = 500) -> int:
        with self._lock, self.connection as conn:
            cursor = conn.execute(
                """
                DELETE FROM viewer_memory
                 WHERE rowid IN (
                    SELECT rowid FROM (
                        SELECT rowid,
                               ROW_NUMBER() OVER (
                                   PARTITION BY viewer_id
                                   ORDER BY weight DESC, updated_at DESC
                               ) AS rank
                          FROM viewer_memory
                    )
                    WHERE rank > ?
                 )
                """,
                (max(1, int(per_viewer)),),
            )
        return int(cursor.rowcount or 0)

    # -- diagnostics -------------------------------------------------------

    def stats(self) -> dict[str, int]:
        with self._lock:
            row = self.connection.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM streams) AS streams,
                    (SELECT COUNT(*) FROM messages) AS messages,
                    (SELECT COUNT(*) FROM events) AS events,
                    (SELECT COUNT(*) FROM viewers) AS viewers,
                    (SELECT COUNT(*) FROM viewer_memory) AS memories
                """
            ).fetchone()
        if row is None:
            return {}
        return {
            name: int(row[name])
            for name in ("streams", "messages", "events", "viewers", "memories")
        }


def _json_or(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _stream_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "title": str(row["title"]),
        "category": str(row["category"]),
        "source_path": str(row["source_path"]),
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
        "meta": _json_or(row["meta_json"], {}),
        "message_count": int(row["message_count"]),
    }


def _message_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "stream_id": str(row["stream_id"]),
        "t_playback": round(float(row["t_playback"]), 3),
        "handle": str(row["handle"]),
        "archetype": str(row["archetype"]),
        "tier": str(row["tier"]),
        "kind": str(row["kind"]),
        "text": str(row["text"]),
        "reply_to": int(row["reply_to"]) if row["reply_to"] is not None else None,
        "reply_handle": row["reply_handle"],
        "reply_text": row["reply_text"],
        "mentions": _json_or(row["mentions_json"], []),
        "emotes": _json_or(row["emotes_json"], []),
        "scene_ref": _json_or(row["scene_ref_json"], None),
        "created_at": float(row["created_at"]),
    }


__all__ = ["MemoryRow", "Repository", "RepositoryError", "utc_now"]
