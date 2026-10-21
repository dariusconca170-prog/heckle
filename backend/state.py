"""Shared, app-scoped runtime state (no module-level globals)."""

from __future__ import annotations

import json
import secrets
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .chatter.memory import MemoryCandidate
from .clock import ClockSnapshot, PlaybackClock
from .config import Settings
from .events import SESSION_ENDED, SESSION_STARTED, EventBus
from .media import MediaItem, MediaLibrary, MediaTokenStore
from .perception.timeline import SceneTimeline
from .persistence.store import PersistenceStore
from .runtime import RuntimeSettings


@dataclass(slots=True)
class SessionRecord:
    id: str
    item: MediaItem
    token: str
    title: str
    category: str
    viewer_count: int
    clock: PlaybackClock
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title or self.item.name,
            "category": self.category,
            "viewer_count": self.viewer_count,
            "media": self.item.as_dict(),
            "media_url": f"/api/media/{self.token}",
            "started_at": self.started_at,
            "lookahead_s": self.clock.lookahead_s,
            "duration": self.clock.duration,
        }


class AppState:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.bus = EventBus()
        self.library = MediaLibrary(settings.media_dir, settings.media_extensions)
        self.tokens = MediaTokenStore()
        # The selected profile owns the real look-ahead. config/default.toml is
        # only a fallback for a fresh checkout without a model selection yet.
        self.lookahead_s = _selected_lookahead(settings.model_profile) or settings.lookahead_s
        self.runtime = RuntimeSettings(settings.data_dir / "settings.json")
        if "mic_enabled" not in self.runtime.saved_keys:
            self.runtime.values["mic_enabled"] = bool(settings.mic_enabled)
        if "lookahead_s" in self.runtime.saved_keys:
            self.lookahead_s = float(self.runtime.values["lookahead_s"])
        if "chat_target_buffer" in self.runtime.saved_keys:
            settings.chat_target_buffer = int(self.runtime.values["chat_target_buffer"])
        self.clock = PlaybackClock(
            lookahead_s=self.lookahead_s,
            seek_threshold_s=settings.seek_threshold_s,
        )
        self.timeline = SceneTimeline(window_s=90.0)
        self.persistence = PersistenceStore(settings)
        self.session: SessionRecord | None = None
        # Cross-service pressure signals used for cooperative scheduling.
        # They are plain values, not a second bus: the dynamics loop writes
        # them and perception reads them before starting a GPU call.
        self.chat_buffer_level = 20
        self.chat_generating = False
        # Feature -> dynamics cues. Polls, stream events, clips and speech push
        # small dicts here; the dynamics loop is the only consumer, so the
        # LLM is still never in charge of timing.
        self.chat_cues: deque[dict[str, object]] = deque(maxlen=64)
        self.live_viewers = 0
        self.latest_speech: dict[str, object] | None = None
        self.polls: object | None = None
        self.stream_events: object | None = None
        self.clips: object | None = None
        self.mic: object | None = None
        self.screen: object | None = None

    @classmethod
    def build(cls, settings: Settings) -> AppState:
        settings.ensure_dirs()
        return cls(settings)

    def list_media(self) -> list[MediaItem]:
        return self.library.list()

    def start_session(
        self,
        *,
        media_id: str,
        title: str = "",
        category: str = "",
        viewer_count: int = 1200,
    ) -> SessionRecord:
        item = self.library.resolve(media_id)
        if self.session is not None:
            self.end_session()
        token = self.tokens.issue(item)
        clock = PlaybackClock(
            lookahead_s=self.lookahead_s,
            seek_threshold_s=self.settings.seek_threshold_s,
        )
        record = SessionRecord(
            id=secrets.token_hex(8),
            item=item,
            token=token,
            title=title.strip() or item.name,
            category=category.strip(),
            viewer_count=max(0, int(viewer_count)),
            clock=clock,
        )
        self.clock = clock
        self.session = record
        self.timeline.reset()
        self.chat_cues.clear()
        self.live_viewers = record.viewer_count
        self.latest_speech = None
        # Create the stream row and load the previous streams' persona memory
        # before anything publishes SESSION_STARTED, so the first chat batch of
        # the second stream already knows the first one happened.
        self.persistence.begin_session(
            record.id,
            title=record.title,
            category=record.category,
            source_path=str(item.path),
            meta={
                "media_id": item.name,
                "viewer_count": record.viewer_count,
                "lookahead_s": self.lookahead_s,
            },
        )
        self.bus.publish_nowait(
            SESSION_STARTED,
            {
                "session": record.as_dict(),
                "clock": clock.snapshot().as_dict(),
            },
        )
        return record

    def end_session(self) -> SessionRecord | None:
        record = self.session
        if record is None:
            return None
        # Finalise and extract memory before announcing the end: a caller that
        # starts the next stream immediately must see this stream's memory.
        self.persistence.end_session(record.id)
        self.bus.publish_nowait(
            SESSION_ENDED,
            {"session_id": record.id, "clock": record.clock.snapshot().as_dict()},
        )
        self.tokens.revoke(record.token)
        record.clock.reset()
        self.chat_cues.clear()
        self.live_viewers = 0
        self.latest_speech = None
        self.session = None
        self.clock = record.clock
        return record

    def clock_snapshot(self) -> ClockSnapshot:
        return (self.session.clock if self.session else self.clock).snapshot()

    def register_roster(self, personas: object) -> None:
        """Tell persistence which handles can carry memory into a prompt.

        Core personas have stable handles across streams; crowd handles are
        generated once at startup. Only core viewers are loaded into prompts,
        so memory does not grow the prompt budget for the crowd tier.
        """
        handles: list[str] = []
        lore: list[str] = []
        for persona in personas:  # type: ignore[attr-defined]
            handle = getattr(persona, "handle", None)
            if not isinstance(handle, str) or not handle:
                continue
            handles.append(handle)
            if getattr(persona, "archetype", "") == "lore-keeper":
                lore.append(handle)
        self.persistence.set_roster(tuple(handles), tuple(lore))

    def memory_for(self, handle: str) -> tuple[str, ...]:
        if not self.settings.memory_enabled:
            return ()
        return self.persistence.memory_for(handle)

    def memory_candidates(self, handle: str) -> tuple[MemoryCandidate, ...]:
        if not self.settings.memory_enabled:
            return ()
        return self.persistence.memory_candidates(handle)

    def close(self) -> None:
        self.end_session()
        self.bus.close()
        self.persistence.close()


def _selected_lookahead(path: Path) -> float | None:
    """Read look-ahead from data/model_profile.json when it is usable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    candidates = [data.get("lookahead_s")]
    profile = data.get("profile")
    if isinstance(profile, dict):
        candidates.append(profile.get("lookahead_s"))
    for value in candidates:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
            if 0.0 <= value <= 30.0:
                return value
    return None
