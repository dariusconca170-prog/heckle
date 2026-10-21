"""Durable side of the app: subscribe to the bus and write what happened.

This service never decides anything. It drains ``CHAT_MESSAGE`` and
``SCENE_EVENT`` events into SQLite and asks the store for a memory checkpoint
on a playback-time interval. If the database is unavailable the events are
dropped — chat is not allowed to slow down because the disk is busy (rule 9).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Any

from .. import events

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..state import AppState

logger = logging.getLogger(__name__)


class PersistenceService:
    def __init__(self, state: AppState) -> None:
        self.state = state
        self.store = state.persistence
        self._task: asyncio.Task[None] | None = None
        self._subscription: events.Subscription | None = None
        self._stopping = False
        self._written = 0
        self._status = "idle"

    def metrics(self) -> dict[str, Any]:
        return {
            "status": self._status,
            "written": self._written,
            **self.store.metrics(),
        }

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stopping = False
        self._subscription = self.state.bus.subscribe(
            [events.CHAT_MESSAGE, events.SCENE_EVENT, events.STREAM_EVENT], maxsize=2048
        )
        self._status = "live" if self.store.enabled else "disabled"
        self._task = asyncio.create_task(self._run(), name="persistence-writer")

    async def stop(self) -> None:
        self._stopping = True
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        if self._subscription is not None:
            self.state.bus.unsubscribe(self._subscription)
            self._subscription = None
        self._status = "stopped"

    async def _run(self) -> None:
        subscription = self._subscription
        if subscription is None:
            return
        while not self._stopping:
            try:
                drained = 0
                while not self._stopping:
                    event = subscription.get_nowait()
                    if event is None:
                        break
                    self._handle(event)
                    drained += 1
                    if drained >= 256:
                        break
                await asyncio.sleep(0.05 if drained else 0.2)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - last-resort guard
                logger.warning("persistence loop error: %s", exc)
                await asyncio.sleep(1.0)

    def _handle(self, event: events.Event) -> None:
        if event.type == events.CHAT_MESSAGE:
            self.store.record_message(event.payload)
        elif event.type == events.SCENE_EVENT:
            self.store.record_event(event.payload)
        elif event.type == events.STREAM_EVENT:
            inner = event.payload.get("event")
            if not isinstance(inner, dict) or inner.get("kind") not in {
                "sub",
                "gift",
                "raid",
                "follow",
            }:
                return
            self.store.record_event({"event": inner})
        else:
            return
        self._written += 1
        session = self.state.session
        if session is not None:
            self.store.checkpoint(session.clock.snapshot().t_play)


__all__ = ["PersistenceService"]
