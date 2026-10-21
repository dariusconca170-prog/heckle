"""WebSocket: browser clock reports in, typed bus events out."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .. import events
from ..state import AppState

router = APIRouter()


def _state(websocket: WebSocket) -> AppState:
    return websocket.app.state.app_state


async def _send(websocket: WebSocket, payload: dict[str, Any], lock: asyncio.Lock) -> None:
    async with lock:
        await websocket.send_json(payload)


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    state = _state(websocket)
    await websocket.accept()
    send_lock = asyncio.Lock()
    subscription = state.bus.subscribe(
        [events.CLOCK_UPDATE, events.CLOCK_SEEK, events.CLOCK_PAUSE, events.CLOCK_PLAY,
         events.SESSION_STARTED, events.SESSION_ENDED, events.SCENE_UPDATE,
         events.SCENE_EVENT, events.TOPIC_SHIFT, events.CHAT_MESSAGE,
         events.CHAT_BUFFER_LEVEL, events.CHAT_CATCHING_UP, events.SYSTEM_STATUS,
         events.STREAM_EVENT, events.POLL_OPENED, events.POLL_VOTE,
         events.POLL_CLOSED, events.POLL_RESULTS, events.CLIP_CANDIDATE,
         events.CLIP_CREATED, events.MIC_SPEECH],
        maxsize=256,
    )

    async def pump() -> None:
        try:
            while True:
                event = await subscription.get()
                if event is None:
                    return
                await _send(
                    websocket,
                    {
                        "type": "event",
                        "event_type": event.type,
                        "payload": event.payload,
                        "ts": event.ts,
                        "seq": event.seq,
                    },
                    send_lock,
                )
        except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
            return

    pump_task = asyncio.create_task(pump(), name="ws-event-pump")

    await _send(
        websocket,
        {
            "type": "hello",
            "config": {
                "app_name": state.settings.app_name,
                "version": state.settings.version,
                "lookahead_s": state.lookahead_s,
                "time_update_hz": state.settings.clock_broadcast_hz,
            },
            "session": state.session.as_dict() if state.session else None,
            "clock": state.clock_snapshot().as_dict(),
            "scene": state.timeline.snapshot(limit=20),
            "perception": (
                websocket.app.state.perception.metrics()
                if getattr(websocket.app.state, "perception", None) is not None
                else {"status": "unavailable"}
            ),
            "dynamics": (
                websocket.app.state.dynamics.metrics()
                if getattr(websocket.app.state, "dynamics", None) is not None
                else {"status": "unavailable"}
            ),
            "polls": (
                websocket.app.state.polls.metrics()
                if getattr(websocket.app.state, "polls", None) is not None
                else {"status": "unavailable"}
            ),
            "stream": (
                websocket.app.state.stream_events.metrics()
                if getattr(websocket.app.state, "stream_events", None) is not None
                else {"status": "unavailable"}
            ),
        },
        send_lock,
    )

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await _send(websocket, {"type": "error", "message": "invalid JSON"}, send_lock)
                continue
            if not isinstance(message, dict):
                await _send(websocket, {"type": "error", "message": "message must be an object"}, send_lock)
                continue
            message_type = message.get("type")
            if message_type == "ping":
                await _send(websocket, {"type": "pong", "t": message.get("t")}, send_lock)
            elif message_type == "time":
                await _handle_time(state, websocket, message, send_lock)
            else:
                await _send(
                    websocket,
                    {"type": "error", "message": f"unknown message type: {message_type!r}"},
                    send_lock,
                )
    except WebSocketDisconnect:
        pass
    except (RuntimeError, asyncio.CancelledError):
        pass
    finally:
        pump_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await pump_task
        state.bus.unsubscribe(subscription)


async def _handle_time(
    state: AppState,
    websocket: WebSocket,
    message: dict[str, Any],
    send_lock: asyncio.Lock,
) -> None:
    if state.session is None:
        await _send(
            websocket,
            {"type": "error", "message": "start a session before sending clock updates"},
            send_lock,
        )
        return
    try:
        t_play = float(message.get("t_play"))
    except (TypeError, ValueError):
        await _send(websocket, {"type": "error", "message": "t_play must be a number"}, send_lock)
        return
    if not math.isfinite(t_play):
        await _send(websocket, {"type": "error", "message": "t_play must be finite"}, send_lock)
        return

    clock = state.session.clock
    duration = message.get("duration")
    if duration is not None:
        with contextlib.suppress(TypeError, ValueError):
            clock.set_duration(float(duration))

    playing = message.get("playing")
    rate = message.get("rate")
    update = clock.update(
        max(0.0, t_play),
        playing=bool(playing) if playing is not None else None,
        rate=float(rate) if isinstance(rate, (int, float)) else None,
        seek=bool(message.get("seek", False)),
    )
    snapshot = update.snapshot.as_dict()
    payload = {
        **snapshot,
        "seek": update.seek,
        "discontinuity": update.discontinuity,
        "session_id": state.session.id,
    }
    # Every report is published for the release queue; discontinuities get a
    # dedicated event so buffers can be flushed (PLAN.md §9.3).
    state.bus.publish_nowait(events.CLOCK_UPDATE, payload)
    if update.seek:
        state.bus.publish_nowait(events.CLOCK_SEEK, payload)
    if update.paused:
        state.bus.publish_nowait(events.CLOCK_PAUSE, payload)
    if update.resumed:
        state.bus.publish_nowait(events.CLOCK_PLAY, payload)
