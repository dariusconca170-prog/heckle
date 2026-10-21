"""HTTP API for stream furniture: polls and synthetic stream events (M6)."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..features.polls import MAX_OPTION_CHARS, MAX_OPTIONS, MAX_QUESTION_CHARS, PollError
from ..features.stream_events import EVENT_KINDS
from ..runtime import RuntimeSettings, SettingsPatch
from ..state import AppState

router = APIRouter(prefix="/api")


class PollRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    options: list[str] = Field(min_length=2, max_length=MAX_OPTIONS)
    duration_s: float = Field(default=60.0, ge=5.0, le=3600.0)
    allow_multiple: bool = False


class VoteRequest(BaseModel):
    handle: str = Field(default="you", min_length=1, max_length=80)
    option_index: int = Field(ge=0, le=MAX_OPTIONS - 1)


class SettingsRequest(BaseModel):
    preset: str | None = Field(default=None, max_length=24)
    content_mode: str | None = Field(default=None, max_length=16)
    intensity: float | None = Field(default=None, ge=0.2, le=3.0)
    chat_target_buffer: int | None = Field(default=None, ge=4, le=60)
    lookahead_s: float | None = Field(default=None, ge=0.0, le=30.0)
    model_path: str | None = Field(default=None, max_length=500)
    model_quant: str | None = Field(default=None, max_length=24)
    tts_enabled: bool | None = None
    mic_enabled: bool | None = None


class TranscriptRequest(BaseModel):
    text: str = Field(min_length=1, max_length=300)
    energy: float = Field(default=0.8, ge=0.0, le=1.0)


class ManualClipRequest(BaseModel):
    start_t: float | None = Field(default=None, ge=0.0)
    end_t: float | None = Field(default=None, ge=0.0)
    reason: str = Field(default="", max_length=280)


def _state(request: Request) -> AppState:
    return request.app.state.app_state


def _polls(request: Request) -> Any:
    service = getattr(request.app.state, "polls", None)
    if service is None:
        raise HTTPException(status_code=503, detail="poll service is not available")
    return service


@router.post("/polls", status_code=status.HTTP_201_CREATED)
def create_poll(request: Request, body: PollRequest) -> dict[str, object]:
    state = _state(request)
    service = _polls(request)
    if state.session is None:
        raise HTTPException(status_code=409, detail="start a stream before creating a poll")
    try:
        poll = service.open(
            body.question,
            [option[:MAX_OPTION_CHARS] for option in body.options],
            duration_s=body.duration_s,
            allow_multiple=body.allow_multiple,
        )
    except PollError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"poll": poll.as_dict(t_play=state.clock_snapshot().t_play)}


@router.get("/polls")
def get_polls(request: Request, limit: int = 20) -> dict[str, object]:
    state = _state(request)
    service = getattr(request.app.state, "polls", None)
    if service is None:
        return {"current": None, "history": [], "enabled": False}
    current = service.snapshot()
    return {
        "enabled": True,
        "current": current,
        "history": service.history(limit=min(max(1, limit), 100)),
        "metrics": service.metrics(),
        "clock": state.clock_snapshot().as_dict(),
    }


@router.post("/polls/close")
def close_poll(request: Request) -> dict[str, object]:
    state = _state(request)
    service = _polls(request)
    poll = service.close(reason="manual")
    if poll is None:
        raise HTTPException(status_code=409, detail="no open poll")
    return {"poll": poll.as_dict(t_play=state.clock_snapshot().t_play)}


@router.post("/polls/{poll_id}/vote")
def vote(request: Request, poll_id: int, body: VoteRequest) -> dict[str, object]:
    service = _polls(request)
    current = service.current()
    if current is None or current.id != poll_id:
        raise HTTPException(status_code=404, detail="unknown poll")
    try:
        poll = service.add_vote(body.handle, body.option_index)
    except PollError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"poll": poll}


@router.post("/stream-events/{kind}", status_code=status.HTTP_201_CREATED)
def trigger_stream_event(request: Request, kind: str) -> dict[str, object]:
    """Fire a sub/raid/follow immediately (acceptance + debugging)."""
    if kind not in EVENT_KINDS:
        raise HTTPException(status_code=422, detail=f"kind must be one of {EVENT_KINDS}")
    state = _state(request)
    service = getattr(request.app.state, "stream_events", None)
    if service is None:
        raise HTTPException(status_code=503, detail="stream event service is not available")
    if state.session is None:
        raise HTTPException(status_code=409, detail="start a stream first")
    moment = service.trigger(kind)
    if moment is None:
        raise HTTPException(status_code=409, detail="could not trigger event")
    return {
        "event": {
            "kind": moment.kind,
            "handle": moment.handle,
            "amount": moment.amount,
            "text": moment.text,
            "t": round(moment.t, 3),
        },
        "viewer_count": service.metrics()["live_viewers"],
    }


@router.get("/clips")
def get_clips(request: Request, limit: int = 50) -> dict[str, object]:
    service = getattr(request.app.state, "clips", None)
    if service is None:
        return {"clips": [], "enabled": False}
    capped = min(max(1, limit), 200)
    return {
        "enabled": True,
        "clips": service.clips(limit=capped),
        "metrics": service.metrics(),
    }


@router.post("/clips", status_code=status.HTTP_201_CREATED)
def create_clip(request: Request, body: ManualClipRequest) -> dict[str, object]:
    state = _state(request)
    service = getattr(request.app.state, "clips", None)
    if service is None:
        raise HTTPException(status_code=503, detail="clip service is not available")
    if state.session is None:
        raise HTTPException(status_code=409, detail="start a stream first")
    now = float(state.clock_snapshot().t_play)
    start = body.start_t if body.start_t is not None else max(0.0, now - 15.0)
    end = body.end_t if body.end_t is not None else now + 5.0
    if end <= start:
        raise HTTPException(status_code=422, detail="end_t must be after start_t")
    try:
        candidate = service.manual_clip(start, end, reason=body.reason)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"clip": candidate.as_dict()}


@router.get("/clips/{clip_id}/file")
def clip_file(request: Request, clip_id: int) -> FileResponse:
    return _clip_asset(request, clip_id, thumb=False)


@router.get("/clips/{clip_id}/thumb")
def clip_thumb(request: Request, clip_id: int) -> FileResponse:
    return _clip_asset(request, clip_id, thumb=True)


def _clip_asset(request: Request, clip_id: int, *, thumb: bool) -> FileResponse:
    service = getattr(request.app.state, "clips", None)
    if service is None:
        raise HTTPException(status_code=503, detail="clip service is not available")
    clip = next(
        (item for item in service.clips(limit=200) if int(item.get("id") or 0) == clip_id),
        None,
    )
    if clip is None:
        raise HTTPException(status_code=404, detail="unknown clip")
    relative = clip.get("thumb_path" if thumb else "path") or ""
    if not relative:
        raise HTTPException(status_code=409, detail="clip is still exporting")
    path = service.resolve(str(relative))
    if path is None:
        raise HTTPException(status_code=404, detail="clip file is missing")
    return FileResponse(
        path,
        media_type="image/jpeg" if thumb else "video/mp4",
        filename=path.name,
    )


@router.get("/screen")
def get_screen(request: Request) -> dict[str, object]:
    service = getattr(request.app.state, "screen", None)
    if service is None:
        return {"enabled": False, "status": "unavailable"}
    return service.metrics()


@router.post("/screen/start")
async def start_screen(request: Request) -> dict[str, object]:
    state = _state(request)
    service = getattr(request.app.state, "screen", None)
    if service is None:
        raise HTTPException(status_code=503, detail="screen service is not available")
    if state.session is None:
        raise HTTPException(status_code=409, detail="start a stream before sharing the screen")
    await service.start()
    await asyncio.sleep(0.05)
    metrics = service.metrics()
    if metrics.get("status") == "unavailable":
        raise HTTPException(status_code=503, detail=str(metrics.get("last_error") or "unavailable"))
    return metrics


@router.post("/screen/stop")
async def stop_screen(request: Request) -> dict[str, object]:
    service = getattr(request.app.state, "screen", None)
    if service is None:
        raise HTTPException(status_code=503, detail="screen service is not available")
    return await service.stop()


@router.post("/screen/toggle")
def toggle_screen(request: Request) -> dict[str, object]:
    service = getattr(request.app.state, "screen", None)
    if service is None:
        raise HTTPException(status_code=503, detail="screen service is not available")
    paused = service.toggle()
    return {**service.metrics(), "paused": paused}


@router.get("/mic")
def get_mic(request: Request) -> dict[str, object]:
    service = getattr(request.app.state, "mic", None)
    if service is None:
        return {"enabled": False, "status": "unavailable"}
    return service.metrics()


@router.post("/mic/start")
async def start_mic(request: Request) -> dict[str, object]:
    service = getattr(request.app.state, "mic", None)
    if service is None:
        raise HTTPException(status_code=503, detail="microphone service is not available")
    await service.start()
    return service.metrics()


@router.post("/mic/stop")
async def stop_mic(request: Request) -> dict[str, object]:
    service = getattr(request.app.state, "mic", None)
    if service is None:
        raise HTTPException(status_code=503, detail="microphone service is not available")
    await service.stop()
    return service.metrics()


@router.post("/mic/transcript", status_code=status.HTTP_201_CREATED)
def mic_transcript(request: Request, body: TranscriptRequest) -> dict[str, object]:
    """Inject a transcript (manual test or browser-side STT fallback)."""
    service = getattr(request.app.state, "mic", None)
    if service is None:
        raise HTTPException(status_code=503, detail="microphone service is not available")
    text = service.inject_transcript(body.text, energy=body.energy)
    if not text:
        raise HTTPException(status_code=422, detail="transcript is empty")
    return {"text": text, "metrics": service.metrics()}


@router.get("/settings")
def get_settings(request: Request) -> dict[str, object]:
    state = _state(request)
    runtime: RuntimeSettings = state.runtime
    return {
        "settings": runtime.snapshot(),
        "effective": {
            "lookahead_s": state.lookahead_s,
            "chat_target_buffer": state.settings.chat_target_buffer,
            "mic_enabled": state.settings.mic_enabled,
            "perception_enabled": state.settings.perception_enabled,
        },
        "model_profile": {
            "path": str(state.settings.model_profile),
            "present": state.settings.model_profile.is_file(),
        },
    }


@router.patch("/settings")
async def update_settings(request: Request, body: SettingsRequest) -> dict[str, object]:
    state = _state(request)
    patch = SettingsPatch(
        preset=body.preset,
        content_mode=body.content_mode,
        intensity=body.intensity,
        chat_target_buffer=body.chat_target_buffer,
        lookahead_s=body.lookahead_s,
        model_path=body.model_path,
        model_quant=body.model_quant,
        tts_enabled=body.tts_enabled,
        mic_enabled=body.mic_enabled,
    )
    previous_mic = state.settings.mic_enabled
    try:
        snapshot = state.runtime.update(patch)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    state.settings.chat_target_buffer = int(snapshot["chat_target_buffer"])
    state.lookahead_s = float(snapshot["lookahead_s"])
    state.settings.mic_enabled = bool(snapshot["mic_enabled"])
    mic = getattr(request.app.state, "mic", None)
    if mic is not None and state.settings.mic_enabled != previous_mic:
        if state.settings.mic_enabled:
            await mic.start()
        else:
            await mic.stop()
    return {
        "settings": snapshot,
        "effective": {
            "lookahead_s": state.lookahead_s,
            "chat_target_buffer": state.settings.chat_target_buffer,
            "mic_enabled": state.settings.mic_enabled,
        },
        "note": "intensity and content mode apply now; look-ahead applies to the next stream",
    }


@router.post("/settings/preset/{name}")
def apply_preset(request: Request, name: str) -> dict[str, object]:
    state = _state(request)
    try:
        snapshot = state.runtime.apply_preset(name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    state.settings.chat_target_buffer = int(snapshot["chat_target_buffer"])
    state.lookahead_s = float(snapshot["lookahead_s"])
    return {"settings": snapshot}


@router.get("/system")
def get_system(request: Request) -> dict[str, object]:
    metrics = getattr(request.app.state, "system", None)
    if metrics is None:
        raise HTTPException(status_code=503, detail="system metrics are not available")
    return metrics.snapshot()


__all__ = ["router"]
