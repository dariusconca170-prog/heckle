"""HTTP API: media library, session lifecycle, clock and ranged video."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import time
from collections.abc import Iterator
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..media import (
    MediaError,
    RangeNotSatisfiable,
    content_disposition,
    content_type_for,
    parse_range,
)
from ..state import AppState

router = APIRouter(prefix="/api")
BOOT_TIME = time.monotonic()

# Videos are streamed straight to disk; this only stops a runaway upload from
# filling the drive. Ten minutes of 1080p is well under 2 GB.
MAX_UPLOAD_BYTES = 8 * 1024**3


class StartSessionRequest(BaseModel):
    media_id: str = Field(min_length=1, max_length=1024)
    title: str = Field(default="", max_length=160)
    category: str = Field(default="", max_length=80)
    viewer_count: int = Field(default=1200, ge=0, le=1_000_000)


class GenerateChatRequest(BaseModel):
    count: int = Field(default=10, ge=1, le=12)


def _state(request: Request) -> AppState:
    return request.app.state.app_state


@router.get("/media")
def list_media(request: Request) -> dict[str, object]:
    state = _state(request)
    items = state.list_media()
    return {"items": [item.as_dict() for item in items], "media_dir": str(state.library.root)}


@router.post("/media/upload", status_code=status.HTTP_201_CREATED)
async def upload_media(
    request: Request,
    name: str = Query(min_length=1, max_length=255),
) -> dict[str, object]:
    """Copy a browser-selected video into the media library.

    The body is the raw file (no multipart dependency); ``name`` is the
    original filename. Directories are stripped, the extension must be one the
    library serves, and an existing file is never overwritten.
    """
    state = _state(request)
    library = state.library
    filename = Path(name.replace("\\", "/")).name.strip()
    if not filename or filename.startswith(".") or "\x00" in filename:
        raise HTTPException(status_code=422, detail="invalid file name")
    suffix = Path(filename).suffix.lower()
    if suffix not in library.extensions:
        raise HTTPException(
            status_code=422,
            detail=(
                f"unsupported video type: {suffix or 'no extension'}; "
                f"use one of {', '.join(library.extensions)}"
            ),
        )

    library.root.mkdir(parents=True, exist_ok=True)
    target = _unique_upload_path(library.root, filename)
    written = 0
    try:
        with target.open("wb") as handle:
            async for chunk in request.stream():
                if not chunk:
                    continue
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="file is larger than the 8 GB limit")
                await asyncio.to_thread(handle.write, chunk)
        if written == 0:
            raise HTTPException(status_code=422, detail="empty file")
    except BaseException:
        target.unlink(missing_ok=True)
        raise

    item = library.resolve(target.relative_to(library.root).as_posix())
    return {"item": item.as_dict(), "uploaded": written}


def _unique_upload_path(root: Path, filename: str) -> Path:
    """Never overwrite a library file; add `` (2)``, `` (3)`` ... instead."""
    target = root / filename
    if not target.exists():
        return target
    stem, suffix = Path(filename).stem, Path(filename).suffix
    for index in range(2, 1000):
        candidate = root / f"{stem} ({index}){suffix}"
        if not candidate.exists():
            return candidate
    raise HTTPException(status_code=409, detail="too many copies of that file name")


@router.post("/session", status_code=status.HTTP_201_CREATED)
def start_session(request: Request, body: StartSessionRequest) -> dict[str, object]:
    state = _state(request)
    chatter = getattr(request.app.state, "chatter", None)
    if chatter is not None:
        chatter.reset()
    try:
        record = state.start_session(
            media_id=body.media_id,
            title=body.title,
            category=body.category,
            viewer_count=body.viewer_count,
        )
    except MediaError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"session": record.as_dict(), "clock": record.clock.snapshot().as_dict()}


@router.get("/session")
def get_session(request: Request) -> dict[str, object]:
    state = _state(request)
    if state.session is None:
        raise HTTPException(status_code=404, detail="no active session")
    return {"session": state.session.as_dict(), "clock": state.session.clock.snapshot().as_dict()}


@router.delete("/session")
def end_session(request: Request) -> dict[str, object]:
    state = _state(request)
    record = state.end_session()
    return {"ended": record.id if record else None}


@router.get("/clock")
def get_clock(request: Request) -> dict[str, object]:
    state = _state(request)
    return {
        "clock": state.clock_snapshot().as_dict(),
        "session_id": state.session.id if state.session else None,
    }


@router.get("/scene")
def get_scene(request: Request) -> dict[str, object]:
    state = _state(request)
    service = getattr(request.app.state, "perception", None)
    return {
        "active": state.session is not None,
        **state.timeline.snapshot(),
        "service": service.metrics() if service is not None else {"status": "unavailable"},
    }


@router.post("/chat/generate")
async def generate_chat(request: Request, body: GenerateChatRequest) -> dict[str, object]:
    chatter = getattr(request.app.state, "chatter", None)
    if chatter is None:
        raise HTTPException(status_code=503, detail="chat service is not available")
    result = await chatter.generate_preview(count=body.count)
    return result.as_dict()


@router.get("/chat/recent")
def get_recent_chat(request: Request, limit: int = 50) -> dict[str, object]:
    chatter = getattr(request.app.state, "chatter", None)
    if chatter is None:
        return {"messages": []}
    limit = min(max(1, limit), 200)
    return {"messages": [message.as_dict() for message in list(chatter.recent)[-limit:]]}


@router.get("/personas")
def get_personas(request: Request) -> dict[str, object]:
    chatter = getattr(request.app.state, "chatter", None)
    if chatter is None:
        return {"personas": []}
    return {"personas": [persona.as_dict() for persona in chatter.roster.personas]}


@router.get("/config")
def get_config(request: Request) -> dict[str, object]:
    state = _state(request)
    settings = state.settings
    return {
        "app_name": settings.app_name,
        "version": settings.version,
        "lookahead_s": state.lookahead_s,
        "time_update_hz": settings.clock_broadcast_hz,
        "llm_profile_present": settings.model_profile.exists(),
        "perception_enabled": settings.perception_enabled,
        "dynamics_enabled": settings.dynamics_enabled,
        "chat_target_buffer": settings.chat_target_buffer,
        "persistence_enabled": settings.persistence_enabled,
        "memory_enabled": settings.memory_enabled,
        "uptime_s": round(time.monotonic() - BOOT_TIME, 1),
    }


@router.get("/dynamics")
def get_dynamics(request: Request) -> dict[str, object]:
    dynamics = getattr(request.app.state, "dynamics", None)
    if dynamics is None:
        return {"enabled": False}
    return {"enabled": True, **dynamics.metrics()}


@router.get("/history/streams")
def history_streams(request: Request, limit: int = 20) -> dict[str, object]:
    """Past streams, newest first (M4: resume after a restart)."""
    state = _state(request)
    repo = state.persistence.repo
    if repo is None:
        return {"streams": [], "persistence": state.persistence.metrics()}
    return {
        "streams": repo.streams(limit=min(max(1, limit), 100)),
        "persistence": state.persistence.metrics(),
    }


@router.get("/history/streams/{stream_id}")
def history_stream(stream_id: str, request: Request) -> dict[str, object]:
    state = _state(request)
    repo = state.persistence.repo
    stream = repo.stream(stream_id) if repo is not None else None
    if stream is None:
        raise HTTPException(status_code=404, detail="unknown stream")
    return {"stream": stream}


@router.get("/history/streams/{stream_id}/messages")
def history_messages(
    stream_id: str,
    request: Request,
    limit: int = 200,
    offset: int = 0,
    newest_first: bool = False,
) -> dict[str, object]:
    state = _state(request)
    repo = state.persistence.repo
    if repo is None:
        raise HTTPException(status_code=503, detail="persistence is disabled")
    if repo.stream(stream_id) is None:
        raise HTTPException(status_code=404, detail="unknown stream")
    messages = repo.messages(
        stream_id,
        limit=min(max(1, limit), 1000),
        offset=max(0, offset),
        newest_first=newest_first,
    )
    return {
        "stream_id": stream_id,
        "messages": messages,
        "total": repo.message_count(stream_id),
        "offset": max(0, offset),
        "limit": min(max(1, limit), 1000),
    }


@router.get("/history/memory")
def history_memory(
    request: Request,
    handle: str | None = None,
    limit: int = 10,
) -> dict[str, object]:
    """Structured persona memory (PLAN.md §8), not raw chat."""
    state = _state(request)
    store = state.persistence
    if store.repo is None:
        return {"memory": [], "viewers": {}, "persistence": store.metrics()}
    capped = min(max(1, limit), 50)
    if handle:
        return {
            "handle": handle,
            "memory": store.raw_memory(handle, limit=capped),
            "loaded_prompt": list(store.memory_for(handle)),
            "persistence": store.metrics(),
        }
    return {
        "viewers": store.all_memory(per_viewer=capped),
        "persistence": store.metrics(),
    }


@router.get("/export/chat")
def export_current_chat(request: Request, format: str = "json") -> Response:
    state = _state(request)
    if state.session is None:
        raise HTTPException(status_code=404, detail="no active session")
    return _chat_export(state, state.session.id, format)


@router.get("/history/streams/{stream_id}/export")
def export_stream_chat(stream_id: str, request: Request, format: str = "json") -> Response:
    state = _state(request)
    repo = state.persistence.repo
    if repo is None:
        raise HTTPException(status_code=503, detail="persistence is disabled")
    if repo.stream(stream_id) is None:
        raise HTTPException(status_code=404, detail="unknown stream")
    return _chat_export(state, stream_id, format)


def _chat_export(state: AppState, stream_id: str, format: str) -> Response:
    repo = state.persistence.repo
    if repo is None:
        raise HTTPException(status_code=503, detail="persistence is disabled")
    messages = repo.messages(stream_id, limit=100_000)
    kind = format.strip().casefold()
    stamp = stream_id[:16]
    if kind == "csv":
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(
            [
                "id",
                "t_playback",
                "handle",
                "archetype",
                "tier",
                "kind",
                "text",
                "reply_handle",
                "reply_text",
                "mentions",
                "emotes",
            ]
        )
        for message in messages:
            writer.writerow(
                [
                    message["id"],
                    message["t_playback"],
                    message["handle"],
                    message["archetype"],
                    message["tier"],
                    message["kind"],
                    message["text"],
                    message.get("reply_handle") or "",
                    message.get("reply_text") or "",
                    " ".join(message.get("mentions") or []),
                    " ".join(message.get("emotes") or []),
                ]
            )
        return Response(
            content=buffer.getvalue(),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="chat-{stamp}.csv"'},
        )
    if kind != "json":
        raise HTTPException(status_code=422, detail="format must be json or csv")
    body = json.dumps(
        {"stream_id": stream_id, "messages": messages},
        ensure_ascii=False,
        indent=2,
    )
    return Response(
        content=body,
        media_type="application/json; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="chat-{stamp}.json"'},
    )


@router.get("/media/{token}")
def get_media(token: str, request: Request) -> Response:
    return _media_response(token, request, head_only=False)


@router.head("/media/{token}")
def head_media(token: str, request: Request) -> Response:
    return _media_response(token, request, head_only=True)


def _media_response(token: str, request: Request, *, head_only: bool) -> Response:
    state = _state(request)
    item = state.tokens.resolve(token)
    if item is None:
        raise HTTPException(status_code=404, detail="unknown media token")
    path: Path = item.path
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise HTTPException(status_code=410, detail="media file is no longer available") from exc

    try:
        byte_range = parse_range(request.headers.get("range"), size)
    except RangeNotSatisfiable as exc:
        raise HTTPException(
            status_code=status.HTTP_416_RANGE_NOT_SATISFIABLE,
            detail="range not satisfiable",
            headers={"Content-Range": f"bytes */{size}", "Accept-Ranges": "bytes"},
        ) from exc

    if byte_range is None:
        start, end = 0, max(size - 1, 0)
        response_status = status.HTTP_200_OK
        length = size
    else:
        start, end = byte_range
        response_status = status.HTTP_206_PARTIAL_CONTENT
        length = end - start + 1

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
        "Content-Disposition": content_disposition(path),
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
    }
    if byte_range is not None:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"

    media_type = content_type_for(path)
    if head_only or size == 0:
        return Response(status_code=response_status, headers=headers, media_type=media_type)

    return StreamingResponse(
        _file_chunks(path, start, length),
        status_code=response_status,
        headers=headers,
        media_type=media_type,
    )


def _file_chunks(path: Path, start: int, length: int, chunk_size: int = 256 * 1024) -> Iterator[bytes]:
    remaining = length
    with path.open("rb") as handle:
        handle.seek(start)
        while remaining > 0:
            data = handle.read(min(chunk_size, remaining))
            if not data:
                break
            remaining -= len(data)
            yield data
