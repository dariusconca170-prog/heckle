"""FastAPI application entry point.

Run with: ``uvicorn backend.main:app --host 127.0.0.1 --port 8000``.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from .api import features, overlay, routes, ws
from .chatter.service import ChatterService
from .config import Settings, get_settings
from .dynamics.service import DynamicsService
from .features.clips import ClipService
from .features.mic import MicService
from .features.polls import PollService
from .features.screen import ScreenService
from .features.stream_events import StreamEventService
from .features.system import SystemMetrics
from .perception.service import PerceptionService
from .persistence.service import PersistenceService
from .state import AppState

BOOT_TIME = time.monotonic()
_MISSING_FRONTEND = """<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Twitch Simulator</title>
<style>body{font-family:system-ui;background:#0e0e10;color:#efeff1;padding:3rem;line-height:1.6}
code{background:#18181b;padding:.2rem .4rem;border-radius:4px}</style>
<h1>Frontend is not built</h1>
<p>Install Node.js, then run:</p>
<pre><code>cd frontend
npm install
npm run build</code></pre>
<p>Then reload this page. The API is already running; see <a href="/health">/health</a>.</p>
</html>"""


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.ensure_dirs()
    state = AppState.build(settings)
    perception = PerceptionService(state)
    chatter = ChatterService(state)
    dynamics = DynamicsService(state, chatter)
    persistence = PersistenceService(state)
    polls = PollService(
        state,
        personas=[*chatter.roster.personas, *dynamics.crowd.personas[:36]],
    )
    stream_events = StreamEventService(
        state,
        names=[persona.handle for persona in dynamics.crowd.personas],
    )
    clips = ClipService(state)
    mic = MicService(state)
    screen = ScreenService(state)
    system = SystemMetrics(state, chatter, dynamics, clips)
    state.polls = polls
    state.stream_events = stream_events
    state.clips = clips
    state.mic = mic
    state.screen = screen

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Persistence starts first so the first released message of a session
        # is already being logged.
        await persistence.start()
        if settings.dynamics_enabled:
            await dynamics.start()
            if settings.polls_enabled:
                await polls.start()
            await stream_events.start()
            if settings.clips_enabled:
                await clips.start()
        if settings.mic_enabled:
            await mic.start()
        if settings.perception_enabled and settings.model_profile.is_file():
            await perception.start()
        try:
            yield
        finally:
            await screen.stop()
            await mic.stop()
            await clips.stop()
            await stream_events.stop()
            await polls.stop()
            await dynamics.stop()
            await perception.stop()
            await persistence.stop()
            await chatter.close()
            state.close()

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.app_state = state
    app.state.perception = perception
    app.state.chatter = chatter
    app.state.dynamics = dynamics
    app.state.persistence = persistence
    app.state.polls = polls
    app.state.stream_events = stream_events
    app.state.clips = clips
    app.state.mic = mic
    app.state.screen = screen
    app.state.system = system
    app.include_router(routes.router)
    app.include_router(features.router)
    app.include_router(overlay.router)
    app.include_router(ws.router)

    @app.get("/health", tags=["system"])
    def health(request: Request) -> dict[str, object]:
        current: AppState = request.app.state.app_state
        session = current.session
        return {
            "status": "ok",
            "app": current.settings.app_name,
            "version": current.settings.version,
            "uptime_s": round(time.monotonic() - BOOT_TIME, 1),
            "session_id": session.id if session else None,
            "clock": current.clock_snapshot().as_dict(),
            "frontend_built": (current.settings.frontend_dist / "index.html").is_file(),
            "llm_profile_present": current.settings.model_profile.is_file(),
            "perception": perception.metrics(),
            "scene_samples": current.timeline.snapshot(limit=1)["sample_count"],
            "chat": chatter.metrics(),
            "dynamics": dynamics.metrics(),
            "persistence": persistence.metrics(),
            "polls": polls.metrics(),
            "stream_events": stream_events.metrics(),
            "clips": clips.metrics(),
            "mic": mic.metrics(),
            "screen": screen.metrics(),
        }

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    dist: Path = settings.frontend_dist
    if (dist / "index.html").is_file():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    else:

        @app.get("/", include_in_schema=False)
        def missing_frontend() -> HTMLResponse:
            return HTMLResponse(_MISSING_FRONTEND, status_code=200)

    return app


app = create_app()
