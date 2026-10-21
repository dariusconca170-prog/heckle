from __future__ import annotations

import random

from backend import events
from backend.chatter.guard import check
from backend.chatter.service import ChatterService
from backend.config import Settings
from backend.db.repo import MemoryRow
from backend.dynamics.service import DynamicsService, Planned
from backend.state import AppState


async def _dynamics(settings: Settings) -> tuple[AppState, DynamicsService, object]:
    state = AppState.build(settings)
    record = state.start_session(media_id="sample.mp4", viewer_count=1200)
    record.clock.set_duration(30.0)
    record.clock.update(0.0, playing=True)
    chatter = ChatterService(state, rng=random.Random(2))
    dynamics = DynamicsService(state, chatter, rng=random.Random(3))
    dynamics._begin_session(record)
    dynamics.hawkes.set_context(active_chatters=80, intensity=5.0)
    return state, dynamics, record


async def test_refill_release_and_seek(settings: Settings) -> None:
    state, dynamics, record = await _dynamics(settings)
    subscription = state.bus.subscribe([events.CHAT_MESSAGE], maxsize=512)
    dynamics._inject_first(0.0)
    await dynamics._refill(0.0, record.id, record.clock.generation)
    assert dynamics.metrics()["buffer_level"] > 0
    for step in range(1, 81):
        dynamics._release_due(step * 0.5)

    payloads = []
    while (event := subscription.get_nowait()) is not None:
        payloads.append(event.payload)

    assert payloads
    assert any(payload["text"] == "first" for payload in payloads)
    assert sum(1 for payload in payloads if payload["text"] == "first") == 1
    for payload in payloads:
        assert payload["word_count"] <= 12
        assert check(payload["text"]).ok
    assert len(dynamics.chatter.recent) > 0

    dynamics._handle_seek(record.clock.generation + 1, 12.0)
    assert dynamics.metrics()["buffer_level"] == 0
    assert dynamics.hawkes.t == 12.0


async def test_lore_beat_replaces_a_core_slot(settings: Settings) -> None:
    state, dynamics, record = await _dynamics(settings)
    assert state.persistence.repo is not None
    state.persistence.repo.save_memory(
        [
            MemoryRow(
                handle="lore_dump",
                kind="lore",
                key="fail:12",
                value="fail: player falls in the cave",
                weight=0.9,
                stream_id=record.id,
            )
        ]
    )
    state.persistence.refresh_memory()
    assert state.memory_candidates("lore_dump")
    dynamics.chatter._enabled = True
    hype = dynamics.chatter.roster.by_handle("xX_deadlift_Xx")
    assert hype is not None
    planned: list[Planned] = [
        Planned(trigger=1.0, release=2.0, persona=hype, tier="core"),
        Planned(trigger=1.5, release=2.5, persona=hype, tier="core"),
    ]
    handles = dynamics._plan_lore_beat(planned, 12.0)
    assert handles == {"lore_dump"}
    assert any(item.persona.handle == "lore_dump" for item in planned)
    # Still pending until the scheduled message exists: a seek that flushes the
    # batch must not silently drop the beat.
    assert dynamics._plan_lore_beat(planned, 20.0) == {"lore_dump"}
    dynamics._lore_beat_done = True
    dynamics._lore_beat_pending = None
    assert dynamics._plan_lore_beat(planned, 30.0) == set()


async def test_out_of_order_release_is_dropped(settings: Settings) -> None:
    state, dynamics, record = await _dynamics(settings)
    subscription = state.bus.subscribe([events.CHAT_MESSAGE], maxsize=64)
    # Build one message directly, then release it far past the 5 s lateness
    # ceiling. The scheduler drops it instead of showing stale chat.
    message = dynamics._make_message(
        dynamics.crowd.personas[0],
        "late line",
        source="crowd",
        t=0.0,
        text_override="late line",
    )
    dynamics._insert_message(
        message,
        release_t=1.0,
        trigger_t=1.0,
        tier="crowd",
        ready=True,
    )
    dynamics._release_due(20.0)
    assert dynamics.metrics()["dropped"] == 1
    assert subscription.get_nowait() is None
