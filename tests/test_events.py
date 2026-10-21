from __future__ import annotations

import pytest

from backend.events import CLOCK_UPDATE, EventBus


@pytest.mark.asyncio
async def test_fanout_filters_by_type() -> None:
    bus = EventBus()
    clocks = bus.subscribe([CLOCK_UPDATE])
    all_events = bus.subscribe()
    bus.publish_nowait(CLOCK_UPDATE, {"t_play": 1.0})
    bus.publish_nowait("chat.message", {"text": "W"})
    clock_event = await clocks.get()
    all_first = await all_events.get()
    all_second = await all_events.get()
    assert clock_event is not None and clock_event.type == CLOCK_UPDATE
    assert all_first is not None and all_first.type == CLOCK_UPDATE
    assert all_second is not None and all_second.type == "chat.message"
    bus.close()


@pytest.mark.asyncio
async def test_slow_subscriber_drops_oldest() -> None:
    bus = EventBus()
    sub = bus.subscribe(maxsize=2)
    bus.publish_nowait("a")
    bus.publish_nowait("b")
    bus.publish_nowait("c")
    assert sub.dropped == 1
    first = await sub.get()
    second = await sub.get()
    assert first is not None and first.type == "b"
    assert second is not None and second.type == "c"
    bus.close()
    assert await sub.get() is None


def test_wildcard_subscription() -> None:
    bus = EventBus()
    sub = bus.subscribe(["clock.*"])
    bus.publish_nowait("clock.seek", {})
    bus.publish_nowait("chat.message", {})
    assert sub.wants("clock.seek")
    assert not sub.wants("chat.message")
    bus.close()


def test_event_sequence_is_monotonic() -> None:
    bus = EventBus()
    one = bus.publish_nowait("one")
    two = bus.publish_nowait("two")
    assert two.seq == one.seq + 1
    bus.close()
    with pytest.raises(RuntimeError):
        bus.publish_nowait("three")
