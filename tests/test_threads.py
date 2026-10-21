from __future__ import annotations

import random

from backend import events
from backend.chatter.service import ChatterService
from backend.config import Settings
from backend.dynamics.service import DynamicsService
from backend.dynamics.threads import ParentRef, ReplyPlanner, build_argument, is_replyable
from backend.llm.prompting import build_chatter_messages
from backend.state import AppState


def _parents() -> list[ParentRef]:
    return [
        ParentRef(
            id=f"p{index}",
            handle=f"viewer{index}",
            text=text,
            release_t=10.0 + index,
            archetype="normie",
        )
        for index, text in enumerate(
            ["why did he push there", "that was clean", "no way", "L take", "what was that"]
        )
    ]


def test_reply_rate_and_structural_rules() -> None:
    planner = ReplyPlanner(random.Random(7), reply_rate=0.20)
    parents = _parents()
    replies = 0
    for index in range(4000):
        release = 20.0 + index * 0.001
        parent = planner.maybe_reply("me", release, parents)
        if parent is None:
            continue
        replies += 1
        assert parent.handle != "me"
        assert parent.release_t <= release - 1.0
    rate = replies / 4000
    assert 0.17 <= rate <= 0.23, rate


def test_reply_rejects_self_and_future() -> None:
    planner = ReplyPlanner(random.Random(3), reply_rate=1.0)
    own = ParentRef("own", "me", "my own line", 5.0)
    future = ParentRef("future", "you", "not released yet", 40.0)
    assert planner.maybe_reply("me", 30.0, [own, future]) is None
    assert planner.maybe_reply("me", 20.0, _parents()) is not None


def test_mention_rate() -> None:
    planner = ReplyPlanner(random.Random(11), mention_rate=0.04)
    hits = sum(1 for _ in range(5000) if planner.maybe_mention("streamer"))
    rate = hits / 5000
    assert 0.03 <= rate <= 0.05, rate
    assert ReplyPlanner(random.Random(1), mention_rate=1.0).maybe_mention("streamer") == "@streamer"


def test_argument_thread_escalates_and_stops() -> None:
    rng = random.Random(21)
    for _ in range(30):
        lines = build_argument(rng)
        assert 5 <= len(lines) <= 7
        for index, line in enumerate(lines):
            assert len(line.text.split()) <= 12
            if index == 0:
                assert line.reply_index is None
            else:
                assert line.reply_index is not None
                assert 0 <= line.reply_index < index
        final = lines[-1]
        assert final.persona.handle not in {lines[0].persona.handle, lines[1].persona.handle}
        assert final.reply_index == len(lines) - 2
        assert any(
            word in final.text for word in ("shut up", "both of you", "talk about")
        )


def test_is_replyable_flags_questions_and_hot_takes() -> None:
    assert is_replyable("why would you push there")
    assert is_replyable("that was trash")
    assert not is_replyable("")


async def _dynamics(settings: Settings) -> tuple[AppState, DynamicsService, object]:
    state = AppState.build(settings)
    record = state.start_session(media_id="sample.mp4", viewer_count=1200, title="cave_run")
    record.clock.set_duration(120.0)
    record.clock.update(0.0, playing=True)
    chatter = ChatterService(state, rng=random.Random(4))
    dynamics = DynamicsService(state, chatter, rng=random.Random(5))
    dynamics._begin_session(record)
    dynamics._recent_refs.extend(_parents())
    return state, dynamics, record


async def test_refill_creates_replies_and_mentions(settings: Settings) -> None:
    _state, dynamics, record = await _dynamics(settings)
    for _ in range(8):
        dynamics._recent_refs.extend(_parents())
        await dynamics._refill(10.0, record.id, record.clock.generation)

    messages = [item.message for item in dynamics._buffer]
    assert messages
    replies = [message for message in messages if message.reply_to]
    assert replies, "expected at least one planned reply"
    parent_by_id = {parent.id: parent for parent in dynamics._recent_refs}
    for message in replies:
        assert message.reply_to_handle
        assert message.reply_to_handle != message.handle
        parent = parent_by_id.get(message.reply_to or "")
        if parent is not None:
            assert parent.handle == message.reply_to_handle
        assert message.reply_to_text
    mentions = [message for message in messages if message.mentions]
    assert all(mention.startswith("@") for message in mentions for mention in message.mentions)
    assert all(not message.reply_to for message in mentions)


async def test_argument_is_scheduled_as_a_thread(settings: Settings) -> None:
    _state, dynamics, _record = await _dynamics(settings)
    dynamics._next_argument_at = 0.0
    dynamics._maybe_argument(20.0)
    assert dynamics.metrics()["arguments"] == 1
    argument = [item.message for item in dynamics._buffer]
    assert len(argument) >= 5
    assert argument[0].reply_to is None
    assert all(message.reply_to for message in argument[1:])
    assert len({message.handle for message in argument}) >= 2
    assert any("shut up both of you" in message.text for message in argument) or any(
        "both of you are wrong" in message.text for message in argument
    ) or any("can we talk about the game" in message.text for message in argument)


async def test_reply_rate_exceeds_fifteen_percent(settings: Settings) -> None:
    state = AppState.build(settings)
    record = state.start_session(media_id="sample.mp4", viewer_count=1200, title="threads")
    record.clock.set_duration(240.0)
    record.clock.update(0.0, playing=True)
    chatter = ChatterService(state, rng=random.Random(12))
    dynamics = DynamicsService(state, chatter, rng=random.Random(13))
    dynamics._begin_session(record)
    dynamics.hawkes.set_context(active_chatters=80, intensity=4.0)
    subscription = state.bus.subscribe([events.CHAT_MESSAGE], maxsize=2048)
    released = []
    for second in range(1, 91):
        dynamics.hawkes.advance_to(float(second))
        dynamics._release_due(float(second))
        dynamics._maybe_refill(float(second))
        if dynamics._refill_task is not None:
            await dynamics._refill_task
        while (event := subscription.get_nowait()) is not None:
            released.append(event.payload)
    assert len(released) > 100
    replies = [message for message in released if message.get("reply_to")]
    rate = len(replies) / len(released)
    assert rate >= 0.15, f"reply rate {rate:.2f} below the M5 floor"
    assert all(message["reply_to_handle"] != message["handle"] for message in replies)
    mentions = [message for message in released if message.get("mentions")]
    assert 0.015 <= len(mentions) / len(released) <= 0.07
    state.close()


async def test_reply_directive_reaches_the_prompt(settings: Settings) -> None:
    state = AppState.build(settings)
    prompt = build_chatter_messages(
        scene={"summary": "a player hides behind a car", "action": "hiding"},
        personas=[
            {
                "handle": "mothgirl_enjoyer",
                "archetype": "meme",
                "style": "lowercase",
                "mood": "feral",
            }
        ],
        reply_targets={
            "mothgirl_enjoyer": {"handle": "cozybean", "text": "why would you push there"}
        },
    )
    user = prompt[-1]["content"]
    assert 'replying to cozybean: "why would you push there"' in user
    assert "answers that message directly" in user
    state.close()


def test_threads_persist_with_parent_links(settings: Settings) -> None:
    state = AppState.build(settings)
    record = state.start_session(media_id="sample.mp4", title="threads")
    store = state.persistence
    store.record_message(
        {
            "id": "live-parent",
            "handle": "cozybean",
            "archetype": "wholesome",
            "text": "why would you push there",
            "kind": "chat",
            "tier": "core",
            "t": 4.0,
            "release_t": 4.2,
            "created_at": 1.0,
        }
    )
    store.record_message(
        {
            "id": "live-child",
            "handle": "not_a_bot_92",
            "archetype": "backseater",
            "text": "because the cooldown was up",
            "kind": "chat",
            "tier": "core",
            "t": 5.0,
            "release_t": 5.4,
            "created_at": 2.0,
            "reply_to": "live-parent",
            "mentions": [],
            "emotes": ["LUL"],
        }
    )
    rows = store.repo.messages(record.id)
    assert len(rows) == 2
    child = rows[1]
    assert child["reply_to"] == rows[0]["id"]
    assert child["reply_handle"] == "cozybean"
    assert child["reply_text"] == "why would you push there"
    assert child["emotes"] == ["LUL"]
    state.close()
