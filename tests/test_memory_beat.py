from __future__ import annotations

import random
from dataclasses import dataclass

from backend.chatter.memory import (
    MemoryCandidate,
    lore_reference,
    mentions_memory,
)
from backend.chatter.service import ChatterService
from backend.config import Settings
from backend.db.repo import MemoryRow
from backend.perception.scene import SceneState
from backend.state import AppState


@dataclass
class _Completion:
    text: str


class _FakeClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[list[dict[str, str]]] = []

    async def chat(self, messages, **_kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(messages)
        return _Completion(self.text)

    async def close(self) -> None:
        return None


def _candidate(kind: str, key: str, value: str) -> MemoryCandidate:
    return MemoryCandidate(handle="lore_dump", kind=kind, key=key, value=value, weight=0.8)


def test_mentions_memory_matches_content_tokens() -> None:
    candidates = [_candidate("lore", "fail:12", "fail: player falls in the cave")]
    assert mentions_memory("still thinking about that cave", candidates)
    assert mentions_memory("the fail was insane", candidates)
    assert not mentions_memory("nice frame", candidates)


def test_generic_words_do_not_count_as_a_memory_reference() -> None:
    # "last" appears in this opinion and in the lore-keeper template, but the
    # template is not a reference to it.
    candidates = [_candidate("opinion", "last:likes", "likes last")]
    assert not mentions_memory("just like last stream", candidates)
    assert mentions_memory("still saying why push there", [
        _candidate("catchphrase", "why push there", "why push there")
    ])


def test_lore_reference_prefers_a_catchphrase() -> None:
    candidates = [
        _candidate("lore", "fail:12", "fail: player falls in the cave"),
        _candidate("catchphrase", "why push there", "why push there"),
    ]
    line = lore_reference(candidates)
    assert line == 'bro still says "why push there"'


def test_lore_reference_falls_back_to_event_kind() -> None:
    line = lore_reference([_candidate("lore", "fail:12", "fail: player falls in the cave")])
    assert line == "still not over that fail"
    assert lore_reference([]) == ""


async def _state_with_memory(settings: Settings) -> tuple[AppState, ChatterService]:
    state = AppState.build(settings)
    record = state.start_session(media_id="sample.mp4", title="cave run")
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
    client = _FakeClient('["lore_dump: nice"]')
    chatter = ChatterService(state, client=client, rng=random.Random(4))
    return state, chatter


async def test_requested_lore_is_never_dropped(settings: Settings) -> None:
    state, chatter = await _state_with_memory(settings)
    persona = chatter.roster.by_handle("lore_dump")
    assert persona is not None
    scene = SceneState(
        t=12.0,
        summary="player hides behind a rusty car",
        action="hiding",
        mood="tense",
        surprise=0.6,
    )
    result = await chatter.generate(
        scene,
        personas=[persona],
        publish=False,
        lore_handles={"lore_dump"},
    )
    message = result.messages[0]
    assert message.source == "template"
    assert message.text == "still not over that fail"
    # The request reached the model as a task, not just background memory.
    user = chatter._client.calls[0][1]["content"]  # type: ignore[union-attr]
    assert "memory: remembers fail: player falls in the cave" in user
    assert "work one of these memories into this line" in user
    await chatter.close()
    state.close()


async def test_lore_line_from_model_is_kept(settings: Settings) -> None:
    state = AppState.build(settings)
    record = state.start_session(media_id="sample.mp4")
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
    chatter = ChatterService(
        state,
        client=_FakeClient('["lore_dump: still thinking about that cave"]'),
        rng=random.Random(5),
    )
    persona = chatter.roster.by_handle("lore_dump")
    assert persona is not None
    scene = SceneState(t=12.0, summary="player hides", action="hiding")
    result = await chatter.generate(
        scene,
        personas=[persona],
        publish=False,
        lore_handles={"lore_dump"},
    )
    assert result.messages[0].source == "model"
    assert result.messages[0].text == "still thinking about that cave"
    await chatter.close()
    state.close()
