from __future__ import annotations

import json

import pytest

from backend.chatter.guard import MAX_WORDS, check, classify, dedupe_key, template_for
from backend.chatter.personas import CORE_ROSTER, PersonaRoster
from backend.chatter.service import ChatterService
from backend.config import Settings
from backend.llm.client import Completion
from backend.perception.scene import SceneState
from backend.state import AppState


def _scene() -> SceneState:
    return SceneState(
        t=12.0,
        summary="a knight rolls under a giant sword swing",
        action="rolling under sword",
        objects=["knight", "sword"],
        mood="tense",
        surprise=0.7,
        confidence=0.9,
    )


def test_guard_drops_refusals_narration_and_long_lines() -> None:
    assert not check("As an AI, I can't describe that.").ok
    assert check("As an AI, I can't describe that.").reason == "refusal"
    assert check("The player is hiding behind a car").reason == "narrator"
    assert check("the scene appears to show a car").reason == "narrator"
    assert check(" ".join(["word"] * (MAX_WORDS + 1))).reason == "too_long"
    assert check("I can't believe he just did that LUL").ok
    assert check("W", persona_caps=True).text == "W"
    assert classify("This is inappropriate and offensive") == "moralized"
    assert classify("02:15 - this cave is a big turn") == "meta"
    assert check("*this* is *not* boring").text == "this is not boring"
    assert dedupe_key("why push there?") == dedupe_key("why push there")
    assert dedupe_key("gg and you got this <3 <3") == dedupe_key("gg and you got this")


def test_templates_are_valid_chat() -> None:
    for persona in CORE_ROSTER:
        for index in range(4):
            text = template_for(persona.archetype, index)
            result = check(text, persona_caps=persona.caps)
            assert result.ok, (persona.handle, text, result.reason)
            assert len(result.text.split()) <= MAX_WORDS


class FakeChatClient:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.calls = 0

    async def chat(self, messages, **kwargs) -> Completion:
        self.calls += 1
        return Completion(
            text=json.dumps(self.lines),
            model="fake",
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
            elapsed_s=0.05,
        )

    async def close(self) -> None:
        return None


class SequencedChatClient:
    def __init__(self, batches: list[list[str]]) -> None:
        self.batches = batches
        self.calls = 0

    async def chat(self, messages, **kwargs) -> Completion:
        lines = self.batches[min(self.calls, len(self.batches) - 1)]
        self.calls += 1
        return Completion(
            text=json.dumps(lines),
            model="fake",
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
            elapsed_s=0.05,
        )

    async def close(self) -> None:
        return None


@pytest.fixture
def state(settings: Settings) -> AppState:
    return AppState.build(settings)


async def test_generate_filters_and_fills_templates(state: AppState) -> None:
    roster = PersonaRoster((CORE_ROSTER[0], CORE_ROSTER[1], CORE_ROSTER[2], CORE_ROSTER[8]))
    client = FakeChatClient(
        [
            "W",
            "The player is hiding behind a car",
            "As an AI, I can't help with that.",
            "LUL",
        ]
    )
    service = ChatterService(state, client=client, roster=roster)  # type: ignore[arg-type]
    service.choose = lambda count: list(roster.personas)  # type: ignore[method-assign]
    result = await service.generate(_scene(), count=4)
    assert client.calls == 1
    assert result.fallback is False
    assert len(result.messages) == 4
    assert all(message.word_count <= MAX_WORDS for message in result.messages)
    assert result.messages[0].text == "W"
    assert result.messages[0].source == "model"
    assert result.messages[1].source == "template"
    assert result.messages[2].source == "template"
    assert result.messages[3].text == "LUL"
    assert all(classify(message.text) == "ok" for message in result.messages)
    assert len({message.text.casefold() for message in result.messages}) == 4


async def test_handle_prefixes_are_mapped_and_stripped(state: AppState) -> None:
    roster = PersonaRoster((CORE_ROSTER[0], CORE_ROSTER[1]))
    client = FakeChatClient(["xX_deadlift_Xx: W", "mothgirl_enjoyer: lol"])
    service = ChatterService(state, client=client, roster=roster)  # type: ignore[arg-type]
    result = await service.generate(_scene(), count=2)
    by_handle = {message.handle: message for message in result.messages}
    assert by_handle["xX_deadlift_Xx"].text == "W"
    assert by_handle["mothgirl_enjoyer"].text == "lol"
    assert all(":" not in message.text for message in result.messages)


async def test_generate_degrades_to_templates_without_model(state: AppState) -> None:
    service = ChatterService(state)
    result = await service.generate(_scene(), count=10)
    assert result.fallback is True
    assert len(result.messages) == 10
    assert all(message.source == "template" for message in result.messages)
    assert len({message.text.casefold() for message in result.messages}) == 10


async def test_dedupe_against_recent_batch(state: AppState) -> None:
    roster = PersonaRoster((CORE_ROSTER[0], CORE_ROSTER[1]))
    client = FakeChatClient(["W", "LUL"])
    service = ChatterService(state, client=client, roster=roster)  # type: ignore[arg-type]
    first = await service.generate(_scene(), count=2)
    second = await service.generate(_scene(), count=2)
    first_texts = {message.text.casefold() for message in first.messages}
    second_texts = {message.text.casefold() for message in second.messages}
    assert first_texts.isdisjoint(second_texts)


def test_reset_clears_recent(state: AppState) -> None:
    service = ChatterService(state)
    service.recent.append(  # type: ignore[arg-type]
        type("M", (), {"text": "x"})()
    )
    assert len(service.recent) == 1
    service.reset()
    assert len(service.recent) == 0


async def test_dedupe_ignores_trailing_emote_variants(state: AppState) -> None:
    cozy = next(persona for persona in CORE_ROSTER if persona.handle == "cozybean")
    roster = PersonaRoster((cozy,))
    client = SequencedChatClient(
        [
            ["cozybean: gg and you got this <3"],
            ["cozybean: gg and you got this"],
        ]
    )
    service = ChatterService(state, client=client, roster=roster)  # type: ignore[arg-type]
    first = await service.generate(_scene(), count=1, publish=False)
    second = await service.generate(_scene(), count=1, publish=False)
    assert first.messages[0].source == "model"
    assert second.messages[0].source == "template"
