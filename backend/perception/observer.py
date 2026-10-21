"""Run one frame through the multimodal observer and validate the result.

The observer is the only component allowed to turn pixels into scene facts.
Everything downstream (timeline, chatter, clips) consumes ``SceneState`` and
never sees the raw frame. A malformed response is an observation failure, not
a made-up scene: the service backs off and tries again on the next tick.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from ..llm.client import Completion, LLMClient
from ..llm.json_repair import JSONRepairError, parse_object
from ..llm.prompting import observer_system, observer_user
from .frames import Frame
from .scene import Mood, SceneState

_WHITESPACE = re.compile(r"\s+")
_WORD = re.compile(r"[^\W_]+(?:['-][^\W_]+)*", re.UNICODE)

_MOOD_ALIASES: dict[str, Mood] = {
    "tension": "tense",
    "suspense": "tense",
    "scary": "tense",
    "fear": "tense",
    "relaxed": "calm",
    "chill": "calm",
    "happy": "wholesome",
    "cute": "wholesome",
    "sweet": "wholesome",
    "funny": "funny",
    "humor": "funny",
    "comedy": "funny",
    "chaos": "chaotic",
    "energetic": "chaotic",
    "angry": "chaotic",
    "sad": "sad",
    "emotional": "sad",
    "sexy": "horny",
    "sexual": "horny",
    "aroused": "horny",
    "disgusting": "gross",
    "gross": "gross",
    "neutral": "neutral",
}


class ObservationError(RuntimeError):
    """The model output could not be turned into a valid SceneState."""

    def __init__(self, message: str, *, raw: str = "") -> None:
        super().__init__(message)
        self.raw = raw


@dataclass(slots=True)
class Observation:
    scene: SceneState
    completion: Completion
    latency_s: float


class Observer:
    def __init__(
        self,
        client: LLMClient,
        *,
        max_tokens: int = 80,
        temperature: float = 0.2,
    ) -> None:
        self.client = client
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def observe(self, frame: Frame, t: float) -> Observation:
        started = time.perf_counter()
        completion = await self.client.complete_vision(
            frame.data,
            observer_user(t),
            system_prompt=observer_system(),
            mime=frame.mime,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        latency = time.perf_counter() - started
        try:
            scene = scene_from_payload(completion.text, t)
        except (JSONRepairError, ValueError, TypeError) as exc:
            raise ObservationError(
                f"observer returned an invalid scene: {exc}", raw=completion.text
            ) from exc
        return Observation(scene=scene, completion=completion, latency_s=latency)


def scene_from_payload(text_or_mapping: str | dict[str, Any], t: float) -> SceneState:
    """Parse one observer response and coerce the fields we depend on.

    The contract stays strict (``SceneState`` rejects unknown fields and bad
    moods), but local models occasionally emit a number as a string or a close
    mood synonym. Coercing those is preferable to dropping an otherwise useful
    observation.
    """
    if isinstance(text_or_mapping, str):
        payload = parse_object(text_or_mapping)
    elif isinstance(text_or_mapping, dict):
        payload = text_or_mapping
    else:
        raise ObservationError(f"observer payload must be an object, got {type(text_or_mapping).__name__}")

    summary = _text(payload.get("summary"), fallback="unclear")
    summary = _cap_words(summary, 35)
    action = _cap_words(_text(payload.get("action"), fallback="scene"), 8)
    on_screen_text = _text(payload.get("on_screen_text"), fallback="")
    mood = _mood(payload.get("mood"))
    surprise = _clamped_float(payload.get("surprise"), 0.0)
    confidence = _clamped_float(payload.get("confidence"), 0.5)
    scene_change = _as_bool(payload.get("scene_change"), False)
    if summary.casefold() == "unclear":
        confidence = min(confidence, 0.25)

    objects = _objects(payload.get("objects"))
    try:
        return SceneState(
            t=max(0.0, float(t)),
            summary=summary,
            action=action,
            objects=objects,
            mood=mood,
            surprise=surprise,
            scene_change=scene_change,
            on_screen_text=on_screen_text,
            confidence=confidence,
        )
    except ValueError as exc:
        raise ObservationError(f"invalid SceneState: {exc}") from exc


def _text(value: Any, *, fallback: str) -> str:
    if isinstance(value, str):
        text = _WHITESPACE.sub(" ", value).strip()
        return text or fallback
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, list):
        joined = " ".join(str(item) for item in value if isinstance(item, (str, int, float)))
        text = _WHITESPACE.sub(" ", joined).strip()
        return text or fallback
    return fallback


def _cap_words(value: str, limit: int) -> str:
    words = _WORD.findall(value)
    if not words:
        return "unclear"
    if len(words) <= limit:
        return value
    return " ".join(words[:limit])


def _mood(value: Any) -> Mood:
    text = _text(value, fallback="neutral").casefold()
    if text in _MOOD_ALIASES:
        return _MOOD_ALIASES[text]
    allowed: tuple[Mood, ...] = (
        "tense",
        "calm",
        "funny",
        "chaotic",
        "sad",
        "horny",
        "wholesome",
        "gross",
        "neutral",
    )
    if text in allowed:
        return text  # type: ignore[return-value]
    return "neutral"


def _clamped_float(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return min(1.0, max(0.0, number))


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"true", "yes", "y", "1"}:
            return True
        if lowered in {"false", "no", "n", "0"}:
            return False
    return default


def _objects(value: Any) -> list[str]:
    if isinstance(value, str):
        raw = re.split(r"[,;|]", value)
    elif isinstance(value, list):
        raw = value
    else:
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = _WHITESPACE.sub(" ", str(item)).strip()
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text[:80])
    return result[:24]
