"""The frozen perception -> chat contract (PLAN.md §5.2).

Do not add/remove/rename fields without asking. ``speech_text``,
``speech_energy`` and ``source`` are filled by the audio layer (video track or
microphone), not by the observer model, but they are present from M0 so the
contract never changes underneath later milestones.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Mood = Literal[
    "tense",
    "calm",
    "funny",
    "chaotic",
    "sad",
    "horny",
    "wholesome",
    "gross",
    "neutral",
]
Source = Literal["video_audio", "mic"]

_WHITESPACE = re.compile(r"\s+")


class SceneState(BaseModel):
    """One observation of what is on screen right now."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    t: float = Field(ge=0, description="Playback timestamp this observation describes.")
    summary: str = Field(min_length=1, description="<= 35 words, present tense, concrete nouns.")
    action: str = Field(min_length=1, description="What is happening, 2-5 words.")
    objects: list[str] = Field(default_factory=list)
    mood: Mood = "neutral"
    surprise: float = Field(default=0.0, ge=0.0, le=1.0)
    scene_change: bool = False
    on_screen_text: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    speech_text: str = ""
    speech_energy: float = Field(default=0.0, ge=0.0, le=1.0)
    source: Source = "video_audio"

    @field_validator("summary", "action", "on_screen_text", "speech_text", mode="before")
    @classmethod
    def _collapse_whitespace(cls, value: object) -> object:
        if isinstance(value, str):
            return _WHITESPACE.sub(" ", value).strip()
        return value

    @field_validator("summary")
    @classmethod
    def _summary_word_cap(cls, value: str) -> str:
        if len(value.split()) > 35:
            raise ValueError("summary must be <= 35 words")
        return value

    @field_validator("objects")
    @classmethod
    def _clean_objects(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        cleaned: list[str] = []
        for item in value:
            text = _WHITESPACE.sub(" ", str(item)).strip()
            key = text.casefold()
            if text and key not in seen:
                seen.add(key)
                cleaned.append(text)
        return cleaned[:24]
