"""Output guard for generated chat.

This is the last line between the model and the UI. It is deliberately
deterministic and cheap: no second model, no content blocklist. It drops
refusals, assistant narration, captions and over-long lines, then the service
fills the hole with an in-character template so the batch stays complete.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_WORDS = 12

_WHITESPACE = re.compile(r"\s+")
_WORD = re.compile(r"\S+")
_TRAILING_PERIOD = re.compile(r"\.+$")
_WRAPPING_QUOTES = re.compile(r"^[\s\"'`]+|[\s\"'`]+$")

REFUSAL_PATTERNS = (
    re.compile(r"^\s*as an ai\b", re.IGNORECASE),
    re.compile(
        r"\b(?:i|we)\s+(?:can'?t|cannot|won'?t|am unable|are unable|'m unable|'re unable|"
        r"'m not able|are not able)\b(?=.{0,80}\b(?:help|assist|provide|create|write|describe|"
        r"engage|do that|comply|answer|generate|continue|participate|depict|show)\b)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:i'?d rather not|i must decline|i have to decline|i'?m not comfortable)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bi'?m sorry,?\s+but\b", re.IGNORECASE),
    re.compile(r"\b(?:content policy|community guidelines|language model)\b", re.IGNORECASE),
)
MORALIZED_PATTERNS = (
    re.compile(
        r"\b(?:inappropriate|offensive|harmful|disrespectful|unethical|not appropriate|"
        r"not acceptable|not okay)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:please|you should|we should|must|need to)\s+(?:avoid|refrain|be mindful|respect|"
        r"consider|not)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:i'?m sorry|i apologi[sz]e)\b", re.IGNORECASE),
)
NARRATOR_PATTERNS = (
    re.compile(r"^(?:the|this|that)\s+(?:image|frame|scene|video|picture)\b", re.IGNORECASE),
    re.compile(r"^(?:we|you)\s+can\s+see\b", re.IGNORECASE),
    re.compile(r"^(?:it|this|the scene)\s+(?:appears|seems)\s+to\b", re.IGNORECASE),
    re.compile(r"\b(?:in|within)\s+(?:this|the)\s+(?:image|frame|scene|video)\b", re.IGNORECASE),
    re.compile(r"^a\s+(?:man|woman|person|player)\s+(?:is|are)\b", re.IGNORECASE),
    re.compile(
        r"^the\s+(?:player|man|woman|person|chef|streamer)\s+is\s+(?:currently\s+)?"
        r"(?:hiding|crouching|running|walking|standing|sitting|holding|looking|facing|"
        r"fighting|climbing|jumping|swimming|driving|eating|cooking|talking|stirring)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bthe image (?:shows|depicts)\b", re.IGNORECASE),
    re.compile(r"\b(?:appears|seems) to be\b", re.IGNORECASE),
)
ASSISTANT_PATTERNS = (
    re.compile(r"\b(?:as an ai|ai model|simulated|simulation|as a language model)\b", re.IGNORECASE),
    re.compile(r"\b(?:viewers|prompt|system message)\b", re.IGNORECASE),
)
META_PATTERNS = (
    re.compile(r"^\s*\d{1,2}:\d{2}\b"),
    re.compile(
        r"\b(?:precise timestamps|points out typos|remembrancer|soft surrealism|"
        r"soft surreal vibe|non-?sequitur observations|reacts with images-in-words)\b",
        re.IGNORECASE,
    ),
)


@dataclass(slots=True, frozen=True)
class GuardResult:
    text: str
    ok: bool
    reason: str = ""


def normalize_text(text: str) -> str:
    value = text.strip().strip("\"'`")
    value = value.replace("*", "").replace("`", "")
    value = _WHITESPACE.sub(" ", value)
    value = _TRAILING_PERIOD.sub("", value)
    return value.strip()


def classify(text: str) -> str:
    cleaned = normalize_text(text)
    if not cleaned:
        return "empty"
    for pattern in REFUSAL_PATTERNS:
        if pattern.search(cleaned):
            return "refusal"
    for pattern in MORALIZED_PATTERNS:
        if pattern.search(cleaned):
            return "moralized"
    for pattern in NARRATOR_PATTERNS:
        if pattern.search(cleaned):
            return "narrator"
    for pattern in ASSISTANT_PATTERNS:
        if pattern.search(cleaned):
            return "assistant"
    for pattern in META_PATTERNS:
        if pattern.search(cleaned):
            return "meta"
    if len(cleaned.split()) > MAX_WORDS:
        return "too_long"
    return "ok"


def check(text: str, *, persona_caps: bool = False) -> GuardResult:
    cleaned = normalize_text(text)
    if persona_caps and cleaned:
        cleaned = cleaned.upper()
    reason = classify(cleaned)
    if reason != "ok":
        return GuardResult(text=cleaned, ok=False, reason=reason)
    return GuardResult(text=cleaned, ok=True)


_DEDUPE_EMOTES = frozenset(
    {
        "lul",
        "kekw",
        "pog",
        "pogchamp",
        "monkas",
        "sadge",
        "copium",
        "pepehands",
        "catjam",
        "kreygasm",
        "biblethump",
        "weirdchamp",
        "5head",
        "ez",
        "clap",
        "o7",
        "gigachad",
        "heart",
        "w",
        "l",
    }
)


def dedupe_key(text: str) -> str:
    lowered = normalize_text(text).casefold().replace("<3", " heart ")
    base = re.sub(r"[^\w\s]", " ", lowered)
    words = [word for word in _WHITESPACE.sub(" ", base).split() if word not in _DEDUPE_EMOTES]
    # Pure punctuation/emote messages still need a stable key so the repeat
    # guard can see them.
    return " ".join(words).strip() or lowered.strip()


_TEMPLATES: dict[str, tuple[str, ...]] = {
    "hype": ("LETS GO", "W", "POG", "we are so back"),
    "meme": ("what am i looking at", "bro", "this is cinema", "[copypasta]"),
    "backseater": ("that was a mistake", "should have waited", "classic"),
    "wholesome": ("you got this", "gg", "nice one"),
    "angry": ("trash", "L", "uninstall"),
    "doomer": ("we're cooked", "this is so over", "it's joever"),
    "normie": ("what happened", "why", "is this good"),
    "lore-keeper": ("just like last stream", "the lore grows"),
    "emote-only": ("LUL", "KEKW", "Pog"),
    "parser": ("ratio", "typo", "grammar"),
    "lurker": ("W", "lol", "hm"),
    "incongruent": ("the moon is loud", "kettle noises", "sure"),
}


def template_for(archetype: str, index: int = 0) -> str:
    options = _TEMPLATES.get(archetype) or _TEMPLATES["normie"]
    return options[index % len(options)]


def word_count(text: str) -> int:
    return len(_WORD.findall(normalize_text(text)))
