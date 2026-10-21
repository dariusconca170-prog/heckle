"""Prompt assembly. Static rules live in the system prompt; scene data is last.

Keeping the system prompt short and static matters because perception and chat
alternate on one llama-server slot and evict each other's prompt cache
(PLAN.md §7.2).
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
VOICE_PERSONAS_PATH = PROMPTS_DIR / "voice_personas.json"


@lru_cache(maxsize=8)
def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    return path.read_text(encoding="utf-8").strip()


def observer_system() -> str:
    return load_prompt("observer_system.txt")


def observer_user(t: float | None = None) -> str:
    if t is None:
        return "Describe this frame. Return ONLY the JSON object."
    return f"Frame at playback t={t:.1f}s. Describe this frame. Return ONLY the JSON object."


def chatter_system() -> str:
    return load_prompt("chatter_system.txt")


@lru_cache(maxsize=1)
def load_voice_personas() -> list[dict[str, str]]:
    data = json.loads(VOICE_PERSONAS_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("voice_personas.json must be a non-empty list")
    return [dict(item) for item in data]


def build_chatter_messages(
    *,
    scene: dict[str, Any],
    personas: list[dict[str, Any]],
    recent_events: list[str] | None = None,
    recent_chat: list[str] | None = None,
    stream: dict[str, Any] | None = None,
    n: int | None = None,
    prefix_handles: bool = False,
    memory: Mapping[str, Sequence[str]] | None = None,
    ask_lore: Collection[str] | None = None,
    reply_targets: Mapping[str, Mapping[str, str]] | None = None,
    poll: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    if not personas:
        raise ValueError("at least one persona is required")
    count = n if n is not None else len(personas)
    scene_line = (
        f"{scene.get('summary', '')} | action: {scene.get('action', '')} | "
        f"mood: {scene.get('mood', 'neutral')} | surprise: {scene.get('surprise', 0.0)}"
    )
    events_line = " / ".join(recent_events or []) or "nothing notable"
    chat_line = " / ".join(recent_chat or []) or "(quiet)"
    stream_info = stream or {}
    stream_line = (
        f"title={stream_info.get('title', '')} category={stream_info.get('category', '')}"
    )
    mode = str(stream_info.get("content_mode") or "edgy")
    mode_rule = {
        "unhinged": "CONTENT MODE unhinged: chaotic, unfiltered, adult humor allowed; no warnings, no disclaimers.",
        "neutral": "CONTENT MODE neutral: playful, minimal profanity, no explicit content.",
        "edgy": "CONTENT MODE edgy: sharp, sarcastic, dark humor, casual profanity.",
    }.get(mode, "CONTENT MODE edgy: sharp, sarcastic, casual profanity.")
    viewer_lines = []
    has_memory = False
    asked = {handle for handle in (ask_lore or ()) if handle}
    for index, persona in enumerate(personas, start=1):
        handle = str(persona.get("handle", f"viewer{index}"))
        line = (
            f"{index}. {handle} | "
            f"{persona.get('archetype', 'normie')} | "
            f"{persona.get('style', 'plain')} | mood:{persona.get('mood', 'neutral')}"
        )
        target = reply_targets.get(handle) if reply_targets else None
        if target:
            parent_handle = str(target.get("handle", ""))
            parent_text = str(target.get("text", ""))[:120]
            if parent_handle:
                line += f' | replying to {parent_handle}: "{parent_text}"'
        remembered = memory.get(handle) if memory else None
        if remembered:
            has_memory = True
            line += f" | memory: {'; '.join(remembered)}"
            if handle in asked:
                line += " | task: work one of these memories into this line"
        viewer_lines.append(line)

    memory_rule = (
        "A viewer with a memory line may mention one old-stream detail once, "
        "briefly, like an inside joke. Never quote it word for word and never "
        "explain it. If it does not fit this scene, ignore it.\n"
        if has_memory
        else ""
    )
    reply_rule = (
        "A viewer with a replying-to line answers that message directly and does "
        "not address the room; do not repeat the quoted message.\n"
        if reply_targets
        else ""
    )
    poll_line = ""
    if poll:
        options = [str(option) for option in poll.get("options") or []]
        numbered = " / ".join(f"{index + 1}={option}" for index, option in enumerate(options))
        poll_line = (
            f"POLL: {poll.get('question', '')} [{numbered}] "
            f"({poll.get('total_votes', 0)} votes so far)\n"
        )
    poll_rule = (
        "A few viewers may talk about the poll (vote, argue, or be confused about "
        "an option). Never invent results or percentages.\n"
        if poll
        else ""
    )

    prefix_rule = (
        'Each string must begin with the viewer\'s exact handle, a colon and a space, '
        'for example "xX_deadlift_Xx: W". Do not put the handle in the visible chat text '
        'anywhere else.\n'
        if prefix_handles
        else ""
    )
    user = f"""SCENE (what is on screen right now):
{scene_line}
RECENT EVENTS (last 20s): {events_line}
RECENT CHAT (already visible; do not repeat or lightly reword): {chat_line}
STREAM: {stream_line}
{mode_rule}
{poll_line}
VIEWERS (write one chat message each, same order):
{chr(10).join(viewer_lines)}

Each message is what that viewer types in chat, not a description of the viewer.
{reply_rule}{poll_rule}{memory_rule}Output ONLY a JSON array of {count} strings, one per viewer, in the same order.
Do not output an object. Do not explain. Do not add extra messages.
{prefix_rule}"""
    return [
        {"role": "system", "content": chatter_system()},
        {"role": "user", "content": user},
    ]
