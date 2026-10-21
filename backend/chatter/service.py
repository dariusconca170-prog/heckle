"""Batched chat generation (M2).

One model call produces one message per selected persona. The service never
lets a raw refusal, caption or assistant sentence through: rejected lines become
in-character templates, and a failed model call degrades to templates for the
whole batch. Timing is deliberately absent here; M3's scheduler decides when a
message is released.
"""

from __future__ import annotations

import random
import secrets
import time
from collections import deque
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .. import events
from ..llm.client import LLMClient, LLMError
from ..llm.json_repair import JSONRepairError, parse_string_array
from ..llm.prompting import build_chatter_messages
from ..perception.scene import SceneState
from .guard import MAX_WORDS, check, dedupe_key, template_for, word_count
from .memory import lore_reference, mentions_memory
from .personas import Persona, PersonaRoster

if TYPE_CHECKING:  # pragma: no cover - import cycle only for type checking
    from ..state import AppState


@dataclass(slots=True)
class ChatMessage:
    id: str
    handle: str
    archetype: str
    text: str
    source: str
    batch_id: str
    t: float
    created_at: float
    word_count: int
    dedupe_key: str = ""
    kind: str = "chat"
    reply_to: str | None = None
    reply_to_handle: str = ""
    reply_to_text: str = ""
    mentions: tuple[str, ...] = ()
    emotes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "handle": self.handle,
            "archetype": self.archetype,
            "text": self.text,
            "source": self.source,
            "batch_id": self.batch_id,
            "t": round(self.t, 3),
            "created_at": round(self.created_at, 3),
            "word_count": self.word_count,
            "kind": self.kind,
            "reply_to": self.reply_to,
            "reply_to_handle": self.reply_to_handle,
            "reply_to_text": self.reply_to_text,
            "mentions": list(self.mentions),
            "emotes": list(self.emotes),
        }


@dataclass(slots=True)
class BatchResult:
    batch_id: str
    scene: SceneState
    messages: list[ChatMessage]
    elapsed_s: float
    fallback: bool = False
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "scene": self.scene.model_dump(),
            "messages": [message.as_dict() for message in self.messages],
            "elapsed_s": round(self.elapsed_s, 3),
            "fallback": self.fallback,
            "error": self.error,
        }


class ChatterService:
    def __init__(
        self,
        state: AppState,
        *,
        client: LLMClient | None = None,
        roster: PersonaRoster | None = None,
        rng: random.Random | None = None,
        recent_limit: int = 50,
    ) -> None:
        self.state = state
        self.settings = state.settings
        self.roster = roster or PersonaRoster()
        self.rng = rng or random.Random()
        self.recent: deque[ChatMessage] = deque(maxlen=max(10, recent_limit))
        self._client = client
        self._owns_client = client is None
        self._enabled = client is not None or self.settings.model_profile.is_file()
        self.last_error = ""
        self._completions = 0
        self._tokens = 0
        self._decode_s = 0.0
        self._last_tok_s = 0.0
        self._last_elapsed_s = 0.0
        # Stable core handles are the only ones that can carry memory into a
        # prompt; tell persistence which of them keep lore.
        register = getattr(state, "register_roster", None)
        if callable(register):
            register(self.roster.personas)

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.close()
        self._client = None

    def reset(self) -> None:
        self.recent.clear()
        self.last_error = ""
        self._completions = 0
        self._tokens = 0
        self._decode_s = 0.0
        self._last_tok_s = 0.0
        self._last_elapsed_s = 0.0

    def metrics(self) -> dict[str, Any]:
        average = self._tokens / self._decode_s if self._decode_s > 0 else 0.0
        return {
            "enabled": self._enabled,
            "recent": len(self.recent),
            "last_error": self.last_error,
            "completions": self._completions,
            "completion_tokens": self._tokens,
            "last_tok_s": round(self._last_tok_s, 2),
            "avg_tok_s": round(average, 2),
            "last_elapsed_s": round(self._last_elapsed_s, 3),
        }

    def remember(self, message: ChatMessage) -> None:
        if any(existing.id == message.id for existing in self.recent):
            return
        self.recent.append(message)

    def choose(self, count: int) -> list[Persona]:
        return self.roster.choose(count, rng=self.rng)

    def recent_texts(self, limit: int = 30) -> list[str]:
        return [message.text for message in list(self.recent)[-max(1, limit) :]]

    async    def generate(
        self,
        scene: SceneState,
        *,
        count: int = 10,
        stream: dict[str, str] | None = None,
        personas: list[Persona] | tuple[Persona, ...] | None = None,
        publish: bool = True,
        memory: Mapping[str, Sequence[str]] | None = None,
        lore_handles: Collection[str] | None = None,
        reply_targets: Mapping[str, Mapping[str, str]] | None = None,
        poll: Mapping[str, Any] | None = None,
    ) -> BatchResult:
        if personas is None:
            count = min(max(1, int(count)), len(self.roster.personas))
            selected = self.choose(count)
        else:
            selected = list(personas)
            if not selected:
                raise ValueError("at least one persona is required")
            count = min(len(selected), len(self.roster.personas))
            selected = selected[:count]
        personas = selected
        batch_id = secrets.token_hex(4)
        started = time.perf_counter()
        descriptions = [
            event.detail or event.kind
            for event in self.state.timeline.events(limit=5)
            if event.detail or event.kind
        ]
        raw_messages: list[str] = []
        fallback = False
        error = ""

        if self._enabled:
            try:
                client = await self._get_client()
                prompt = build_chatter_messages(
                    scene=scene.model_dump(),
                    personas=[persona.prompt_dict() for persona in personas],
                    recent_events=descriptions,
                    recent_chat=self.recent_texts(),
                    stream=stream or self._stream_info(),
                    n=count,
                    prefix_handles=True,
                    memory=memory if memory is not None else self._memory_map(personas),
                    ask_lore=lore_handles,
                    reply_targets=reply_targets,
                    poll=poll,
                )
                completion = await client.chat(
                    prompt,
                    max_tokens=min(260, 40 + count * 18),
                    temperature=0.9,
                    top_p=0.95,
                    extra={"repeat_penalty": 1.06},
                )
                raw_messages = parse_string_array(completion.text)
                self._completions += 1
                self._tokens += int(completion.completion_tokens or 0)
                self._decode_s += float(completion.elapsed_s or 0.0)
                self._last_tok_s = float(completion.decode_tokens_per_s)
                self._last_elapsed_s = float(completion.elapsed_s)
            except (LLMError, JSONRepairError, ValueError, TypeError) as exc:
                fallback = True
                error = str(exc)[:400]
                self.last_error = error
            except Exception as exc:  # pragma: no cover - defensive
                fallback = True
                error = f"chat generation failed: {exc}"
                self.last_error = error

        if not raw_messages:
            fallback = True

        by_handle, unprefixed = _split_handle_prefixes(raw_messages, personas)
        messages: list[ChatMessage] = []
        recent_keys = {
            message.dedupe_key or dedupe_key(message.text) for message in self.recent
        }
        used_keys: set[str] = set()
        now = time.time()
        for index, persona in enumerate(personas):
            raw = by_handle.get(persona.handle)
            if raw is None:
                raw = unprefixed.pop(0) if unprefixed else ""
            guarded = check(raw, persona_caps=persona.caps)
            text = guarded.text
            source = "model"
            if not guarded.ok:
                text = ""
                source = "template"
            key = dedupe_key(text)
            forbidden = recent_keys | used_keys
            if not text or key in forbidden:
                source = "template"
                for attempt in range(12):
                    candidate = template_for(persona.archetype, index + attempt)
                    candidate_key = dedupe_key(candidate)
                    if candidate_key not in forbidden:
                        text = candidate
                        key = candidate_key
                        break
                else:
                    text = template_for(persona.archetype, index)
                    suffix = persona.emotes[0] if persona.emotes else "LUL"
                    text = f"{text} {suffix}"
                    key = dedupe_key(text)
            if word_count(text) > MAX_WORDS:
                text = " ".join(text.split()[:MAX_WORDS])
                key = dedupe_key(text)
            used_keys.add(key)
            message = ChatMessage(
                id=secrets.token_hex(6),
                handle=persona.handle,
                archetype=persona.archetype,
                text=text,
                source=source,
                batch_id=batch_id,
                t=scene.t,
                created_at=now,
                word_count=word_count(text),
                dedupe_key=key,
            )
            messages.append(message)
            self.recent.append(message)
            if publish:
                self.state.bus.publish_nowait(
                    events.CHAT_MESSAGE, {**message.as_dict(), "kind": "preview"}
                )

        if lore_handles:
            self._apply_lore_fallback(messages, lore_handles)

        return BatchResult(
            batch_id=batch_id,
            scene=scene,
            messages=messages,
            elapsed_s=time.perf_counter() - started,
            fallback=fallback,
            error=error,
        )

    def _apply_lore_fallback(
        self, messages: list[ChatMessage], lore_handles: Collection[str]
    ) -> None:
        """Guarantee one old-stream reference when it was explicitly requested.

        The model is asked to work memory into the line. When it ignores the
        request, the memory still has to surface, so it is said in the persona's
        own template voice rather than silently disappearing.
        """
        wanted = {handle for handle in lore_handles if handle}
        memory_candidates = getattr(self.state, "memory_candidates", None)
        if not wanted or not callable(memory_candidates):
            return
        for message in messages:
            if message.handle not in wanted or not message.text:
                continue
            candidates = list(memory_candidates(message.handle))
            if not candidates or mentions_memory(message.text, candidates):
                continue
            fallback = lore_reference(candidates)
            if not fallback or not check(fallback).ok:
                continue
            message.text = fallback
            message.source = "template"
            message.word_count = word_count(fallback)
            message.dedupe_key = dedupe_key(fallback)

    async def generate_preview(self, *, count: int = 10) -> BatchResult:
        scene = self.state.timeline.latest
        if scene is None:
            scene = SceneState(
                t=self.state.clock_snapshot().t_play,
                summary="the stream is between moments; chat is warming up",
                action="stream starting",
                objects=[],
                mood="neutral",
                surprise=0.2,
                confidence=0.2,
            )
        return await self.generate(scene, count=count, stream=self._stream_info())

    def has_memory(self, handle: str) -> bool:
        memory_candidates = getattr(self.state, "memory_candidates", None)
        if not callable(memory_candidates):
            return False
        return bool(memory_candidates(handle))

    def _memory_map(self, personas: Sequence[Persona]) -> dict[str, list[str]]:
        """Prompt-ready memory for the personalities in this batch."""
        if not self.settings.memory_enabled:
            return {}
        memory_for = getattr(self.state, "memory_for", None)
        if not callable(memory_for):
            return {}
        described: dict[str, list[str]] = {}
        for persona in personas:
            remembered = memory_for(persona.handle)
            if remembered:
                described[persona.handle] = list(remembered)
        return described

    def _stream_info(self) -> dict[str, str]:
        session = self.state.session
        runtime = getattr(self.state, "runtime", None)
        return {
            "title": session.title if session else "local stream",
            "category": session.category if session else "",
            "content_mode": runtime.content_mode() if runtime is not None else "edgy",
        }

    async def _get_client(self) -> LLMClient:
        if self._client is None:
            self._client = LLMClient(
                base_url=f"http://{self.settings.llama_host}:{self.settings.llama_port}",
                model=self._model_name(),
                timeout_s=self.settings.llm_timeout_s,
            )
            self._owns_client = True
        return self._client

    def _model_name(self) -> str:
        try:
            from ..llm.profile import load_profile

            return load_profile(str(self.settings.model_profile)).alias
        except Exception:
            return "twitch-sim"


def _split_handle_prefixes(
    raw_messages: list[str], personas: list[Persona]
) -> tuple[dict[str, str], list[str]]:
    """Pull optional ``handle: message`` prefixes off model output.

    The 4B usually follows the handle instruction, but it is not guaranteed.
    When it does not, the unprefixed lines are consumed in order as a fallback
    rather than being discarded.
    """
    by_handle: dict[str, str] = {}
    unprefixed: list[str] = []
    handles = {persona.handle.casefold(): persona.handle for persona in personas}
    for raw in raw_messages:
        text = raw.strip()
        prefix, separator, remainder = text.partition(":")
        canonical = handles.get(prefix.strip().casefold())
        if separator and canonical and remainder.strip():
            by_handle.setdefault(canonical, remainder.strip())
        else:
            unprefixed.append(text)
    return by_handle, unprefixed
