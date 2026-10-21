"""Minimal OpenAI-compatible client for the pinned llama-server sidecar."""

from __future__ import annotations

import base64
import mimetypes
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx


class LLMError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass(slots=True)
class Completion:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    elapsed_s: float
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def decode_tokens_per_s(self) -> float:
        if self.elapsed_s <= 0 or self.completion_tokens <= 0:
            return 0.0
        return self.completion_tokens / self.elapsed_s

    @property
    def total_tokens_per_s(self) -> float:
        if self.elapsed_s <= 0:
            return 0.0
        return self.total_tokens / self.elapsed_s

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "text": self.text,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "elapsed_s": round(self.elapsed_s, 4),
            "decode_tokens_per_s": round(self.decode_tokens_per_s, 3),
            "total_tokens_per_s": round(self.total_tokens_per_s, 3),
            "finish_reason": self.finish_reason,
            "usage": self.usage,
        }


class LLMClient:
    """One in-flight request at a time is the app's rule; this client just talks."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        model: str = "twitch-sim",
        *,
        timeout_s: float = 180.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
            trust_env=False,
            headers={"Content-Type": "application/json"},
        )
        self._resolved_model: str | None = model if model and model != "auto" else None

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def health(self) -> bool:
        try:
            response = await self._client.get("/health", timeout=5.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def _resolve_model(self) -> str:
        if self._resolved_model:
            return self._resolved_model
        try:
            response = await self._client.get("/v1/models", timeout=10.0)
            response.raise_for_status()
            data = response.json()
            models = data.get("data", []) if isinstance(data, dict) else []
            if models:
                self._resolved_model = str(models[0].get("id"))
                return self._resolved_model
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            raise LLMError(f"cannot resolve llama-server model: {exc}") from exc
        raise LLMError("llama-server exposes no models")

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 200,
        temperature: float = 0.9,
        top_p: float = 0.95,
        top_k: int | None = None,
        seed: int | None = None,
        stop: list[str] | None = None,
        response_format: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Completion:
        model = await self._resolve_model()
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        if top_k is not None:
            payload["top_k"] = top_k
        if seed is not None:
            payload["seed"] = seed
        if stop:
            payload["stop"] = stop
        if response_format is not None:
            payload["response_format"] = response_format
        if extra:
            payload.update(extra)

        started = time.perf_counter()
        try:
            response = await self._client.post("/v1/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise LLMError(f"llama-server request failed: {exc}") from exc
        elapsed = time.perf_counter() - started
        if response.status_code >= 400:
            raise LLMError(
                f"llama-server returned HTTP {response.status_code}",
                status=response.status_code,
                body=response.text[:2000],
            )
        try:
            data = response.json()
            choice = data["choices"][0]
            message = choice.get("message", {})
            content = message.get("content", "")
            if isinstance(content, list):
                content = "".join(
                    str(part.get("text", "")) if isinstance(part, dict) else str(part)
                    for part in content
                )
            usage = data.get("usage") or {}
            return Completion(
                text=str(content).strip(),
                model=str(data.get("model", model)),
                prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                total_tokens=int(usage.get("total_tokens", 0) or 0),
                elapsed_s=elapsed,
                finish_reason=choice.get("finish_reason"),
                usage=usage,
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError(f"malformed llama-server response: {exc}", body=response.text[:2000]) from exc

    async def complete_vision(
        self,
        image: bytes | Path,
        prompt: str,
        *,
        system_prompt: str | None = None,
        mime: str | None = None,
        max_tokens: int = 120,
        temperature: float = 0.4,
        response_format: dict[str, Any] | None = None,
    ) -> Completion:
        if isinstance(image, Path):
            image_bytes = image.read_bytes()
            guessed, _ = mimetypes.guess_type(image.name)
            mime = mime or guessed or "image/jpeg"
        else:
            image_bytes = image
            mime = mime or "image/jpeg"
        data_uri = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        )
        return await self.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
        )
