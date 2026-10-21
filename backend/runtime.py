"""Runtime settings and presets (M10).

``config/default.toml`` is deployment configuration; this file is the small set
of knobs the studio screen is allowed to change while the app is running. It is
written to ``data/settings.json`` so a restart keeps the choice without editing
the checked-in config.

Only values that can be applied live are mutated on ``Settings``. Look-ahead is
the exception: it is stored here and applied to the next stream, because
changing it mid-stream would invalidate every pre-generated message.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CONTENT_MODES = ("unhinged", "edgy", "neutral")

PRESETS: dict[str, dict[str, Any]] = {
    "calm": {
        "label": "Calm",
        "content_mode": "neutral",
        "intensity": 0.7,
        "chat_target_buffer": 12,
        "lookahead_s": 5.5,
    },
    "balanced": {
        "label": "Balanced",
        "content_mode": "edgy",
        "intensity": 1.0,
        "chat_target_buffer": 20,
        "lookahead_s": 5.5,
    },
    "chaos": {
        "label": "Chaos",
        "content_mode": "unhinged",
        "intensity": 1.8,
        "chat_target_buffer": 28,
        "lookahead_s": 5.5,
    },
}

DEFAULTS: dict[str, Any] = {
    "preset": "balanced",
    "content_mode": "edgy",
    "intensity": 1.0,
    "chat_target_buffer": 20,
    "lookahead_s": 5.5,
    "model_path": "",
    "model_quant": "Q4_K_M",
    "tts_enabled": False,
    "mic_enabled": False,
}


@dataclass(slots=True)
class SettingsPatch:
    preset: str | None = None
    content_mode: str | None = None
    intensity: float | None = None
    chat_target_buffer: int | None = None
    lookahead_s: float | None = None
    model_path: str | None = None
    model_quant: str | None = None
    tts_enabled: bool | None = None
    mic_enabled: bool | None = None


class RuntimeSettings:
    """JSON-backed overrides with validation and atomic writes."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.values: dict[str, Any] = dict(DEFAULTS)
        self.saved_keys: set[str] = set()
        self.last_error = ""
        self.revision = 0
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        for key, value in data.items():
            if key in DEFAULTS:
                self.values[key] = value
                self.saved_keys.add(key)
        self.revision += 1
        try:
            self._validate(self.values)
        except ValueError as exc:
            self.last_error = str(exc)
            self.values = dict(DEFAULTS)

    def snapshot(self) -> dict[str, Any]:
        return {
            **self.values,
            "content_modes": list(CONTENT_MODES),
            "presets": [
                {"name": name, **config}
                for name, config in PRESETS.items()
            ],
            "path": str(self.path),
            "revision": self.revision,
            "last_error": self.last_error,
        }

    def update(self, patch: SettingsPatch) -> dict[str, Any]:
        candidate = dict(self.values)
        for key, value in asdict(patch).items():
            if value is not None:
                candidate[key] = value
        if patch.preset is not None:
            if patch.preset not in PRESETS:
                raise ValueError(f"unknown preset {patch.preset!r}")
            preset = PRESETS[patch.preset]
            candidate.update({k: v for k, v in preset.items() if k != "label"})
        self._validate(candidate)
        self.values = candidate
        self.revision += 1
        self._save()
        return self.snapshot()

    def apply_preset(self, name: str) -> dict[str, Any]:
        return self.update(SettingsPatch(preset=name))

    def content_mode(self) -> str:
        return str(self.values.get("content_mode") or "edgy")

    def intensity(self) -> float:
        return float(self.values.get("intensity") or 1.0)

    def _validate(self, values: dict[str, Any]) -> None:
        mode = values.get("content_mode")
        if mode not in CONTENT_MODES:
            raise ValueError(f"content_mode must be one of {CONTENT_MODES}")
        try:
            intensity = float(values.get("intensity"))
        except (TypeError, ValueError) as exc:
            raise ValueError("intensity must be a number") from exc
        if not 0.2 <= intensity <= 3.0:
            raise ValueError("intensity must be between 0.2 and 3.0")
        try:
            buffer = int(values.get("chat_target_buffer"))
        except (TypeError, ValueError) as exc:
            raise ValueError("chat_target_buffer must be an integer") from exc
        if not 4 <= buffer <= 60:
            raise ValueError("chat_target_buffer must be between 4 and 60")
        try:
            lookahead = float(values.get("lookahead_s"))
        except (TypeError, ValueError) as exc:
            raise ValueError("lookahead_s must be a number") from exc
        if not 0.0 <= lookahead <= 30.0:
            raise ValueError("lookahead_s must be between 0 and 30")
        quant = str(values.get("model_quant") or "")
        if not quant or len(quant) > 24:
            raise ValueError("model_quant is required")
        if not isinstance(values.get("tts_enabled"), bool):
            raise ValueError("tts_enabled must be a boolean")
        if not isinstance(values.get("mic_enabled"), bool):
            raise ValueError("mic_enabled must be a boolean")

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=".settings-", suffix=".json"
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(self.values, stream, indent=2, sort_keys=True)
                stream.write("\n")
            os.replace(temp_name, self.path)
            self.last_error = ""
        except OSError as exc:
            self.last_error = str(exc)[:400]
            with contextlib.suppress(OSError):
                os.unlink(temp_name)


__all__ = [
    "CONTENT_MODES",
    "DEFAULTS",
    "PRESETS",
    "RuntimeSettings",
    "SettingsPatch",
]
