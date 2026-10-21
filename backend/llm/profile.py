"""Model profiles and the exact llama-server command for each candidate.

The winner is written by ``scripts/bakeoff.py`` to ``data/model_profile.json``.
Until then, these TOML files are inputs to the measurement, not decisions.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "config" / "models"


class ProfileError(ValueError):
    pass


@dataclass(slots=True)
class ModelProfile:
    name: str
    model_id: str
    model_path: Path
    mmproj_path: Path
    alias: str = "twitch-sim"
    ctx: int = 4096
    n_gpu_layers: int = 99
    n_cpu_moe: int | None = None
    threads: int = 8
    batch: int = 2048
    ubatch: int = 2048
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    flash_attn: bool = True
    jinja: bool = True
    disable_thinking: bool = True
    host: str = "127.0.0.1"
    port: int = 8080
    lookahead_s: float = 5.5
    executable: str | None = None
    extra_args: list[str] = field(default_factory=list)

    def missing_files(self) -> list[Path]:
        return [path for path in (self.model_path, self.mmproj_path) if not path.is_file()]

    def resolve_executable(self, override: str | None = None) -> str:
        candidate = override or self.executable or os.environ.get("TWITCHSIM_LLAMA_SERVER")
        if candidate:
            return candidate
        return "llama-server"

    def command(self, executable: str | None = None) -> list[str]:
        exe = self.resolve_executable(executable)
        cmd = [
            exe,
            "-m",
            str(self.model_path),
            "--mmproj",
            str(self.mmproj_path),
            "-c",
            str(self.ctx),
            "-np",
            "1",
            "-ngl",
            str(self.n_gpu_layers),
            "--alias",
            self.alias,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "-ctk",
            self.cache_type_k,
            "-ctv",
            self.cache_type_v,
            "-b",
            str(self.batch),
            "-ub",
            str(self.ubatch),
            "-t",
            str(self.threads),
        ]
        if self.n_cpu_moe is not None:
            cmd.extend(["-ncmoe", str(self.n_cpu_moe)])
        if self.flash_attn:
            # Current llama.cpp builds treat --flash-attn as an optional-value
            # flag; without an explicit value the following --jinja is parsed
            # as that value. Older builds accepted the bare flag, but this
            # project pins a release with the optional value (PLAN.md §7.4).
            cmd.extend(["--flash-attn", "on"])
        if self.jinja:
            cmd.append("--jinja")
        if self.disable_thinking:
            cmd.extend(["--chat-template-kwargs", '{"enable_thinking":false}'])
        cmd.extend(self.extra_args)
        return cmd

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_id": self.model_id,
            "model_path": str(self.model_path),
            "mmproj_path": str(self.mmproj_path),
            "command": self.command(),
            "lookahead_s": self.lookahead_s,
            "host": self.host,
            "port": self.port,
        }


def _resolve(path_value: str) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(path_value)))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_profile(name_or_path: str) -> ModelProfile:
    candidate = Path(name_or_path)
    if not candidate.is_file():
        candidate = MODELS_DIR / f"{name_or_path}.toml"
    if not candidate.is_file():
        raise ProfileError(
            f"model profile not found: {name_or_path} "
            f"(looked in {MODELS_DIR} and as a path)"
        )
    if candidate.suffix.lower() == ".json":
        return _profile_from_json(candidate)
    return _profile_from_toml(candidate)


def _profile_from_json(path: Path) -> ModelProfile:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "profile" in data and isinstance(data["profile"], dict):
        data = data["profile"]
    return _profile_from_mapping(data, path)


def _profile_from_toml(path: Path) -> ModelProfile:
    import tomllib

    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return _profile_from_mapping(data, path)


def _profile_from_mapping(data: dict[str, Any], path: Path) -> ModelProfile:
    try:
        return ModelProfile(
            name=str(data.get("name") or path.stem),
            model_id=str(data.get("model_id", "")),
            model_path=_resolve(str(data["model"])),
            mmproj_path=_resolve(str(data["mmproj"])),
            alias=str(data.get("alias", "twitch-sim")),
            ctx=int(data.get("ctx", 4096)),
            n_gpu_layers=int(data.get("n_gpu_layers", 99)),
            n_cpu_moe=(int(data["n_cpu_moe"]) if data.get("n_cpu_moe") is not None else None),
            threads=int(data.get("threads", 8)),
            batch=int(data.get("batch", 2048)),
            ubatch=int(data.get("ubatch", 2048)),
            cache_type_k=str(data.get("cache_type_k", "q8_0")),
            cache_type_v=str(data.get("cache_type_v", "q8_0")),
            flash_attn=bool(data.get("flash_attn", True)),
            jinja=bool(data.get("jinja", True)),
            disable_thinking=bool(data.get("disable_thinking", True)),
            host=str(data.get("host", "127.0.0.1")),
            port=int(data.get("port", 8080)),
            lookahead_s=float(data.get("lookahead_s", 5.5)),
            executable=(str(data["executable"]) if data.get("executable") else None),
            extra_args=[str(arg) for arg in data.get("extra_args", [])],
        )
    except KeyError as exc:
        raise ProfileError(f"{path}: missing required key {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise ProfileError(f"{path}: invalid profile: {exc}") from exc


def physical_core_default() -> int:
    """Best-effort physical core count; profiles can override."""
    try:
        import psutil

        return max(1, psutil.cpu_count(logical=False) or 8)
    except Exception:  # pragma: no cover - psutil is a dependency
        return max(1, (os.cpu_count() or 8) // 2)


if __name__ == "__main__":  # pragma: no cover - tiny debug helper
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m backend.llm.profile <name-or-path>")
    profile = load_profile(sys.argv[1])
    print(json.dumps(profile.as_dict(), indent=2))
