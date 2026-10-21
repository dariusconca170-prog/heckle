"""Launch, probe and stop the pinned llama-server process.

The app never silently starts a different model. A profile is explicit; missing
files fail loudly with the download command. Startup logs are parsed for the
two facts that change the whole latency budget: whether the vision projector
landed on GPU, and the real slot count / per-slot context.
"""

from __future__ import annotations

import asyncio
import os
import re
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from ..config import PROJECT_ROOT
from .profile import ModelProfile


class ServerStartError(RuntimeError):
    pass


@dataclass(slots=True)
class ServerInfo:
    profile: str
    command: list[str]
    log_path: Path
    pid: int
    startup_s: float
    mmproj_on_gpu: bool | None
    slots: int | None
    ctx_per_slot: int | None

    def as_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "command": self.command,
            "log_path": str(self.log_path),
            "pid": self.pid,
            "startup_s": round(self.startup_s, 2),
            "mmproj_on_gpu": self.mmproj_on_gpu,
            "slots": self.slots,
            "ctx_per_slot": self.ctx_per_slot,
        }


_SLOT_RE = re.compile(r"n_slots\s*[:=]\s*(\d+)")
_CTX_SLOT_RE = re.compile(r"n_ctx_slot\s*[:=]\s*(\d+)")
_CTX_PER_SEQ_RE = re.compile(r"n_ctx_per_seq\s*[:=]\s*(\d+)")


class ServerManager:
    def __init__(
        self,
        profile: ModelProfile,
        *,
        executable: str | None = None,
        host: str | None = None,
        port: int | None = None,
        log_dir: Path | None = None,
        startup_timeout_s: float = 240.0,
    ) -> None:
        self.profile = profile
        self.executable = executable
        self.host = host or profile.host
        self.port = port or profile.port
        self.log_dir = log_dir or (PROJECT_ROOT / "data" / "logs")
        self.startup_timeout_s = startup_timeout_s
        self.process: subprocess.Popen[str] | None = None
        self.info: ServerInfo | None = None
        self._log_handle = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def model_name(self) -> str:
        return self.profile.alias

    def _command(self) -> list[str]:
        return self.profile.command(self.executable)

    async def start(self) -> ServerInfo:
        if self.process is not None and self.process.poll() is None:
            if self.info is not None:
                return self.info
            raise ServerStartError("server process exists but has no startup info")

        missing = self.profile.missing_files()
        if missing:
            listing = "\n".join(f"  - {path}" for path in missing)
            raise ServerStartError(
                "model files are missing:\n"
                f"{listing}\n"
                "Run scripts/fetch_models.ps1 (or .sh), or edit the paths in "
                f"config/models/{self.profile.name}.toml."
            )
        if _port_open(self.host, self.port):
            raise ServerStartError(
                f"{self.host}:{self.port} is already in use. Stop the other llama-server "
                "or start this profile on a different port."
            )

        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        log_path = self.log_dir / f"llama-server-{self.profile.name}-{stamp}.log"
        command = self._command()
        self._log_handle = log_path.open("w", encoding="utf-8", errors="replace")
        self._log_handle.write("$ " + " ".join(_shell_quote(part) for part in command) + "\n\n")
        self._log_handle.flush()

        kwargs: dict[str, object] = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            kwargs["start_new_session"] = True
        started = time.perf_counter()
        self.process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            **kwargs,
        )

        deadline = time.monotonic() + self.startup_timeout_s
        last_error = ""
        async with httpx.AsyncClient(timeout=3.0, trust_env=False) as client:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise ServerStartError(
                        f"llama-server exited with code {self.process.returncode} during startup.\n"
                        f"Log: {log_path}\n{self.tail_log(40)}"
                    )
                try:
                    response = await client.get(f"{self.base_url}/health")
                    if response.status_code == 200:
                        break
                    last_error = f"HTTP {response.status_code}"
                except httpx.HTTPError as exc:
                    last_error = str(exc)
                await asyncio.sleep(1.0)
            else:
                await self.stop()
                raise ServerStartError(
                    f"llama-server did not become healthy within {self.startup_timeout_s:.0f}s "
                    f"({last_error}).\nLog: {log_path}\n{self.tail_log(40)}"
                )

        startup_s = time.perf_counter() - started
        log_text = self._read_log()
        self.info = ServerInfo(
            profile=self.profile.name,
            command=command,
            log_path=log_path,
            pid=self.process.pid,
            startup_s=startup_s,
            mmproj_on_gpu=_parse_mmproj_on_gpu(log_text),
            slots=_first_int(_SLOT_RE, log_text),
            ctx_per_slot=_first_int(_CTX_SLOT_RE, log_text)
            or _first_int(_CTX_PER_SEQ_RE, log_text),
        )
        return self.info

    async def stop(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            if os.name == "nt":
                process.terminate()
            else:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    process.terminate()
            try:
                await asyncio.to_thread(process.wait, 20)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    process.kill()
                else:
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        process.kill()
                await asyncio.to_thread(process.wait, 5)
        self.process = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    async def __aenter__(self) -> ServerManager:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()

    def _read_log(self) -> str:
        if self.info is not None:
            path = self.info.log_path
        elif self._log_handle is not None:
            path = Path(self._log_handle.name)
        else:
            return ""
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def tail_log(self, lines: int = 40) -> str:
        text = self._read_log()
        if not text:
            return "(log is empty)"
        return "\n".join(text.splitlines()[-lines:])


def _first_int(pattern: re.Pattern[str], text: str) -> int | None:
    match = pattern.search(text)
    return int(match.group(1)) if match else None


def _parse_mmproj_on_gpu(log_text: str) -> bool | None:
    """Best-effort parse of llama.cpp's projector placement log lines."""
    result: bool | None = None
    for line in log_text.splitlines():
        low = line.lower()
        if not any(token in low for token in ("mmproj", "clip", "projector")):
            continue
        if "cpu" in low and not any(token in low for token in ("cuda", "gpu")):
            result = False
        elif any(token in low for token in ("cuda", "gpu")):
            result = True
    return result


def _port_open(host: str, port: int, timeout_s: float = 0.75) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _shell_quote(value: str) -> str:
    if not value or any(char.isspace() for char in value):
        return f'"{value}"'
    return value
