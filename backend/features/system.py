"""VRAM / RAM / tok-s metrics for the studio HUD (M10).

nvidia-smi is optional: a machine without it (or without a driver) reports
``available: false`` and the UI hides the VRAM row instead of failing. Results
are cached for a couple of seconds because the HUD polls at that rate.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..chatter.service import ChatterService
    from ..dynamics.service import DynamicsService
    from ..state import AppState

try:
    import psutil
except Exception:  # pragma: no cover - psutil is a required dependency
    psutil = None  # type: ignore[assignment]


@dataclass(slots=True)
class GpuStats:
    available: bool = False
    name: str = ""
    total_mb: int = 0
    used_mb: int = 0
    free_mb: int = 0
    utilization: int = 0
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "name": self.name,
            "total_mb": self.total_mb,
            "used_mb": self.used_mb,
            "free_mb": self.free_mb,
            "utilization": self.utilization,
            "error": self.error,
        }


@dataclass(slots=True)
class SystemMetrics:
    state: AppState
    chatter: ChatterService
    dynamics: DynamicsService
    clips: Any = None
    cache_s: float = 2.0
    _gpu: GpuStats = field(default_factory=GpuStats)
    _gpu_at: float = 0.0
    _gpu_error: str = ""

    def snapshot(self) -> dict[str, Any]:
        gpu = self._gpu_cached()
        ram: dict[str, Any] = {"available": False}
        if psutil is not None:
            memory = psutil.virtual_memory()
            ram = {
                "available": True,
                "total_mb": int(memory.total / (1024 * 1024)),
                "used_mb": int(memory.used / (1024 * 1024)),
                "free_mb": int(memory.available / (1024 * 1024)),
                "percent": round(float(memory.percent), 1),
            }
        chat = self.chatter.metrics()
        dynamics = self.dynamics.metrics()
        session = self.state.session
        return {
            "gpu": gpu.as_dict(),
            "ram": ram,
            "chat": chat,
            "dynamics": {
                "rate": dynamics.get("rate"),
                "buffer_level": dynamics.get("buffer_level"),
                "target_buffer": dynamics.get("target_buffer"),
                "active_chatters": dynamics.get("active_chatters"),
                "replies": dynamics.get("replies"),
                "reply_rate": dynamics.get("reply_rate"),
                "arguments": dynamics.get("arguments"),
                "mentions": dynamics.get("mentions"),
            },
            "viewers": self.state.live_viewers or (session.viewer_count if session else 0),
            "session_id": session.id if session else None,
            "clips": self.clips.metrics() if self.clips is not None else None,
            "content_mode": (
                self.state.runtime.content_mode()
                if getattr(self.state, "runtime", None) is not None
                else "edgy"
            ),
        }

    def _gpu_cached(self) -> GpuStats:
        now = time.monotonic()
        if now - self._gpu_at < self.cache_s:
            return self._gpu
        self._gpu_at = now
        self._gpu = self._read_gpu()
        return self._gpu

    def _read_gpu(self) -> GpuStats:
        if not shutil.which("nvidia-smi"):
            return GpuStats(available=False, error="nvidia-smi not found")
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=1.5,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return GpuStats(available=False, error=str(exc)[:200])
        if result.returncode != 0 or not result.stdout.strip():
            return GpuStats(
                available=False,
                error=(result.stderr.strip() or "nvidia-smi returned no data")[:200],
            )
        line = result.stdout.strip().splitlines()[0]
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            return GpuStats(available=False, error="unexpected nvidia-smi output")
        try:
            total, used, free, utilization = (int(parts[index]) for index in (1, 2, 3, 4))
        except ValueError:
            return GpuStats(available=False, error="unexpected nvidia-smi numbers")
        return GpuStats(
            available=True,
            name=parts[0],
            total_mb=total,
            used_mb=used,
            free_mb=free,
            utilization=utilization,
        )


__all__ = ["GpuStats", "SystemMetrics"]
