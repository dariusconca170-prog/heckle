#!/usr/bin/env python3
"""M0.5 bake-off harness.

This is the only script allowed to decide which model the app builds against.
It records real measurements; it never invents a winner. A run is *partial*
unless the app is open with an active playback session, ffmpeg is decoding, and
an STT load process is running for the whole 30 minutes.

Examples
--------
    # 9B: 30-minute full-load gate (start the app, open the player first)
    python scripts/bakeoff.py run --profile 9b

    # 35B challenger, same conditions
    python scripts/bakeoff.py run --profile 35b

    # Generate/rate blind voice samples, then select a winner
    python scripts/bakeoff.py voice --profile 9b --blind
    python scripts/bakeoff.py rate-voice --label <label>
    python scripts/bakeoff.py select
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import os
import platform
import random
import re
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
import psutil  # noqa: E402

from backend.llm.client import Completion, LLMClient, LLMError  # noqa: E402
from backend.llm.json_repair import JSONRepairError, parse_json, parse_string_array  # noqa: E402
from backend.llm.profile import ModelProfile, load_profile  # noqa: E402
from backend.llm.prompting import (  # noqa: E402
    build_chatter_messages,
    load_voice_personas,
    observer_system,
    observer_user,
)
from backend.llm.server_manager import ServerInfo, ServerManager, ServerStartError  # noqa: E402
from backend.perception.scene import SceneState  # noqa: E402

DATA_DIR = ROOT / "data"
BAKEOFF_DIR = DATA_DIR / "bakeoff"
BAKEOFF_MD = DATA_DIR / "bakeoff.md"
SELECTION_PATH = BAKEOFF_DIR / "selection.json"
MODEL_PROFILE_PATH = DATA_DIR / "model_profile.json"
VOICE_SCENES_PATH = ROOT / "config" / "voice_scenes.json"

FLOOR_TOK_S = 25.0
FLOOR_BATCH_SECONDS = 4.5
FLOOR_RAM_MB = 1024
FLOOR_VRAM_FREE_MB = 500
GATE_MINUTES = 30.0


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "sample"


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 3) if values else None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return round(ordered[index], 3)


def _slope(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / denominator


def latest_runs() -> dict[str, dict[str, Any]]:
    BAKEOFF_DIR.mkdir(parents=True, exist_ok=True)
    runs: dict[str, dict[str, Any]] = {}
    for path in sorted(BAKEOFF_DIR.glob("*-run-*.json"), key=lambda p: p.stat().st_mtime):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        profile = str(data.get("profile", ""))
        if profile:
            data["_path"] = str(path)
            runs[profile] = data
    return runs


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def voice_scores() -> dict[str, dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}
    for path in BAKEOFF_DIR.glob("voice_*.json"):
        data = load_json(path)
        if not data or data.get("score") is None:
            continue
        profile = data.get("profile")
        if not profile:
            continue
        data["_path"] = str(path)
        previous = scores.get(str(profile))
        if previous is None or str(data.get("rated_at", "")) >= str(previous.get("rated_at", "")):
            scores[str(profile)] = data
    return scores


def refusal_results() -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for path in BAKEOFF_DIR.glob("refusal_*.json"):
        data = load_json(path)
        if data and data.get("profile"):
            data["_path"] = str(path)
            results[str(data["profile"])] = data
    return results


def accuracy_results() -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for path in BAKEOFF_DIR.glob("accuracy_*.json"):
        data = load_json(path)
        if data and data.get("profile"):
            data["_path"] = str(path)
            results[str(data["profile"])] = data
    return results


# --------------------------------------------------------------------------
# Hardware / process sampling
# --------------------------------------------------------------------------


def nvidia_stats() -> dict[str, float | None]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        first = result.stdout.strip().splitlines()[0]
        total, used, free, util = [float(part.strip()) for part in first.split(",")]
        return {"total_mb": total, "used_mb": used, "free_mb": free, "util_pct": util}
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return {"total_mb": None, "used_mb": None, "free_mb": None, "util_pct": None}


def process_memory(pid: int) -> dict[str, float | None]:
    try:
        process = psutil.Process(pid)
        info = process.memory_info()
        return {
            "rss_mb": round(getattr(info, "rss", 0) / (1024 * 1024), 1),
            "pagefile_mb": round(getattr(info, "pagefile", 0) / (1024 * 1024), 1)
            if hasattr(info, "pagefile")
            else None,
            "page_faults": float(getattr(info, "num_page_faults", 0))
            if hasattr(info, "num_page_faults")
            else None,
        }
    except (psutil.Error, OSError):
        return {"rss_mb": None, "pagefile_mb": None, "page_faults": None}


def collect_sample(*, start_monotonic: float, server_pid: int, load: LoadGenerator) -> dict[str, Any]:
    gpu = nvidia_stats()
    memory = psutil.virtual_memory()
    process = process_memory(server_pid)
    app = load.check_app()
    return {
        "minute": round((time.monotonic() - start_monotonic) / 60.0, 3),
        "gpu_total_mb": gpu["total_mb"],
        "gpu_used_mb": gpu["used_mb"],
        "gpu_free_mb": gpu["free_mb"],
        "gpu_util_pct": gpu["util_pct"],
        "ram_available_mb": round(memory.available / (1024 * 1024), 1),
        "ram_used_mb": round(memory.used / (1024 * 1024), 1),
        "server_rss_mb": process["rss_mb"],
        "page_faults": process["page_faults"],
        "pagefile_mb": process["pagefile_mb"],
        "ffmpeg_alive": load.ffmpeg_alive(),
        "app_alive": app["health"],
        "app_session": app["session"],
        "stt_alive": load.stt_alive(),
    }


# --------------------------------------------------------------------------
# Full-app load
# --------------------------------------------------------------------------


@dataclass(slots=True)
class LoadStatus:
    ffmpeg: bool = False
    app: bool = False
    app_session: bool = False
    stt: bool = False

    @property
    def all(self) -> bool:
        return self.ffmpeg and self.app and self.app_session and self.stt

    def as_dict(self) -> dict[str, bool]:
        return {
            "ffmpeg": self.ffmpeg,
            "app": self.app,
            "app_session": self.app_session,
            "stt": self.stt,
            "all": self.all,
        }


class LoadGenerator:
    """ffmpeg decode + app/session probe + optional STT process."""

    def __init__(self, *, app_url: str | None, stt_command: str | None, media: Path | None) -> None:
        self.app_url = app_url.rstrip("/") if app_url else None
        self.stt_command = stt_command
        self.media = media
        self.ffmpeg: subprocess.Popen[bytes] | None = None
        self.stt: subprocess.Popen[bytes] | None = None
        self._client = httpx.Client(timeout=3.0, trust_env=False)

    def start(self) -> LoadStatus:
        if self.media and self.media.is_file():
            try:
                self.ffmpeg = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-loglevel",
                        "error",
                        "-stream_loop",
                        "-1",
                        "-i",
                        str(self.media),
                        "-vf",
                        "scale=1280:-2",
                        "-f",
                        "null",
                        "-",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError:
                self.ffmpeg = None
        if self.stt_command:
            try:
                tokens = shlex.split(self.stt_command, posix=os.name != "nt")
                self.stt = subprocess.Popen(
                    tokens,
                    cwd=str(ROOT),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except (OSError, ValueError):
                self.stt = None
        return self.status()

    def status(self) -> LoadStatus:
        app = self.check_app()
        return LoadStatus(
            ffmpeg=self.ffmpeg_alive(),
            app=app["health"],
            app_session=app["session"],
            stt=self.stt_alive(),
        )

    def ffmpeg_alive(self) -> bool:
        return self.ffmpeg is not None and self.ffmpeg.poll() is None

    def stt_alive(self) -> bool:
        return self.stt is not None and self.stt.poll() is None

    def check_app(self) -> dict[str, bool]:
        if not self.app_url:
            return {"health": False, "session": False}
        health = False
        session = False
        try:
            response = self._client.get(f"{self.app_url}/health")
            health = response.status_code == 200
        except httpx.HTTPError:
            pass
        if health:
            try:
                response = self._client.get(f"{self.app_url}/api/session")
                session = response.status_code == 200
            except httpx.HTTPError:
                pass
        return {"health": health, "session": session}

    def stop(self) -> None:
        for process in (self.ffmpeg, self.stt):
            if process is not None and process.poll() is None:
                with contextlib.suppress(OSError):
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(OSError):
                        process.kill()
        self.ffmpeg = None
        self.stt = None
        self._client.close()


# --------------------------------------------------------------------------
# Model calls
# --------------------------------------------------------------------------


def prepare_frame(media: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    filters = "scale='min(512,iw)':-2"
    commands = []
    if media.is_file():
        commands.append(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-ss",
                "10",
                "-i",
                str(media),
                "-frames:v",
                "1",
                "-vf",
                filters,
                str(destination),
            ]
        )
        commands.append(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-ss",
                "0",
                "-i",
                str(media),
                "-frames:v",
                "1",
                "-vf",
                filters,
                str(destination),
            ]
        )
    commands.append(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=512x288:rate=1",
            "-frames:v",
            "1",
            str(destination),
        ]
    )
    for command in commands:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode == 0 and destination.is_file() and destination.stat().st_size > 0:
            return destination
    raise RuntimeError("could not extract or synthesize a benchmark frame (is ffmpeg installed?)")


def _vision_valid(text: str) -> bool:
    try:
        data = parse_json(text)
        SceneState(t=0.0, **data)
        return True
    except (JSONRepairError, ValueError, TypeError):
        return False


def _chat_valid(text: str) -> int:
    try:
        messages = parse_string_array(text)
    except JSONRepairError:
        return 0
    return len([message for message in messages if message.strip()])


async def run_vision(
    client: LLMClient, image: bytes, *, max_tokens: int = 90
) -> tuple[Completion, bool]:
    completion = await client.complete_vision(
        image,
        observer_user(),
        system_prompt=observer_system(),
        max_tokens=max_tokens,
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return completion, _vision_valid(completion.text)


async def run_chat_batch(
    client: LLMClient,
    *,
    scene: dict[str, Any],
    personas: list[dict[str, Any]],
    recent_chat: list[str] | None = None,
) -> tuple[Completion, int]:
    messages = build_chatter_messages(
        scene=scene,
        personas=personas,
        recent_events=["", "a quiet moment"],
        recent_chat=recent_chat,
        stream={"title": "late night run", "category": "variety"},
        n=len(personas),
    )
    completion = await client.chat(
        messages,
        max_tokens=180,
        temperature=0.9,
        top_p=0.95,
        stop=["\n\n\n"],
    )
    return completion, _chat_valid(completion.text)


@dataclass(slots=True)
class CallRecord:
    kind: str
    minute: float
    elapsed_s: float
    completion_tokens: int
    tok_s: float
    valid: bool | int
    text_preview: str = ""
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "minute": round(self.minute, 3),
            "elapsed_s": round(self.elapsed_s, 3),
            "completion_tokens": self.completion_tokens,
            "tok_s": round(self.tok_s, 3),
            "valid": self.valid,
            "text_preview": self.text_preview,
            "error": self.error,
        }


async def throughput_phase(
    client: LLMClient,
    *,
    image: bytes,
    scene: dict[str, Any],
    personas: list[dict[str, Any]],
    runs: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    vision_results: list[dict[str, Any]] = []
    chat_results: list[dict[str, Any]] = []
    for index in range(runs):
        completion, valid = await run_vision(client, image)
        vision_results.append(
            {
                "run": index + 1,
                **completion.as_dict(),
                "json_valid": valid,
            }
        )
        completion, valid_count = await run_chat_batch(
            client,
            scene=scene,
            personas=personas,
            recent_chat=[message for result in chat_results[-1:] for message in [result["text"]]],
        )
        chat_results.append(
            {
                "run": index + 1,
                **completion.as_dict(),
                "messages_valid": valid_count,
            }
        )
    return vision_results, chat_results


async def sampling_loop(
    stop: asyncio.Event,
    *,
    start_monotonic: float,
    server_pid: int,
    load: LoadGenerator,
    interval_s: float,
    samples: list[dict[str, Any]],
) -> None:
    while not stop.is_set():
        try:
            sample = await asyncio.to_thread(
                collect_sample,
                start_monotonic=start_monotonic,
                server_pid=server_pid,
                load=load,
            )
            samples.append(sample)
        except Exception:  # sampling must never kill the run
            pass
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=interval_s)


async def sustained_phase(
    client: LLMClient,
    *,
    duration_min: float,
    image: bytes,
    scene: dict[str, Any],
    personas: list[dict[str, Any]],
    server_pid: int,
    load: LoadGenerator,
    samples: list[dict[str, Any]],
    sample_interval_s: float,
    allow_early_abort: bool,
) -> tuple[list[CallRecord], str | None]:
    start = time.monotonic()
    end = start + duration_min * 60.0
    records: list[CallRecord] = []
    stop_sampling = asyncio.Event()
    sampler = asyncio.create_task(
        sampling_loop(
            stop_sampling,
            start_monotonic=start,
            server_pid=server_pid,
            load=load,
            interval_s=sample_interval_s,
            samples=samples,
        ),
        name="bakeoff-sampler",
    )
    abort_reason: str | None = None
    recent_chat: list[str] = []
    try:
        call_index = 0
        while time.monotonic() < end:
            minute = (time.monotonic() - start) / 60.0
            try:
                if call_index % 4 == 0:
                    completion, valid = await run_vision(client, image)
                    records.append(
                        CallRecord(
                            kind="vision",
                            minute=minute,
                            elapsed_s=completion.elapsed_s,
                            completion_tokens=completion.completion_tokens,
                            tok_s=completion.decode_tokens_per_s,
                            valid=valid,
                            text_preview=completion.text[:240],
                        )
                    )
                else:
                    completion, valid_count = await run_chat_batch(
                        client,
                        scene=scene,
                        personas=personas,
                        recent_chat=recent_chat,
                    )
                    try:
                        recent_chat = parse_string_array(completion.text)[-3:]
                    except JSONRepairError:
                        recent_chat = []
                    records.append(
                        CallRecord(
                            kind="chat",
                            minute=minute,
                            elapsed_s=completion.elapsed_s,
                            completion_tokens=completion.completion_tokens,
                            tok_s=completion.decode_tokens_per_s,
                            valid=valid_count,
                            text_preview=completion.text[:240],
                        )
                    )
            except LLMError as exc:
                records.append(
                    CallRecord(
                        kind="error",
                        minute=minute,
                        elapsed_s=0.0,
                        completion_tokens=0,
                        tok_s=0.0,
                        valid=False,
                        error=str(exc),
                    )
                )
                abort_reason = f"model call failed: {exc}"
                break
            call_index += 1

            if allow_early_abort and minute >= 5.0:
                chat_records = [record for record in records if record.kind == "chat"]
                recent = [record.tok_s for record in chat_records if record.minute >= minute - 5.0]
                available = [
                    float(sample["ram_available_mb"])
                    for sample in samples
                    if sample.get("ram_available_mb") is not None
                ]
                if recent and _median(recent) is not None and (_median(recent) or 0) < FLOOR_TOK_S:
                    abort_reason = (
                        f"early abort at minute {minute:.1f}: chat tok/s "
                        f"{_median(recent):.1f} < {FLOOR_TOK_S}"
                    )
                    break
                if available and min(available) < 1536:
                    abort_reason = (
                        f"early abort at minute {minute:.1f}: only "
                        f"{min(available):.0f} MB RAM available (< 1536 MB)"
                    )
                    break
    finally:
        stop_sampling.set()
        with contextlib.suppress(asyncio.CancelledError):
            await sampler

    # One final sample after the load ends, for the trend.
    with contextlib.suppress(Exception):
        samples.append(
            await asyncio.to_thread(
                collect_sample,
                start_monotonic=start,
                server_pid=server_pid,
                load=load,
            )
        )
    return records, abort_reason


# --------------------------------------------------------------------------
# Metrics / report
# --------------------------------------------------------------------------


def summarize_run(
    *,
    profile: ModelProfile,
    info: ServerInfo,
    args: argparse.Namespace,
    started_at: str,
    ended_at: str,
    actual_min: float,
    throughput_vision: list[dict[str, Any]],
    throughput_chat: list[dict[str, Any]],
    sustained: list[CallRecord],
    samples: list[dict[str, Any]],
    load_status: LoadStatus,
    abort_reason: str | None,
    no_crash: bool,
) -> dict[str, Any]:
    chat_through = [float(result["decode_tokens_per_s"]) for result in throughput_chat]
    chat_sustained = [record for record in sustained if record.kind == "chat" and record.valid]
    final_chat = [record.tok_s for record in chat_sustained if record.minute >= max(0, actual_min - 5)]
    if not final_chat:
        final_chat = [record.tok_s for record in chat_sustained[-10:]]
    tok_end = _median(final_chat)
    tok_p50 = _median([record.tok_s for record in chat_sustained])
    tok_p95 = _percentile([record.tok_s for record in chat_sustained], 0.95)
    batch_times = [float(result["elapsed_s"]) for result in throughput_chat]
    batch_p95 = _percentile(batch_times, 0.95)

    vram_peak = max(
        (float(s["gpu_used_mb"]) for s in samples if s.get("gpu_used_mb") is not None),
        default=None,
    )
    vram_free_min = min(
        (float(s["gpu_free_mb"]) for s in samples if s.get("gpu_free_mb") is not None),
        default=None,
    )
    ram_available = [
        float(s["ram_available_mb"]) for s in samples if s.get("ram_available_mb") is not None
    ]
    ram_min = min(ram_available, default=None)
    ram_end = ram_available[-1] if ram_available else None

    page_fault_samples = [
        (float(s["minute"]), float(s["page_faults"]))
        for s in samples
        if s.get("page_faults") is not None
    ]
    page_fault_slope = _slope(
        [item[0] for item in page_fault_samples], [item[1] for item in page_fault_samples]
    )
    pagefile_samples = [
        (float(s["minute"]), float(s["pagefile_mb"]))
        for s in samples
        if s.get("pagefile_mb") is not None
    ]
    pagefile_growth = (
        round(pagefile_samples[-1][1] - pagefile_samples[0][1], 1)
        if len(pagefile_samples) >= 2
        else None
    )

    full_load_ok = load_status.all
    duration_ok = actual_min + 0.1 >= max(GATE_MINUTES, float(args.duration_min))
    tok_ok = tok_end is not None and tok_end >= FLOOR_TOK_S
    batch_ok = batch_p95 is not None and batch_p95 <= FLOOR_BATCH_SECONDS
    vram_ok = vram_free_min is not None and vram_free_min >= FLOOR_VRAM_FREE_MB
    ram_ok = ram_min is not None and ram_min >= FLOOR_RAM_MB
    page_fault_ok = page_fault_slope is None or page_fault_slope < 5000
    passed = bool(
        full_load_ok
        and duration_ok
        and tok_ok
        and batch_ok
        and vram_ok
        and ram_ok
        and page_fault_ok
        and no_crash
        and abort_reason is None
    )
    return {
        "profile": profile.name,
        "model_id": profile.model_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_min": round(actual_min, 3),
        "command": profile.command(),
        "server": info.as_dict(),
        "settings": {
            "runs": args.runs,
            "duration_min_requested": args.duration_min,
            "sample_interval_s": args.sample_interval,
            "media": str(args.media) if args.media else None,
            "app_url": args.app_url or None,
            "stt_load": args.stt_load or None,
        },
        "full_load": load_status.as_dict(),
        "throughput": {
            "vision": throughput_vision,
            "chat": throughput_chat,
            "chat_p50_tok_s": _median(chat_through),
            "chat_p95_tok_s": _percentile(chat_through, 0.95),
            "chat_batch_p95_s": batch_p95,
        },
        "summary": {
            "tok_s_end": tok_end,
            "tok_s_p50": tok_p50,
            "tok_s_p95": tok_p95,
            "batch8_p95_s": batch_p95,
            "vram_peak_mb": vram_peak,
            "vram_free_min_mb": vram_free_min,
            "ram_available_min_mb": ram_min,
            "ram_available_end_mb": ram_end,
            "page_faults_per_min": round(page_fault_slope, 1) if page_fault_slope is not None else None,
            "pagefile_growth_mb": pagefile_growth,
            "sample_count": len(samples),
            "calls": len(sustained),
            "chat_calls": len(chat_sustained),
        },
        "floor": {
            "full_load_ok": full_load_ok,
            "duration_ok": duration_ok,
            "tok_s_ok": tok_ok,
            "batch_ok": batch_ok,
            "vram_ok": vram_ok,
            "ram_ok": ram_ok,
            "page_fault_ok": page_fault_ok,
            "no_crash": no_crash,
            "pass": passed,
        },
        "abort_reason": abort_reason,
        "samples": samples,
        "sustained": [record.as_dict() for record in sustained],
    }


def _fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.1f}{suffix}"
    return f"{value}{suffix}"


def render_bakeoff_md() -> None:
    runs = latest_runs()
    selection = load_json(SELECTION_PATH)
    voice = voice_scores()
    refusals = refusal_results()
    accuracy = accuracy_results()

    winner = selection.get("winner") if selection else None
    provisional = bool(selection and selection.get("provisional"))
    if provisional:
        status = "PROVISIONAL — M0.5 waived; not a measured winner"
    elif winner:
        status = "COMPLETE"
    else:
        status = "INCOMPLETE — M1 is blocked"
    lines = [
        "# M0.5 model bake-off",
        "",
        f"- Machine: `{platform.platform()}`",
        f"- Python: `{platform.python_version()}`",
        f"- Updated: `{utc_now()}`",
        f"- Status: **{status}**",
        f"- Winner: **{winner or 'not selected'}{' (provisional)' if provisional else ''}**",
        "",
        "The gate from `PLAN.md` §7.4: 30 minutes under full app load (browser + active",
        "playback session, ffmpeg decoding, STT running), ≥ 25 tok/s at minute 30,",
        "8-message batch p95 ≤ 4.5 s, ≥ 1 GB RAM free, ≥ 500 MB VRAM free, no page-fault",
        "growth, no crash. A model that misses any line is out. If both pass, the higher",
        "blind chat-voice score wins; a tie goes to the 9B.",
        "",
        "## Runs",
        "",
        "| Profile | Started | Duration | Full load | tok/s @ end | Chat batch p95 | VRAM peak | VRAM free min | RAM free min | mmproj | Floor | Voice | Refusals | Accuracy | Notes |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in ("9b", "35b", "4b"):
        run = runs.get(name)
        if not run:
            continue
        summary = run.get("summary", {})
        floor = run.get("floor", {})
        server = run.get("server") or {}
        full_load = run.get("full_load") or {}
        vscore = voice.get(name, {}).get("score")
        refusal = refusals.get(name, {}).get("summary", {})
        acc = accuracy.get(name, {}).get("score")
        if run.get("startup_failure"):
            note = "launch failed"
        elif run.get("abort_reason"):
            note = "aborted"
        else:
            note = "—"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{name}`",
                    str(run.get("started_at", "—")),
                    _fmt(run.get("duration_min"), " min"),
                    "yes" if full_load.get("all") else "**no**",
                    _fmt(summary.get("tok_s_end")),
                    _fmt(summary.get("batch8_p95_s"), " s"),
                    _fmt(summary.get("vram_peak_mb"), " MB"),
                    _fmt(summary.get("vram_free_min_mb"), " MB"),
                    _fmt(summary.get("ram_available_min_mb"), " MB"),
                    {True: "GPU", False: "**CPU**", None: "?"}.get(server.get("mmproj_on_gpu"), "?"),
                    "**PASS**" if floor.get("pass") else "FAIL",
                    _fmt(vscore),
                    (
                        "pass"
                        if refusal.get("pass")
                        else f"{refusal.get('refused', '—')} refused / {refusal.get('moralized', '—')} moralized"
                    ),
                    _fmt(acc),
                    note,
                ]
            )
            + " |"
        )
    if not runs:
        lines.append("| — | No measurement runs recorded | | | | | | | | | | | | | |")

    failures = [
        (name, run.get("abort_reason"))
        for name, run in runs.items()
        if run.get("abort_reason") or run.get("startup_failure")
    ]
    if failures:
        lines.extend(["", "## Failure details", ""])
        for name, reason in failures:
            lines.append(f"- `{name}`: {reason or 'llama-server did not reach readiness'}")

    lines.extend(["", "## Run files", ""])
    for name, run in sorted(runs.items()):
        lines.append(f"- `{name}`: `{run.get('_path', '')}`")
    if not runs:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Evidence still required",
            "",
            "- `scripts/refusal_audit.py` results for both 9B and 35B (0 refused, ≤ 2 moralized).",
            "- blind chat-voice rating for both (`bakeoff.py voice` then `rate-voice`).",
            "- 30-frame perception accuracy for both (`bakeoff.py accuracy`, ≥ 80 %).",
            "- `bakeoff.py select` refuses to name a winner until every line above exists.",
            "",
        ]
    )
    if selection:
        lines.extend(
            [
                "## Decision",
                "",
                f"**{selection.get('winner')}** — {selection.get('rationale', '')}",
                "",
                *(
                    [
                        "> Provisional: M0.5 was explicitly waived by the user. These are "
                        "not measured gate results and must be replaced by `bakeoff.py select` "
                        "after real 9B/35B/4B evidence exists.",
                        "",
                    ]
                    if provisional
                    else []
                ),
                "Winning launch command:",
                "",
                "```bash",
                " ".join(selection.get("command", [])),
                "```",
                "",
            ]
        )
        lookahead = selection.get("lookahead_s")
        if lookahead is not None:
            lines.extend(
                [
                    f"Look-ahead for the app: **{lookahead:.1f} s**. The app reads "
                    f"`data/model_profile.json`; do not re-tune on launch.",
                    "",
                ]
            )
    BAKEOFF_DIR.mkdir(parents=True, exist_ok=True)
    BAKEOFF_MD.write_text("\n".join(lines), encoding="utf-8")


def estimate_lookahead(run: dict[str, Any]) -> float:
    """t_scene + t_batch + 2 s, computed from this run's real numbers."""
    vision = [float(item["elapsed_s"]) for item in run.get("throughput", {}).get("vision", [])]
    chat = [float(item["elapsed_s"]) for item in run.get("throughput", {}).get("chat", [])]
    scene_time = _median(vision) or 0.0
    batch_time = _median(chat) or 0.0
    return round(max(3.0, min(12.0, scene_time + batch_time + 2.0)), 1)


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


async def cmd_run(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    BAKEOFF_DIR.mkdir(parents=True, exist_ok=True)
    if not profile.model_path.is_file() or not profile.mmproj_path.is_file():
        missing = "\n".join(f"  - {path}" for path in profile.missing_files())
        print(f"missing model files:\n{missing}")
        print("Run scripts/fetch_models.ps1 (or .sh), or edit config/models/<name>.toml.")
        return 2

    media = Path(args.media).expanduser().resolve() if args.media else None
    if media is not None and not media.is_file():
        print(f"warning: media not found: {media}; ffmpeg load will be absent")
        media_for_load = None
    else:
        media_for_load = media
    frame_path = prepare_frame(media or Path(), BAKEOFF_DIR / "work" / "frame.jpg")
    image = frame_path.read_bytes()
    personas = load_voice_personas()
    scenes = json.loads(VOICE_SCENES_PATH.read_text(encoding="utf-8"))
    scene = random.Random(7).choice(scenes)

    load = LoadGenerator(
        app_url=args.app_url or None,
        stt_command=args.stt_load or None,
        media=media_for_load,
    )
    load_status = load.start()
    if not args.allow_partial and not load_status.all:
        load.stop()
        print(
            "refusing to start a gate run without full app load.\n"
            f"  ffmpeg decoding: {load_status.ffmpeg}\n"
            f"  app /health: {load_status.app}\n"
            f"  active playback session: {load_status.app_session}\n"
            f"  STT load process: {load_status.stt}\n"
            "Start the app, open a video so /api/session is active, and pass a real\n"
            "STT workload with --stt-load. Use --allow-partial only for harness smoke tests."
        )
        return 2

    manager = ServerManager(
        profile,
        executable=args.llama_server or None,
        host=args.host or None,
        port=args.port or None,
        startup_timeout_s=args.startup_timeout,
    )
    client = LLMClient(manager.base_url, manager.model_name, timeout_s=args.timeout)
    samples: list[dict[str, Any]] = []
    throughput_vision: list[dict[str, Any]] = []
    throughput_chat: list[dict[str, Any]] = []
    sustained: list[CallRecord] = []
    abort_reason: str | None = None
    started_at = utc_now()
    started = time.monotonic()
    no_crash = True
    info: ServerInfo | None = None

    try:
        print(f"[bakeoff] starting {profile.name} ...")
        info = await manager.start()
        print(
            f"[bakeoff] ready in {info.startup_s:.1f}s; mmproj on GPU: {info.mmproj_on_gpu}; "
            f"slots={info.slots} ctx/slot={info.ctx_per_slot}"
        )
        if info.mmproj_on_gpu is False:
            print("[bakeoff] WARNING: vision projector is on CPU; perception latency will suffer")
        print(f"[bakeoff] full load: {load.status().as_dict()}")
        if not await client.health():
            raise RuntimeError("llama-server does not answer /health after startup")

        print("[bakeoff] warmup ...")
        await run_vision(client, image, max_tokens=30)
        await run_chat_batch(client, scene=scene, personas=personas)

        print(f"[bakeoff] throughput: {args.runs} vision + {args.runs} chat batches ...")
        throughput_vision, throughput_chat = await throughput_phase(
            client,
            image=image,
            scene=scene,
            personas=personas,
            runs=args.runs,
        )
        print(
            f"[bakeoff] chat p50 {statistics.median([r['decode_tokens_per_s'] for r in throughput_chat]):.1f} tok/s, "
            f"p95 batch {_percentile([r['elapsed_s'] for r in throughput_chat], 0.95):.2f}s"
        )

        print(f"[bakeoff] sustained full-load loop: {args.duration_min} min ...")
        sustained, abort_reason = await sustained_phase(
            client,
            duration_min=args.duration_min,
            image=image,
            scene=scene,
            personas=personas,
            server_pid=info.pid,
            load=load,
            samples=samples,
            sample_interval_s=args.sample_interval,
            allow_early_abort=not args.no_early_abort,
        )
        no_crash = manager.process is not None and manager.process.poll() is None
        if abort_reason:
            print(f"[bakeoff] {abort_reason}")
    except (ServerStartError, LLMError, RuntimeError, OSError) as exc:
        no_crash = False
        abort_reason = str(exc)
        print(f"[bakeoff] failed: {exc}")
    finally:
        final_load_status = load.status()
        load.stop()
        await client.close()
        await manager.stop()
        if info is not None:
            actual_min = (time.monotonic() - started) / 60.0
            ended_at = utc_now()
            result = summarize_run(
                profile=profile,
                info=info,
                args=args,
                started_at=started_at,
                ended_at=ended_at,
                actual_min=actual_min,
                throughput_vision=throughput_vision,
                throughput_chat=throughput_chat,
                sustained=sustained,
                samples=samples,
                load_status=final_load_status,
                abort_reason=abort_reason,
                no_crash=no_crash,
            )
            result["run_file"] = str(BAKEOFF_DIR / f"{profile.name}-run-{stamp()}.json")
            path = Path(result["run_file"])
            path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            render_bakeoff_md()
            summary = result["summary"]
            floor = result["floor"]
            print(f"[bakeoff] wrote {path}")
            print(
                f"[bakeoff] result: tok/s@end={_fmt(summary.get('tok_s_end'))}, "
                f"batch8 p95={_fmt(summary.get('batch8_p95_s'), 's')}, "
                f"VRAM free min={_fmt(summary.get('vram_free_min_mb'), 'MB')}, "
                f"RAM free min={_fmt(summary.get('ram_available_min_mb'), 'MB')}, "
                f"floor={'PASS' if floor.get('pass') else 'FAIL'}"
            )
            if not floor.get("pass"):
                print("[bakeoff] gate not met; M1 remains blocked.")
    if info is None:
        # A failed launch is still evidence: record it so the gate does not
        # pretend the model was never tried. It can never pass the floor.
        actual_min = (time.monotonic() - started) / 60.0
        result = {
            "profile": profile.name,
            "model_id": profile.model_id,
            "started_at": started_at,
            "ended_at": utc_now(),
            "duration_min": round(actual_min, 3),
            "command": profile.command(args.llama_server or None),
            "server": None,
            "startup_failure": True,
            "settings": {
                "runs": args.runs,
                "duration_min_requested": args.duration_min,
                "sample_interval_s": args.sample_interval,
                "media": str(args.media) if args.media else None,
                "app_url": args.app_url or None,
                "stt_load": args.stt_load or None,
            },
            "full_load": final_load_status.as_dict(),
            "throughput": {
                "vision": throughput_vision,
                "chat": throughput_chat,
                "chat_p50_tok_s": None,
                "chat_p95_tok_s": None,
                "chat_batch_p95_s": None,
            },
            "summary": {
                "tok_s_end": None,
                "tok_s_p50": None,
                "tok_s_p95": None,
                "batch8_p95_s": None,
                "vram_peak_mb": None,
                "vram_free_min_mb": None,
                "ram_available_min_mb": None,
                "ram_available_end_mb": None,
                "page_faults_per_min": None,
                "pagefile_growth_mb": None,
                "sample_count": len(samples),
                "calls": len(sustained),
                "chat_calls": 0,
            },
            "floor": {
                "full_load_ok": final_load_status.all,
                "duration_ok": False,
                "tok_s_ok": False,
                "batch_ok": False,
                "vram_ok": False,
                "ram_ok": False,
                "page_fault_ok": False,
                "no_crash": False,
                "pass": False,
            },
            "abort_reason": abort_reason or "llama-server did not reach readiness",
            "samples": samples,
            "sustained": [record.as_dict() for record in sustained],
        }
        result["run_file"] = str(BAKEOFF_DIR / f"{profile.name}-run-{stamp()}.json")
        path = Path(result["run_file"])
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        render_bakeoff_md()
        print(f"[bakeoff] wrote failed-launch row {path}")
        print("[bakeoff] gate not met; M1 remains blocked.")
        return 1
    return 0


async def _client_for(args: argparse.Namespace) -> tuple[LLMClient, ServerManager | None, ModelProfile]:
    profile = load_profile(args.profile)
    if getattr(args, "base_url", None):
        client = LLMClient(args.base_url, profile.alias, timeout_s=args.timeout)
        return client, None, profile
    manager = ServerManager(
        profile,
        executable=args.llama_server or None,
        host=args.host or None,
        port=args.port or None,
        startup_timeout_s=args.startup_timeout,
    )
    await manager.start()
    client = LLMClient(manager.base_url, manager.model_name, timeout_s=args.timeout)
    return client, manager, profile


async def cmd_voice(args: argparse.Namespace) -> int:
    BAKEOFF_DIR.mkdir(parents=True, exist_ok=True)
    client, manager, profile = await _client_for(args)
    try:
        personas = load_voice_personas()
        scenes = json.loads(VOICE_SCENES_PATH.read_text(encoding="utf-8"))
        label = args.label or (f"blind-{random.randint(1000, 9999)}" if args.blind else profile.name)
        samples: list[dict[str, Any]] = []
        valid_scenes = 0
        total_messages = 0
        over_12 = 0
        total_words = 0
        refusal_hits = 0
        for index, scene in enumerate(scenes, start=1):
            completion, _ = await run_chat_batch(
                client, scene=scene, personas=personas, recent_chat=[]
            )
            try:
                messages = parse_string_array(completion.text)
            except JSONRepairError:
                messages = []
            if messages:
                valid_scenes += 1
            for message in messages:
                total_messages += 1
                words = len(message.split())
                total_words += words
                if words > 12:
                    over_12 += 1
                if re.search(
                    r"\b(as an ai|i can'?t|i cannot|i won'?t|in this scene|the image shows)\b",
                    message,
                    re.IGNORECASE,
                ):
                    refusal_hits += 1
            samples.append(
                {
                    "scene": scene,
                    "messages": messages,
                    "completion": completion.as_dict(),
                }
            )
            print(f"[voice] {index}/{len(scenes)}: {len(messages)} messages", flush=True)
        payload = {
            "profile": profile.name,
            "label": label,
            "created_at": utc_now(),
            "score": None,
            "rated_at": None,
            "metrics": {
                "scenes": len(scenes),
                "valid_scenes": valid_scenes,
                "messages": total_messages,
                "avg_words": round(total_words / total_messages, 2) if total_messages else None,
                "over_12_words": over_12,
                "refusal_pattern_hits": refusal_hits,
            },
            "samples": samples,
            "server": {"profile": profile.name, "lookahead_s": profile.lookahead_s},
        }
        path = BAKEOFF_DIR / f"voice_{_safe_label(label)}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[voice] wrote {path}")
        print(f"[voice] label: {label}")
        print(f"[voice] rate it with: python scripts/bakeoff.py rate-voice --label {label}")
        return 0
    finally:
        await client.close()
        if manager is not None:
            await manager.stop()


def cmd_rate_voice(args: argparse.Namespace) -> int:
    candidates = []
    for path in BAKEOFF_DIR.glob("voice_*.json"):
        data = load_json(path)
        if data and (args.label is None or data.get("label") == args.label):
            candidates.append((path, data))
    if not candidates:
        print("no voice sample file found")
        return 1
    if len(candidates) > 1:
        print("multiple voice files found; pass --label:")
        for path, data in candidates:
            print(f"  {data.get('label')} ({path.name})")
        return 1
    path, data = candidates[0]
    print(f"label: {data.get('label')}  profile: {data.get('profile')}")
    for sample in data.get("samples", [])[:5]:
        scene = sample.get("scene", {})
        print(f"\nSCENE: {scene.get('summary', '')}")
        for message in sample.get("messages", []):
            print(f"  {message}")
    score = args.score
    if score is None:
        try:
            score = float(input("\nRate 1-5: sounds like real Twitch chat, not an assistant: "))
        except (EOFError, ValueError):
            print("invalid score")
            return 1
    if not 1.0 <= score <= 5.0:
        print("score must be between 1 and 5")
        return 1
    data["score"] = round(score, 2)
    data["rated_at"] = utc_now()
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    render_bakeoff_md()
    print(f"recorded {score:.2f} for profile {data.get('profile')}")
    return 0


async def cmd_accuracy(args: argparse.Namespace) -> int:
    manifest = Path(args.manifest).expanduser()
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    if not manifest.is_file():
        print(f"manifest not found: {manifest}")
        print(
            'Each JSONL line: {"image": "path.jpg", "action_keywords": ["hiding"], '
            '"object_keywords": ["car", "rain"]}'
        )
        return 1
    entries = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(json.loads(line))
    if args.limit:
        entries = entries[: args.limit]
    if not entries:
        print("manifest is empty")
        return 1

    client, manager, profile = await _client_for(args)
    results: list[dict[str, Any]] = []
    try:
        for index, entry in enumerate(entries, start=1):
            image_path = Path(entry["image"])
            if not image_path.is_absolute():
                image_path = manifest.parent / image_path
            if not image_path.is_file():
                results.append({"image": str(image_path), "error": "missing image", "ok": False})
                continue
            completion = await client.complete_vision(
                image_path,
                observer_user(),
                system_prompt=observer_system(),
                max_tokens=120,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            parsed: dict[str, Any] = {}
            error = ""
            try:
                parsed = parse_json(completion.text)
                SceneState(t=0.0, **parsed)
            except (JSONRepairError, ValueError, TypeError) as exc:
                error = str(exc)
            text = f"{parsed.get('summary', '')} {parsed.get('action', '')}".casefold()
            objects = " ".join(str(item) for item in parsed.get("objects", [])).casefold()
            action_keywords = [str(word).casefold() for word in entry.get("action_keywords", [])]
            object_keywords = [str(word).casefold() for word in entry.get("object_keywords", [])]
            action_hit = any(word in text for word in action_keywords) if action_keywords else False
            object_hit = (
                any(word in objects or word in text for word in object_keywords)
                if object_keywords
                else False
            )
            ok = not error and action_hit and object_hit
            results.append(
                {
                    "image": str(image_path),
                    "summary": parsed.get("summary"),
                    "action": parsed.get("action"),
                    "objects": parsed.get("objects"),
                    "confidence": parsed.get("confidence"),
                    "action_hit": action_hit,
                    "object_hit": object_hit,
                    "ok": ok,
                    "error": error,
                    "elapsed_s": round(completion.elapsed_s, 3),
                }
            )
            print(f"[accuracy] {index}/{len(entries)}: {'ok' if ok else 'miss'}")
        hits = sum(1 for result in results if result.get("ok"))
        payload = {
            "profile": profile.name,
            "created_at": utc_now(),
            "manifest": str(manifest),
            "total": len(results),
            "hits": hits,
            "score": round(hits / len(results), 4) if results else 0.0,
            "target": 0.8,
            "results": results,
        }
        path = BAKEOFF_DIR / f"accuracy_{profile.name}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        render_bakeoff_md()
        print(f"[accuracy] {hits}/{len(results)} ({payload['score']:.1%}) -> {path}")
        return 0
    finally:
        await client.close()
        if manager is not None:
            await manager.stop()


def cmd_select(_: argparse.Namespace) -> int:
    runs = latest_runs()
    voice = voice_scores()
    refusals = refusal_results()
    accuracy = accuracy_results()
    missing: list[str] = []
    for name in ("9b", "35b"):
        if name not in runs:
            missing.append(f"missing 30-min run for {name}")
        if name not in voice:
            missing.append(f"missing blind voice score for {name}")
        if name not in refusals or not refusals[name].get("summary", {}).get("pass"):
            missing.append(f"missing/pass refusal audit for {name} (0 refused, <=2 moralized)")
        if name not in accuracy or float(accuracy[name].get("score", 0)) < 0.8:
            missing.append(f"missing accuracy >= 80% for {name}")
    if missing:
        print("cannot select a winner; M0.5 is incomplete:")
        for item in missing:
            print(f"  - {item}")
        return 1

    candidates = [name for name in ("9b", "35b") if runs[name].get("floor", {}).get("pass")]
    if not candidates:
        if "4b" in runs and runs["4b"].get("floor", {}).get("pass"):
            winner = "4b"
            rationale = "9B and 35B missed the hard floor; 4B is the fit fallback."
        else:
            print("no model passed the hard floor; do not start M1. Fix fit/perf first.")
            return 1
    elif len(candidates) == 1:
        winner = candidates[0]
        rationale = f"only {winner} passed the hard floor."
    else:
        score_9b = float(voice["9b"]["score"])
        score_35b = float(voice["35b"]["score"])
        if score_9b == score_35b:
            winner = "9b"
            rationale = f"both passed the floor; voice scores tied at {score_9b:.2f}, tie goes to 9B."
        elif score_35b > score_9b:
            winner = "35b"
            rationale = (
                f"both passed the floor; 35B won the blind voice score "
                f"{score_35b:.2f} vs {score_9b:.2f}."
            )
        else:
            winner = "9b"
            rationale = (
                f"both passed the floor; 9B won the blind voice score "
                f"{score_9b:.2f} vs {score_35b:.2f}."
            )

    run = runs[winner]
    profile = load_profile(winner)
    lookahead = estimate_lookahead(run)
    profile_mapping = {
        "name": profile.name,
        "model_id": profile.model_id,
        "model": str(profile.model_path),
        "mmproj": str(profile.mmproj_path),
        "alias": profile.alias,
        "ctx": profile.ctx,
        "n_gpu_layers": profile.n_gpu_layers,
        "n_cpu_moe": profile.n_cpu_moe,
        "threads": profile.threads,
        "batch": profile.batch,
        "ubatch": profile.ubatch,
        "cache_type_k": profile.cache_type_k,
        "cache_type_v": profile.cache_type_v,
        "flash_attn": profile.flash_attn,
        "jinja": profile.jinja,
        "disable_thinking": profile.disable_thinking,
        "host": profile.host,
        "port": profile.port,
        "lookahead_s": lookahead,
        "extra_args": profile.extra_args,
    }
    selection = {
        "winner": winner,
        "selected_at": utc_now(),
        "rationale": rationale,
        "lookahead_s": lookahead,
        "command": profile.command(),
        "run_file": run.get("_path"),
        "voice_scores": {name: voice[name]["score"] for name in ("9b", "35b")},
        "floor": {name: runs[name].get("floor", {}) for name in ("9b", "35b")},
    }
    SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    SELECTION_PATH.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    MODEL_PROFILE_PATH.write_text(
        json.dumps(
            {
                "winner": winner,
                "selected_at": selection["selected_at"],
                "rationale": rationale,
                "lookahead_s": lookahead,
                "profile": profile_mapping,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    render_bakeoff_md()
    print(f"winner: {winner}")
    print(rationale)
    print(f"look-ahead: {lookahead:.1f}s")
    print(f"wrote {SELECTION_PATH} and {MODEL_PROFILE_PATH}")
    return 0


def cmd_provisional(args: argparse.Namespace) -> int:
    """Record an explicit user waiver when the M0.5 measurements are skipped.

    This is deliberately not the normal selection path. It writes a usable
    profile so the app can run, but marks every artifact as provisional and
    never claims the floor was measured.
    """
    profile = load_profile(args.profile)
    reason = args.reason or (
        "M0.5 was waived by the user; this profile is provisional and has no "
        "measured bake-off evidence."
    )
    lookahead = profile.lookahead_s
    profile_mapping = {
        "name": profile.name,
        "model_id": profile.model_id,
        "model": str(profile.model_path),
        "mmproj": str(profile.mmproj_path),
        "alias": profile.alias,
        "ctx": profile.ctx,
        "n_gpu_layers": profile.n_gpu_layers,
        "n_cpu_moe": profile.n_cpu_moe,
        "threads": profile.threads,
        "batch": profile.batch,
        "ubatch": profile.ubatch,
        "cache_type_k": profile.cache_type_k,
        "cache_type_v": profile.cache_type_v,
        "flash_attn": profile.flash_attn,
        "jinja": profile.jinja,
        "disable_thinking": profile.disable_thinking,
        "host": profile.host,
        "port": profile.port,
        "lookahead_s": lookahead,
        "extra_args": profile.extra_args,
    }
    selection = {
        "winner": profile.name,
        "selected_at": utc_now(),
        "rationale": reason,
        "lookahead_s": lookahead,
        "command": profile.command(),
        "run_file": None,
        "voice_scores": {},
        "floor": {},
        "provisional": True,
        "gate_waived": True,
        "waiver_reason": reason,
    }
    SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    SELECTION_PATH.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    MODEL_PROFILE_PATH.write_text(
        json.dumps(
            {
                "winner": profile.name,
                "selected_at": selection["selected_at"],
                "rationale": reason,
                "lookahead_s": lookahead,
                "profile": profile_mapping,
                "provisional": True,
                "gate_waived": True,
                "waiver_reason": reason,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    render_bakeoff_md()
    print(f"provisional profile: {profile.name}")
    print(reason)
    print(f"look-ahead: {lookahead:.1f}s")
    print(f"wrote {SELECTION_PATH} and {MODEL_PROFILE_PATH}")
    print("This is not a measured M0.5 winner; run 'select' after real evidence to replace it.")
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    runs = latest_runs()
    voice = voice_scores()
    refusals = refusal_results()
    accuracy = accuracy_results()
    if not runs:
        print("no bakeoff runs recorded yet")
        return 1
    for name, run in sorted(runs.items()):
        floor = run.get("floor", {})
        summary = run.get("summary", {})
        print(
            f"{name}: floor={'PASS' if floor.get('pass') else 'FAIL'} "
            f"tok/s@end={summary.get('tok_s_end')} "
            f"batch_p95={summary.get('batch8_p95_s')}s "
            f"full_load={run.get('full_load', {}).get('all')} "
            f"voice={voice.get(name, {}).get('score')} "
            f"refusal_pass={refusals.get(name, {}).get('summary', {}).get('pass')} "
            f"accuracy={accuracy.get(name, {}).get('score')}"
        )
    selection = load_json(SELECTION_PATH)
    if selection and selection.get("provisional"):
        print(f"winner: {selection.get('winner')} (PROVISIONAL — M0.5 waived)")
    else:
        print(f"winner: {selection.get('winner') if selection else 'not selected'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_server_args(target: argparse.ArgumentParser) -> None:
        target.add_argument("--profile", required=True, help="profile name (9b/35b/4b) or TOML/JSON path")
        target.add_argument("--base-url", help="use an already-running llama-server instead of launching")
        target.add_argument("--llama-server", help="llama-server executable (default: PATH or profile)")
        target.add_argument("--host", help="override server host")
        target.add_argument("--port", type=int, help="override server port")
        target.add_argument("--startup-timeout", type=float, default=240.0)
        target.add_argument("--timeout", type=float, default=180.0)

    run = sub.add_parser("run", help="30-minute full-load gate run for one model")
    add_server_args(run)
    run.add_argument("--duration-min", type=float, default=GATE_MINUTES)
    run.add_argument("--runs", type=int, default=20)
    run.add_argument("--media", default=str(ROOT / "data" / "media" / "sample.mp4"))
    run.add_argument("--app-url", default="http://127.0.0.1:8000")
    run.add_argument("--stt-load", default="", help="command that keeps a real STT workload running")
    run.add_argument("--sample-interval", type=float, default=10.0)
    run.add_argument("--allow-partial", action="store_true", help="allow a harness run without full load")
    run.add_argument("--no-early-abort", action="store_true")
    run.set_defaults(func=lambda args: asyncio.run(cmd_run(args)))

    voice = sub.add_parser("voice", help="generate blind chat-voice samples")
    add_server_args(voice)
    voice.add_argument("--blind", action="store_true")
    voice.add_argument("--label")
    voice.add_argument("--scenes", default=str(VOICE_SCENES_PATH))
    voice.set_defaults(func=lambda args: asyncio.run(cmd_voice(args)))

    rate = sub.add_parser("rate-voice", help="record a 1-5 human rating for generated samples")
    rate.add_argument("--label")
    rate.add_argument("--score", type=float)
    rate.set_defaults(func=cmd_rate_voice)

    accuracy = sub.add_parser("accuracy", help="run the 30-frame perception accuracy check")
    add_server_args(accuracy)
    accuracy.add_argument("--manifest", default="data/perception_eval/manifest.jsonl")
    accuracy.add_argument("--limit", type=int)
    accuracy.set_defaults(func=lambda args: asyncio.run(cmd_accuracy(args)))

    select = sub.add_parser("select", help="apply the decision rule and write data/model_profile.json")
    select.set_defaults(func=cmd_select)

    provisional = sub.add_parser(
        "provisional",
        help="record an explicit user waiver and write a usable, clearly provisional profile",
    )
    provisional.add_argument("--profile", required=True, help="profile name (e.g. 4b)")
    provisional.add_argument(
        "--reason",
        default="M0.5 was waived by the user; this profile is provisional and has no measured evidence.",
    )
    provisional.set_defaults(func=cmd_provisional)

    status = sub.add_parser("status", help="show recorded run/evidence status")
    status.set_defaults(func=cmd_status)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
