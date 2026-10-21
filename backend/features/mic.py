"""Microphone capture, VAD and Parakeet STT (M8, PLAN.md §10.5).

The microphone is optional and every dependency is lazy:

* ``sounddevice`` for capture (missing -> status ``unavailable``),
* a small energy VAD so a silent room never calls STT,
* ``onnx-asr`` Parakeet on CPU for transcription.

Transcription feeds the same ``SceneState`` path as video audio and publishes a
``mic.speech`` bus event. The dynamics engine turns that event into 1-3 lines
within a few seconds, so "chat can hear you" does not depend on the vision loop
running. STT never uses the GPU, which keeps the chat model's VRAM untouched.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import math
import queue
import struct
import time
import wave
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .. import events

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..state import AppState

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
FRAME_MS = 30
_FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2


class MicError(RuntimeError):
    """Capture or transcription is not available."""


@dataclass(slots=True)
class SpeechSegment:
    pcm: bytes
    energy: float
    duration_s: float


class VadSegmenter:
    """Energy VAD with pre-roll and an adaptive noise floor.

    It is intentionally not a model: the goal is to avoid running STT on
    silence, not to be a perfect speech detector. ``webrtcvad``/Silero can be
    swapped in behind the same ``push`` method later.
    """

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        frame_ms: int = FRAME_MS,
        threshold: float = 0.012,
        min_speech_s: float = 0.7,
        max_speech_s: float = 8.0,
        silence_s: float = 0.7,
        preroll_s: float = 0.25,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * 2
        self.threshold = max(0.001, float(threshold))
        self.min_frames = max(1, int(min_speech_s * 1000 / frame_ms))
        self.max_frames = max(self.min_frames, int(max_speech_s * 1000 / frame_ms))
        self.silence_frames = max(1, int(silence_s * 1000 / frame_ms))
        self.preroll_frames = max(0, int(preroll_s * 1000 / frame_ms))
        self._pending = b""
        self._preroll: list[bytes] = []
        self._speech: list[bytes] = []
        self._speech_frames = 0
        self._silence = 0
        self._active = False
        self._noise = 0.002
        self._voiced_run = 0

    def push(self, pcm: bytes) -> list[SpeechSegment]:
        self._pending += pcm
        segments: list[SpeechSegment] = []
        while len(self._pending) >= self.frame_bytes:
            frame = self._pending[: self.frame_bytes]
            self._pending = self._pending[self.frame_bytes :]
            segment = self._process(frame)
            if segment is not None:
                segments.append(segment)
        return segments

    def flush(self) -> SpeechSegment | None:
        if not self._active:
            return None
        segment = self._finish()
        return segment

    def _process(self, frame: bytes) -> SpeechSegment | None:
        energy = pcm_energy(frame)
        voiced = energy >= max(self.threshold, self._noise * 3.0)
        if not self._active:
            self._noise = 0.98 * self._noise + 0.02 * energy
            self._preroll.append(frame)
            if len(self._preroll) > self.preroll_frames:
                self._preroll.pop(0)
            self._voiced_run = self._voiced_run + 1 if voiced else 0
            if self._voiced_run >= 2:
                self._active = True
                self._speech = list(self._preroll)
                self._speech_frames = self._voiced_run
                self._silence = 0
                self._preroll.clear()
                self._voiced_run = 0
            return None

        self._speech.append(frame)
        if energy >= self.threshold:
            self._speech_frames += 1
            self._silence = 0
        else:
            self._silence += 1
        if self._silence >= self.silence_frames or len(self._speech) >= self.max_frames:
            return self._finish()
        return None

    def _finish(self) -> SpeechSegment | None:
        frames = self._speech
        speech_frames = self._speech_frames
        self._active = False
        self._speech = []
        self._speech_frames = 0
        self._silence = 0
        self._preroll.clear()
        if not frames or speech_frames < self.min_frames:
            return None
        pcm = b"".join(frames)
        return SpeechSegment(
            pcm=pcm,
            energy=pcm_energy(pcm),
            duration_s=len(pcm) / 2 / self.sample_rate,
        )


def pcm_energy(pcm: bytes) -> float:
    if len(pcm) < 2:
        return 0.0
    count = len(pcm) // 2
    total = 0.0
    for (sample,) in struct.iter_unpack("<h", pcm[: count * 2]):
        total += float(sample) * float(sample)
    rms = math.sqrt(total / max(1, count))
    return min(1.0, rms / 32768.0)


class MicService:
    """Optional capture loop + a manual transcript path for testing."""

    def __init__(
        self,
        state: AppState,
        *,
        engine: Any | None = None,
        capture: Any | None = None,
        segmenter: VadSegmenter | None = None,
    ) -> None:
        self.state = state
        self.settings = state.settings
        self.segmenter = segmenter or VadSegmenter(
            sample_rate=int(self.settings.mic_sample_rate),
            threshold=float(self.settings.mic_silence_threshold),
        )
        self._engine = engine
        self._capture = capture
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self._status = "disabled" if not self.settings.mic_enabled else "idle"
        self._last_error = ""
        self._segments = 0
        self._transcripts = 0
        self._last_text = ""
        self._last_energy = 0.0
        self._last_at: float | None = None
        self._capture_queue: queue.Queue[bytes] | None = None
        self._stream: Any = None

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        if not self.settings.mic_enabled:
            self._status = "disabled"
            return
        if self._task is not None and not self._task.done():
            return
        if self._capture is None:
            self._status = "starting"
        self._stopping = False
        self._task = asyncio.create_task(self._run(), name="mic-service")

    async def stop(self) -> None:
        self._stopping = True
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._close_stream()
        self._status = "stopped" if self.settings.mic_enabled else "disabled"

    def metrics(self) -> dict[str, Any]:
        return {
            "status": self._status,
            "enabled": bool(self.settings.mic_enabled),
            "segments": self._segments,
            "transcripts": self._transcripts,
            "last_text": self._last_text,
            "last_energy": round(self._last_energy, 4),
            "last_at": self._last_at,
            "stt_model": self.settings.mic_stt_model,
            "last_error": self._last_error,
            "device": self._device_name(),
        }

    # -- loop --------------------------------------------------------------

    async def _run(self) -> None:
        try:
            if self._capture is None:
                await asyncio.to_thread(self._open_stream)
            await self._capture_loop()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._status = "unavailable"
            self._last_error = str(exc)[:400]
            logger.warning("microphone unavailable: %s", self._last_error)
        finally:
            self._close_stream()

    async def _capture_loop(self) -> None:
        self._status = "listening"
        while not self._stopping:
            block = await self._next_block()
            if block is None:
                continue
            for segment in self.segmenter.push(block):
                self._segments += 1
                self._consume(segment)
            await asyncio.sleep(0)

    async def _next_block(self) -> bytes | None:
        if self._capture is not None:
            value = self._capture()
            if asyncio.iscoroutine(value):
                value = await value
            return value if isinstance(value, bytes) else None
        if self._capture_queue is None:
            return None
        try:
            return await asyncio.to_thread(self._capture_queue.get, True, 0.5)
        except queue.Empty:
            return None

    def _open_stream(self) -> None:
        try:
            import sounddevice as sd  # type: ignore[import-not-found]
        except Exception as exc:
            raise MicError(
                "sounddevice is not installed; install the 'mic' extra"
            ) from exc
        self._capture_queue = queue.Queue(maxsize=64)

        def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
            if self._capture_queue is None:
                return
            with contextlib.suppress(queue.Full):
                self._capture_queue.put_nowait(bytes(indata))
            if status:
                self._last_error = str(status)[:200]

        self._stream = sd.RawInputStream(
            samplerate=self.settings.mic_sample_rate,
            channels=1,
            dtype="int16",
            blocksize=int(self.settings.mic_sample_rate * 0.1),
            callback=callback,
        )
        self._stream.start()

    def _close_stream(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            with contextlib.suppress(Exception):
                stream.stop()
            with contextlib.suppress(Exception):
                stream.close()

    def _consume(self, segment: SpeechSegment) -> None:
        if len(segment.pcm) < int(self.settings.mic_sample_rate * 0.5) * 2:
            return
        task = asyncio.create_task(self._transcribe(segment), name="mic-transcribe")
        task.add_done_callback(lambda finished: finished.exception())

    async def _transcribe(self, segment: SpeechSegment) -> None:
        self._status = "transcribing"
        try:
            text = await asyncio.to_thread(self._recognize, segment)
        except Exception as exc:
            self._last_error = str(exc)[:400]
            self._status = "listening"
            return
        self._status = "listening"
        if text:
            self.inject_transcript(text, energy=segment.energy, source="mic")

    def _recognize(self, segment: SpeechSegment) -> str:
        if self._engine is not None:
            return str(self._engine(segment.pcm) or "").strip()
        try:
            import onnx_asr  # type: ignore[import-not-found]
        except Exception as exc:
            raise MicError("onnx-asr is not installed; install the 'stt' extra") from exc
        model = onnx_asr.load_model(self.settings.mic_stt_model)
        wav = _wav_bytes(segment.pcm, self.settings.mic_sample_rate)
        result = model.recognize(wav)
        text = getattr(result, "text", result)
        return str(text or "").strip()

    # -- transcript path ---------------------------------------------------

    def inject_transcript(
        self,
        text: str,
        *,
        energy: float = 0.8,
        source: str = "mic",
    ) -> str:
        """Push a transcript into the scene path (API/manual/browser fallback)."""
        cleaned = " ".join(str(text).split())[:300]
        if not cleaned:
            return ""
        self._transcripts += 1
        self._last_text = cleaned
        self._last_energy = min(1.0, max(0.0, float(energy)))
        self._last_at = time.time()
        t_play = self._t_play()
        state = self.state
        state.latest_speech = {
            "text": cleaned,
            "energy": self._last_energy,
            "t": t_play,
            "at": self._last_at,
            "source": source,
        }
        with contextlib.suppress(RuntimeError):
            state.bus.publish_nowait(
                events.MIC_SPEECH,
                {
                    "text": cleaned,
                    "energy": self._last_energy,
                    "t": round(t_play, 3),
                    "source": source,
                },
            )
        cues = getattr(state, "chat_cues", None)
        if cues is not None:
            cues.append({"kind": "speech", "text": cleaned})
        return cleaned

    def _device_name(self) -> str:
        if self._capture is not None:
            return "injected"
        if self._stream is None:
            return ""
        return getattr(self._stream, "device", "") or "default"

    def _t_play(self) -> float:
        try:
            return float(self.state.clock_snapshot().t_play)
        except Exception:  # pragma: no cover - defensive
            return 0.0


def _wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


__all__ = ["MicError", "MicService", "SpeechSegment", "VadSegmenter", "pcm_energy"]
