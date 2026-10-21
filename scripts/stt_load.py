#!/usr/bin/env python3
"""Run the real Parakeet STT workload used by the M0.5 full-load gate.

M8 will wire Parakeet TDT 0.6B v3 into the audio tap. This script is the M0
install/load spike from PLAN.md §5.3: it runs the same ONNX model continuously
on CPU, so the bake-off measures the RAM/CPU contention the shipped app will
actually have instead of a synthetic CPU burner.

It does not claim word accuracy and it is not the M8 feature. It decodes a
media file once with ffmpeg, slices it into chunks, and transcribes chunks in a
loop until killed (or until ``--duration-min`` expires). The encoder cost is
driven by audio duration, so a silent/demo clip still exercises the real
preprocessing, encoder, decoder and runtime allocation path.

Examples
--------
    # install the pinned STT stack once
    .venv/bin/python -m pip install -e ".[stt]"

    # keep a real STT workload alive for a bake-off (bakeoff terminates it)
    .venv/bin/python scripts/stt_load.py --audio data/media/sample.mp4 --threads 4

    # attach it to a gate run
    python scripts/bakeoff.py run --profile 9b \
      --stt-load ".venv/bin/python scripts/stt_load.py --audio data/media/sample.mp4 --threads 4"

The audio argument may be any file ffmpeg can decode. Prefer a spoken clip for
a realistic decode mix; the demo clip is a tone and produces few output tokens.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio",
        default="data/media/sample.mp4",
        help="media file to transcribe in a loop (default: data/media/sample.mp4)",
    )
    parser.add_argument(
        "--model",
        default="nemo-parakeet-tdt-0.6b-v3",
        help="onnx-asr model name or local model directory (default: Parakeet v3)",
    )
    parser.add_argument(
        "--quantization",
        default="int8",
        help="model quantization: int8 for CPU, empty for fp32 (default: int8)",
    )
    parser.add_argument("--threads", type=int, default=4, help="ONNX intra-op threads (default: 4)")
    parser.add_argument(
        "--chunk-seconds", type=float, default=8.0, help="audio chunk length (default: 8 s)"
    )
    parser.add_argument(
        "--duration-min",
        type=float,
        default=0.0,
        help="stop after N minutes; 0 means run until terminated (default: 0)",
    )
    parser.add_argument(
        "--max-audio-seconds",
        type=float,
        default=600.0,
        help="cap decoded audio kept in RAM (default: 600 s)",
    )
    parser.add_argument("--quiet", action="store_true", help="only print startup / errors")
    return parser.parse_args(argv)


def decode_audio(path: Path, max_seconds: float) -> tuple[object, int]:
    """Decode to 16 kHz mono float32 using ffmpeg, without a soundfile dependency."""
    import numpy as np

    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-t",
        str(max_seconds),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise SystemExit("ffmpeg is required on PATH to decode the STT audio") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise SystemExit(f"ffmpeg could not decode {path}: {detail}") from exc
    samples = np.frombuffer(result.stdout, dtype="<f4").copy()
    if not len(samples):
        raise SystemExit(f"ffmpeg produced no audio for {path}")
    return samples, 16000


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audio_path = Path(args.audio)
    if not audio_path.is_absolute():
        audio_path = Path(__file__).resolve().parents[1] / audio_path
    if not audio_path.is_file():
        print(f"audio not found: {audio_path}", file=sys.stderr)
        return 2
    if args.threads < 1:
        print("--threads must be >= 1", file=sys.stderr)
        return 2

    # Set this before onnxruntime is imported so its CPU arena/thread pools see it.
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    try:
        import onnx_asr
        import onnxruntime as rt
    except ImportError as exc:
        print(
            "onnx-asr is not installed. Run:\n"
            '  .venv/bin/python -m pip install -e ".[stt]"',
            file=sys.stderr,
        )
        print(f"({exc})", file=sys.stderr)
        return 2

    samples, sample_rate = decode_audio(audio_path, args.max_audio_seconds)
    chunk_samples = max(1600, int(args.chunk_seconds * sample_rate))
    chunks = [
        samples[offset : offset + chunk_samples]
        for offset in range(0, len(samples), chunk_samples)
        if len(samples[offset : offset + chunk_samples]) >= sample_rate // 2
    ]
    if not chunks:
        print("audio is shorter than 0.5 s; nothing to transcribe", file=sys.stderr)
        return 2

    options = rt.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL

    model_ref: Path | str = args.model
    local_model = Path(args.model)
    if local_model.is_dir():
        model_ref = local_model

    print(
        f"[stt-load] model={args.model} audio={audio_path} "
        f"chunks={len(chunks)} chunk={args.chunk_seconds:.1f}s threads={args.threads} "
        f"providers={rt.get_available_providers()}",
        flush=True,
    )
    model = onnx_asr.load_model(
        model_ref,
        quantization=args.quantization or None,
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    deadline = time.monotonic() + args.duration_min * 60.0 if args.duration_min > 0 else None
    chunks_done = 0
    started = time.monotonic()
    try:
        while deadline is None or time.monotonic() < deadline:
            for chunk in chunks:
                model.recognize(chunk, sample_rate=sample_rate)
                chunks_done += 1
                if not args.quiet and chunks_done % 10 == 0:
                    elapsed = max(1e-6, time.monotonic() - started)
                    audio_s = chunks_done * (len(chunk) / sample_rate)
                    print(
                        f"[stt-load] chunks={chunks_done} "
                        f"audio={audio_s:.0f}s wall={elapsed:.1f}s rtf={audio_s / elapsed:.1f}x",
                        flush=True,
                    )
                if deadline is not None and time.monotonic() >= deadline:
                    break
    except KeyboardInterrupt:
        return 130
    finally:
        with contextlib.suppress(Exception):
            model.close()

    if not args.quiet:
        print(f"[stt-load] stopped after {chunks_done} chunks", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
