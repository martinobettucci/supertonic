"""Incremental PCM streaming example for the Supertonic ONNX runtime."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import soundfile as sf

from helper import load_text_to_speech, load_voice_style


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=ROOT / "assets" / "onnx",
    )
    parser.add_argument(
        "--voice-style",
        type=Path,
        default=ROOT / "assets" / "voice_styles" / "F1.json",
    )
    parser.add_argument("--lang", default="en")
    parser.add_argument("--total-step", type=int, default=3)
    parser.add_argument("--speed", type=float, default=1.05)
    parser.add_argument("--max-segment-chars", type=int, default=120)
    parser.add_argument("--audio-chunk-ms", type=float, default=100.0)
    parser.add_argument("--simulate-playback", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/streaming_output.wav"),
    )
    parser.add_argument(
        "--text",
        default=(
            "Hello, this example emits PCM blocks as soon as each text segment "
            "is ready. The following segment is synthesized in the background "
            "while the current audio is being consumed."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tts = load_text_to_speech(str(args.onnx_dir), use_gpu=False)
    style = load_voice_style([str(args.voice_style)])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    first_audio_at = None
    audio_seconds = 0.0
    last_ready_at = 0.0
    playback_cursor = None
    estimated_underflow = 0.0
    chunk_count = 0
    with sf.SoundFile(
        args.output,
        mode="w",
        samplerate=tts.sample_rate,
        channels=1,
        subtype="PCM_16",
    ) as output:
        for chunk in tts.stream(
            args.text,
            args.lang,
            style,
            total_step=args.total_step,
            speed=args.speed,
            max_segment_chars=args.max_segment_chars,
            audio_chunk_ms=args.audio_chunk_ms,
        ):
            if first_audio_at is None:
                first_audio_at = time.perf_counter() - started
                print(f"time to first audio: {first_audio_at:.3f}s")
            if chunk.is_segment_start:
                print(
                    f"segment {chunk.segment_index}: ready after "
                    f"{chunk.ready_after_seconds:.3f}s: {chunk.text!r}"
                )
            output.write(chunk.audio)
            output.flush()
            last_ready_at = max(last_ready_at, chunk.ready_after_seconds)
            if playback_cursor is None:
                playback_cursor = chunk.ready_after_seconds
            if chunk.ready_after_seconds > playback_cursor:
                estimated_underflow += chunk.ready_after_seconds - playback_cursor
                playback_cursor = chunk.ready_after_seconds
            playback_cursor += chunk.duration_seconds
            audio_seconds += chunk.duration_seconds
            chunk_count += 1
            if args.simulate_playback:
                time.sleep(chunk.duration_seconds)

    elapsed = time.perf_counter() - started
    generation_throughput = (
        audio_seconds / last_ready_at if last_ready_at else float("inf")
    )
    print(f"wrote {chunk_count} chunks ({audio_seconds:.2f}s) to {args.output}")
    print(
        f"generation ready in {last_ready_at:.2f}s, "
        f"throughput: {generation_throughput:.2f}x realtime"
    )
    print(f"estimated playback underflow after first audio: {estimated_underflow:.3f}s")
    print(f"end-to-end wall time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
