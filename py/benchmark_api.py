"""Exercise the batch and WebSocket APIs and save their generated audio."""
from __future__ import annotations

import argparse
import asyncio
import base64
import ipaddress
import json
from pathlib import Path
from urllib.parse import urlparse

import httpx
import soundfile as sf


DEFAULT_TEXT = (
    "Bonjour, votre rendez-vous est confirmé demain matin à neuf heures. "
    "Merci d'arriver quinze minutes avant avec votre ordonnance."
)


def ensure_local_url(base_url: str) -> None:
    parsed = urlparse(base_url)
    if parsed.scheme != "http" or not parsed.hostname:
        raise SystemExit("--base-url must be an HTTP localhost URL")
    hostname = parsed.hostname.strip().lower()
    if hostname == "localhost":
        return
    try:
        if ipaddress.ip_address(hostname).is_loopback:
            return
    except ValueError:
        pass
    raise SystemExit("benchmark_api.py only connects to a localhost server")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--voice", default="F1")
    parser.add_argument("--speed", type=float, default=1.25)
    parser.add_argument("--batch-total-step", type=int, default=8)
    parser.add_argument("--stream-total-step", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-asr", action="store_true")
    parser.add_argument(
        "--max-wer",
        type=float,
        default=None,
        help="Fail after saving artifacts if any measured WER exceeds this value.",
    )
    parser.add_argument("--mode", choices=["batch", "stream", "both"], default="both")
    parser.add_argument("--output-dir", type=Path, default=Path("results/api"))
    return parser.parse_args()


def request_payload(args: argparse.Namespace, total_step: int) -> dict:
    return {
        "text": args.text,
        "language": args.language,
        "voice": args.voice,
        "speed": args.speed,
        "sampling_steps": total_step,
        "seed": args.seed,
        "evaluate_asr": not args.no_asr,
    }


def run_batch(args: argparse.Namespace) -> dict:
    payload = {
        "items": [request_payload(args, args.batch_total_step)],
        "include_audio": True,
    }
    with httpx.Client(timeout=300.0) as client:
        response = client.post(f"{args.base_url}/v1/tts/batch", json=payload)
        response.raise_for_status()
    body = response.json()
    result = body["results"][0]
    audio = base64.b64decode(result["audio"]["data"])
    output = args.output_dir / "batch.wav"
    output.write_bytes(audio)
    write_json(args.output_dir / "batch_metrics.json", result["metrics"])
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"batch audio: {output}")
    return result["metrics"]


async def run_stream(args: argparse.Namespace) -> dict:
    try:
        import websockets
    except ImportError as exc:
        raise SystemExit(
            "Streaming benchmark requires the optional 'websockets' dependency."
        ) from exc

    websocket_url = args.base_url.replace("http://", "ws://").replace(
        "https://", "wss://"
    )
    pcm = bytearray()
    final_metrics = None
    async with websockets.connect(
        f"{websocket_url}/v1/tts/stream",
        max_size=None,
        open_timeout=30,
    ) as websocket:
        await websocket.send(
            json.dumps(request_payload(args, args.stream_total_step))
        )
        async for message in websocket:
            if isinstance(message, bytes):
                pcm.extend(message)
                continue
            event = json.loads(message)
            if event.get("event") == "segment":
                print(
                    f"segment {event['segment_index']} ready after "
                    f"{event['ready_after_seconds']:.3f}s"
                )
            elif event.get("event") == "error":
                raise RuntimeError(event.get("detail") or "stream failed")
            elif event.get("event") == "complete":
                final_metrics = event["metrics"]

    if final_metrics is None:
        raise RuntimeError("stream closed without final metrics")
    samples = memoryview(pcm).cast("h")
    output = args.output_dir / "stream.wav"
    sf.write(
        output,
        samples,
        int(final_metrics.get("sample_rate", 44_100)),
        subtype="PCM_16",
    )
    write_json(args.output_dir / "stream_metrics.json", final_metrics)
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))
    print(f"stream audio: {output}")
    return final_metrics


def main() -> None:
    args = parse_args()
    ensure_local_url(args.base_url)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.max_wer is not None and args.max_wer < 0:
        raise SystemExit("--max-wer cannot be negative")
    request_metrics = {}
    with httpx.Client(timeout=30.0) as client:
        health = client.get(f"{args.base_url}/health")
        health.raise_for_status()
        health_payload = health.json()
        write_json(args.output_dir / "health.json", health_payload)
        print(json.dumps(health_payload, ensure_ascii=False, indent=2))

    if args.mode in {"batch", "both"}:
        request_metrics["batch"] = run_batch(args)
    if args.mode in {"stream", "both"}:
        request_metrics["stream"] = asyncio.run(run_stream(args))

    with httpx.Client(timeout=30.0) as client:
        metrics = client.get(f"{args.base_url}/v1/metrics")
        metrics.raise_for_status()
        metrics_payload = metrics.json()
        write_json(args.output_dir / "server_metrics.json", metrics_payload)
        print(json.dumps(metrics_payload, ensure_ascii=False, indent=2))

    wer_scores = {
        mode: metrics["asr"]["wer"]
        for mode, metrics in request_metrics.items()
        if metrics.get("asr") and metrics["asr"].get("wer") is not None
    }
    violations = {
        mode: score
        for mode, score in wer_scores.items()
        if args.max_wer is not None and score > args.max_wer
    }
    summary = {
        "max_wer": args.max_wer,
        "passed": not violations,
        "wer": wer_scores,
        "violations": violations,
    }
    write_json(args.output_dir / "benchmark_summary.json", summary)
    if violations:
        raise SystemExit(
            "WER quality gate failed: "
            + ", ".join(f"{mode}={score:.4f}" for mode, score in violations.items())
        )


if __name__ == "__main__":
    main()
