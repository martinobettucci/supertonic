"""Launch the persistent Supertonic API on localhost."""
from __future__ import annotations

import argparse
import ipaddress
from pathlib import Path

import uvicorn

from supertonic.server import ServerConfig, create_app


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=ROOT / "assets" / "onnx",
    )
    parser.add_argument(
        "--voice-styles-dir",
        type=Path,
        default=ROOT / "assets" / "voice_styles",
    )
    parser.add_argument("--asr-model", default="openai/whisper-base")
    parser.add_argument(
        "--asr-device",
        choices=["auto", "cpu", "cuda", "gpu"],
        default="auto",
    )
    parser.add_argument("--no-asr", action="store_true")
    parser.add_argument(
        "--lazy-asr",
        action="store_true",
        help="Load Whisper on the first evaluated request instead of startup.",
    )
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def ensure_loopback(host: str) -> None:
    normalized = host.strip().lower()
    if normalized == "localhost":
        return
    try:
        if ipaddress.ip_address(normalized).is_loopback:
            return
    except ValueError:
        pass
    raise SystemExit(
        "serve_api.py only accepts loopback hosts: localhost, 127.0.0.1, or ::1."
    )


def main() -> None:
    args = parse_args()
    ensure_loopback(args.host)
    if not 1 <= args.port <= 65_535:
        raise SystemExit("--port must be between 1 and 65535")
    if args.max_batch_size <= 0:
        raise SystemExit("--max-batch-size must be greater than zero")

    config = ServerConfig(
        onnx_dir=args.onnx_dir.resolve(),
        voice_styles_dir=args.voice_styles_dir.resolve(),
        asr_model=None if args.no_asr else args.asr_model,
        asr_device=args.asr_device,
        preload_asr=not args.lazy_asr,
        max_batch_size=args.max_batch_size,
    )
    app = create_app(config)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=True,
    )


if __name__ == "__main__":
    main()
