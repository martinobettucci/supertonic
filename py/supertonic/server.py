"""Persistent localhost API for full and streaming Supertonic synthesis."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import AliasChoices, BaseModel, Field

from helper import AudioChunk, load_text_to_speech, load_voice_style
from supertonic.asr import AsrEvaluation, WhisperAsr


class SynthesisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2_000)
    language: str = Field(default="en", min_length=2, max_length=16)
    voice: str = Field(default="F1", min_length=1, max_length=64)
    speed: float = Field(default=1.05, ge=0.7, le=2.0)
    total_step: Optional[int] = Field(
        default=None,
        ge=1,
        le=12,
        validation_alias=AliasChoices("sampling_steps", "total_step"),
    )
    seed: Optional[int] = Field(default=42, ge=0, le=4_294_967_295)
    evaluate_asr: bool = True
    max_segment_chars: int = Field(default=120, ge=24, le=500)
    audio_chunk_ms: float = Field(default=100.0, ge=20.0, le=1_000.0)


class BatchSynthesisRequest(BaseModel):
    items: list[SynthesisRequest]
    include_audio: bool = True


@dataclass(frozen=True)
class ServerConfig:
    onnx_dir: Path
    voice_styles_dir: Path
    asr_model: Optional[str] = "openai/whisper-base"
    asr_device: str = "auto"
    preload_asr: bool = True
    max_batch_size: int = 16
    recent_metrics_limit: int = 100


@dataclass
class GenerationMetrics:
    request_id: str
    mode: str
    language: str
    voice: str
    speed: float
    total_step: int
    seed: Optional[int]
    text_characters: int
    sample_rate: int
    audio_duration_seconds: float
    queue_wait_seconds: float
    generation_seconds: float
    total_seconds: float
    realtime_factor: float
    throughput_x_realtime: float
    time_to_first_audio_seconds: Optional[float] = None
    chunk_count: Optional[int] = None
    segment_count: Optional[int] = None
    asr: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MetricsRegistry:
    def __init__(self, recent_limit: int = 100):
        self.started_at = time.time()
        self._lock = threading.Lock()
        self._recent = deque(maxlen=recent_limit)
        self._requests = 0
        self._failures = 0
        self._audio_seconds = 0.0
        self._generation_seconds = 0.0
        self._wer_sum = 0.0
        self._wer_count = 0

    def record(self, metrics: GenerationMetrics) -> None:
        payload = metrics.to_dict()
        with self._lock:
            self._requests += 1
            self._audio_seconds += metrics.audio_duration_seconds
            self._generation_seconds += metrics.generation_seconds
            if metrics.asr and metrics.asr.get("wer") is not None:
                self._wer_sum += float(metrics.asr["wer"])
                self._wer_count += 1
            self._recent.append(payload)

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            throughput = (
                self._audio_seconds / self._generation_seconds
                if self._generation_seconds
                else None
            )
            return {
                "uptime_seconds": time.time() - self.started_at,
                "requests": self._requests,
                "failures": self._failures,
                "audio_seconds": self._audio_seconds,
                "generation_seconds": self._generation_seconds,
                "throughput_x_realtime": throughput,
                "mean_wer": (
                    self._wer_sum / self._wer_count if self._wer_count else None
                ),
                "wer_samples": self._wer_count,
                "recent": list(self._recent),
            }


class SupertonicService:
    """Own the loaded ONNX sessions, voice styles, ASR, and request locks."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.tts = None
        self.voices: dict[str, Any] = {}
        self.asr = (
            WhisperAsr(config.asr_model, config.asr_device)
            if config.asr_model
            else None
        )
        self.metrics = MetricsRegistry(config.recent_metrics_limit)
        self.ready = False
        self.tts_load_seconds: Optional[float] = None
        self._tts_lock = threading.Lock()

    def load(self) -> None:
        started = time.perf_counter()
        self.tts = load_text_to_speech(str(self.config.onnx_dir), use_gpu=False)
        voice_paths = sorted(self.config.voice_styles_dir.glob("*.json"))
        if not voice_paths:
            raise RuntimeError(
                f"No voice styles found in {self.config.voice_styles_dir}."
            )
        self.voices = {
            path.stem: load_voice_style([str(path)]) for path in voice_paths
        }
        self.tts_load_seconds = time.perf_counter() - started
        if self.asr is not None and self.config.preload_asr:
            self.asr.load()
        self.ready = True

    def health(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "sample_rate": getattr(self.tts, "sample_rate", None),
            "voices": sorted(self.voices),
            "tts_load_seconds": self.tts_load_seconds,
            "asr": {
                "enabled": self.asr is not None,
                "model": self.asr.model_id if self.asr else None,
                "device": self.asr.device_label if self.asr else None,
                "load_seconds": self.asr.load_seconds if self.asr else None,
                "loaded": bool(self.asr and self.asr.pipeline is not None),
            },
        }

    def synthesize_full(
        self, request: SynthesisRequest, include_audio: bool
    ) -> dict[str, Any]:
        request = request.model_copy(
            update={"total_step": request.total_step or 8}
        )
        self._require_ready()
        style = self._get_voice(request.voice)
        request_id = uuid.uuid4().hex
        total_started = time.perf_counter()
        wait_started = time.perf_counter()
        with self._tts_lock:
            queue_wait = time.perf_counter() - wait_started
            generation_started = time.perf_counter()
            if request.seed is not None:
                np.random.seed(request.seed)
            wav_batch, duration = self.tts(
                request.text,
                request.language,
                style,
                request.total_step,
                request.speed,
            )
            generation_seconds = time.perf_counter() - generation_started
        target_samples = min(
            wav_batch.shape[1],
            int(round(float(duration[0]) * self.tts.sample_rate)),
        )
        wav = np.ascontiguousarray(wav_batch[0, :target_samples], dtype=np.float32)
        asr_result = self.evaluate_audio(request, wav)
        total_seconds = time.perf_counter() - total_started
        metrics = _build_metrics(
            request_id=request_id,
            mode="full",
            request=request,
            sample_count=len(wav),
            sample_rate=self.tts.sample_rate,
            queue_wait_seconds=queue_wait,
            generation_seconds=generation_seconds,
            total_seconds=total_seconds,
            asr_result=asr_result,
        )
        self.metrics.record(metrics)
        payload = {
            "request_id": request_id,
            "metrics": metrics.to_dict(),
            "audio": None,
        }
        if include_audio:
            payload["audio"] = {
                "encoding": "wav_pcm16_base64",
                "sample_rate": self.tts.sample_rate,
                "data": base64.b64encode(
                    _encode_wav(wav, self.tts.sample_rate)
                ).decode("ascii"),
            }
        return payload

    def stream_chunks(self, request: SynthesisRequest) -> Iterator[AudioChunk]:
        self._require_ready()
        style = self._get_voice(request.voice)
        with self._tts_lock:
            if request.seed is not None:
                np.random.seed(request.seed)
            yield from self.tts.stream(
                request.text,
                request.language,
                style,
                total_step=request.total_step,
                speed=request.speed,
                max_segment_chars=request.max_segment_chars,
                audio_chunk_ms=request.audio_chunk_ms,
            )

    def evaluate_audio(
        self, request: SynthesisRequest, wav: np.ndarray
    ) -> Optional[AsrEvaluation]:
        if not request.evaluate_asr:
            return None
        if self.asr is None:
            raise RuntimeError("ASR evaluation was requested but ASR is disabled.")
        return self.asr.evaluate(
            wav,
            self.tts.sample_rate,
            request.text,
            request.language,
        )

    def record_stream_metrics(self, metrics: GenerationMetrics) -> None:
        self.metrics.record(metrics)

    def record_failure(self) -> None:
        self.metrics.record_failure()

    def _get_voice(self, voice: str):
        try:
            return self.voices[voice]
        except KeyError as exc:
            raise ValueError(
                f"Unknown voice {voice!r}. Available: {', '.join(sorted(self.voices))}"
            ) from exc

    def _require_ready(self) -> None:
        if not self.ready or self.tts is None:
            raise RuntimeError("The Supertonic service is not ready.")


def create_app(
    config: ServerConfig,
    *,
    runtime: Optional[SupertonicService] = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = runtime or SupertonicService(config)
        app.state.runtime = service
        if not service.ready:
            await asyncio.to_thread(service.load)
        yield

    app = FastAPI(
        title="Supertonic Local API",
        version="3.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return app.state.runtime.health()

    @app.get("/v1/metrics")
    async def metrics() -> dict[str, Any]:
        return app.state.runtime.metrics.snapshot()

    @app.post("/v1/tts/batch")
    async def synthesize_batch(body: BatchSynthesisRequest) -> dict[str, Any]:
        service: SupertonicService = app.state.runtime
        if not body.items:
            raise HTTPException(status_code=422, detail="items cannot be empty")
        if len(body.items) > config.max_batch_size:
            raise HTTPException(
                status_code=422,
                detail=f"batch exceeds max size {config.max_batch_size}",
            )
        started = time.perf_counter()
        results = []
        try:
            for item in body.items:
                results.append(
                    await asyncio.to_thread(
                        service.synthesize_full, item, body.include_audio
                    )
                )
        except (RuntimeError, ValueError) as exc:
            service.record_failure()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "count": len(results),
            "wall_seconds": time.perf_counter() - started,
            "results": results,
        }

    @app.websocket("/v1/tts/stream")
    async def synthesize_stream(websocket: WebSocket) -> None:
        service: SupertonicService = app.state.runtime
        await websocket.accept()
        generator = None
        try:
            request = SynthesisRequest.model_validate(await websocket.receive_json())
            request = request.model_copy(
                update={"total_step": request.total_step or 4}
            )
            request_id = uuid.uuid4().hex
            started = time.perf_counter()
            await websocket.send_json(
                {
                    "event": "start",
                    "request_id": request_id,
                    "encoding": "pcm_s16le",
                    "sample_rate": service.tts.sample_rate,
                    "channels": 1,
                }
            )
            generator = service.stream_chunks(request)
            pcm_parts = []
            chunk_count = 0
            segment_indexes = set()
            first_audio = None
            last_ready = 0.0
            while True:
                chunk = await asyncio.to_thread(_next_chunk, generator)
                if chunk is None:
                    break
                if first_audio is None:
                    first_audio = time.perf_counter() - started
                if chunk.is_segment_start:
                    await websocket.send_json(
                        {
                            "event": "segment",
                            "segment_index": chunk.segment_index,
                            "text": chunk.text,
                            "ready_after_seconds": chunk.ready_after_seconds,
                        }
                    )
                pcm_parts.append(chunk.audio)
                chunk_count += 1
                segment_indexes.add(chunk.segment_index)
                last_ready = max(last_ready, chunk.ready_after_seconds)
                await websocket.send_bytes(_pcm16_bytes(chunk.audio))

            wav = (
                np.concatenate(pcm_parts)
                if pcm_parts
                else np.empty(0, dtype=np.float32)
            )
            asr_result = await asyncio.to_thread(service.evaluate_audio, request, wav)
            total_seconds = time.perf_counter() - started
            metrics_payload = _build_metrics(
                request_id=request_id,
                mode="stream",
                request=request,
                sample_count=len(wav),
                sample_rate=service.tts.sample_rate,
                queue_wait_seconds=0.0,
                generation_seconds=last_ready,
                total_seconds=total_seconds,
                asr_result=asr_result,
                first_audio=first_audio,
                chunk_count=chunk_count,
                segment_count=len(segment_indexes),
            )
            service.record_stream_metrics(metrics_payload)
            await websocket.send_json(
                {"event": "complete", "metrics": metrics_payload.to_dict()}
            )
            await websocket.close()
        except WebSocketDisconnect:
            return
        except Exception as exc:
            service.record_failure()
            try:
                await websocket.send_json({"event": "error", "detail": str(exc)})
                await websocket.close(code=1011)
            except Exception:
                pass
        finally:
            if generator is not None:
                await asyncio.to_thread(generator.close)

    return app


def _build_metrics(
    *,
    request_id: str,
    mode: str,
    request: SynthesisRequest,
    sample_count: int,
    sample_rate: int,
    queue_wait_seconds: float,
    generation_seconds: float,
    total_seconds: float,
    asr_result: Optional[AsrEvaluation],
    first_audio: Optional[float] = None,
    chunk_count: Optional[int] = None,
    segment_count: Optional[int] = None,
) -> GenerationMetrics:
    audio_seconds = sample_count / sample_rate if sample_rate else 0.0
    rtf = generation_seconds / audio_seconds if audio_seconds else float("inf")
    throughput = audio_seconds / generation_seconds if generation_seconds else 0.0
    return GenerationMetrics(
        request_id=request_id,
        mode=mode,
        language=request.language,
        voice=request.voice,
        speed=request.speed,
        total_step=request.total_step,
        seed=request.seed,
        text_characters=len(request.text),
        sample_rate=sample_rate,
        audio_duration_seconds=audio_seconds,
        queue_wait_seconds=queue_wait_seconds,
        generation_seconds=generation_seconds,
        total_seconds=total_seconds,
        realtime_factor=rtf,
        throughput_x_realtime=throughput,
        time_to_first_audio_seconds=first_audio,
        chunk_count=chunk_count,
        segment_count=segment_count,
        asr=asr_result.to_dict() if asr_result else None,
    )


def _next_chunk(iterator: Iterator[AudioChunk]) -> Optional[AudioChunk]:
    try:
        return next(iterator)
    except StopIteration:
        return None


def _encode_wav(wav: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, wav, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _pcm16_bytes(wav: np.ndarray) -> bytes:
    pcm = np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
    return (pcm * 32767.0).astype("<i2").tobytes()


def format_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
