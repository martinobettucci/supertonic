from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from helper import AudioChunk
from supertonic.asr import AsrEvaluation
from supertonic.server import (
    MetricsRegistry,
    ServerConfig,
    create_app,
)


class FakeRuntime:
    def __init__(self):
        self.ready = True
        self.tts = SimpleNamespace(sample_rate=1_000)
        self.metrics = MetricsRegistry()
        self.stream_record = None
        self.full_request = None
        self.stream_request = None

    def load(self):
        self.ready = True

    def health(self):
        return {
            "ready": True,
            "sample_rate": 1_000,
            "voices": ["F1"],
            "asr": {"enabled": True, "loaded": True},
        }

    def synthesize_full(self, request, include_audio):
        self.full_request = request
        return {
            "request_id": "batch-request",
            "metrics": {
                "mode": "full",
                "total_step": request.total_step,
                "seed": request.seed,
                "asr": {"transcript": request.text, "wer": 0.0},
            },
            "audio": (
                {
                    "encoding": "wav_pcm16_base64",
                    "sample_rate": 1_000,
                    "data": base64.b64encode(b"RIFFfake").decode("ascii"),
                }
                if include_audio
                else None
            ),
        }

    def stream_chunks(self, request):
        self.stream_request = request
        yield AudioChunk(
            audio=np.full(100, 0.25, dtype=np.float32),
            sample_rate=1_000,
            text=request.text,
            segment_index=0,
            chunk_index=0,
            segment_chunk_index=0,
            is_segment_start=True,
            is_segment_end=False,
            ready_after_seconds=0.1,
            mode="segment-prefetch",
        )
        yield AudioChunk(
            audio=np.full(50, -0.25, dtype=np.float32),
            sample_rate=1_000,
            text=request.text,
            segment_index=0,
            chunk_index=1,
            segment_chunk_index=1,
            is_segment_start=False,
            is_segment_end=True,
            ready_after_seconds=0.2,
            mode="segment-prefetch",
        )

    def evaluate_audio(self, request, wav):
        assert len(wav) == 150
        return AsrEvaluation(
            model="fake-whisper",
            device="cpu",
            transcript=request.text,
            normalized_reference=request.text.lower(),
            normalized_transcript=request.text.lower(),
            wer=0.0,
            latency_seconds=0.01,
        )

    def record_stream_metrics(self, metrics):
        self.stream_record = metrics
        self.metrics.record(metrics)

    def record_failure(self):
        self.metrics.record_failure()


def make_client():
    runtime = FakeRuntime()
    config = ServerConfig(
        onnx_dir=Path("unused"),
        voice_styles_dir=Path("unused"),
        asr_model=None,
    )
    return TestClient(create_app(config, runtime=runtime)), runtime


def test_batch_endpoint_returns_audio_and_asr_metrics():
    client, runtime = make_client()
    with client:
        response = client.post(
            "/v1/tts/batch",
            json={
                "items": [
                    {
                        "text": "Bonjour le monde",
                        "language": "fr",
                        "voice": "F1",
                    }
                ]
            },
        )
    assert response.status_code == 200
    assert runtime.full_request.total_step is None
    assert runtime.full_request.seed == 42
    result = response.json()["results"][0]
    assert result["metrics"]["asr"]["wer"] == 0.0
    assert base64.b64decode(result["audio"]["data"]) == b"RIFFfake"


def test_websocket_stream_emits_pcm_then_final_wer_metrics():
    client, runtime = make_client()
    with client:
        with client.websocket_connect("/v1/tts/stream") as websocket:
            websocket.send_json(
                {
                    "text": "Bonjour le monde",
                    "language": "fr",
                    "voice": "F1",
                }
            )
            assert websocket.receive_json()["event"] == "start"
            assert websocket.receive_json()["event"] == "segment"
            assert len(websocket.receive_bytes()) == 200
            assert len(websocket.receive_bytes()) == 100
            completed = websocket.receive_json()

    assert completed["event"] == "complete"
    assert completed["metrics"]["asr"]["wer"] == 0.0
    assert completed["metrics"]["chunk_count"] == 2
    assert completed["metrics"]["total_step"] == 4
    assert completed["metrics"]["seed"] == 42
    assert runtime.stream_request.total_step == 4
    assert runtime.stream_request.seed == 42
    assert runtime.stream_record is not None


def test_sampling_steps_can_be_selected_for_batch_and_websocket():
    client, runtime = make_client()
    with client:
        response = client.post(
            "/v1/tts/batch",
            json={
                "items": [
                    {
                        "text": "Batch custom steps",
                        "language": "en",
                        "voice": "F1",
                        "sampling_steps": 6,
                    }
                ]
            },
        )
        assert response.status_code == 200
        assert runtime.full_request.total_step == 6
        assert response.json()["results"][0]["metrics"]["total_step"] == 6

        with client.websocket_connect("/v1/tts/stream") as websocket:
            websocket.send_json(
                {
                    "text": "Stream custom steps",
                    "language": "en",
                    "voice": "F1",
                    "sampling_steps": 5,
                }
            )
            assert websocket.receive_json()["event"] == "start"
            assert websocket.receive_json()["event"] == "segment"
            websocket.receive_bytes()
            websocket.receive_bytes()
            completed = websocket.receive_json()

    assert runtime.stream_request.total_step == 5
    assert completed["metrics"]["total_step"] == 5


def test_legacy_total_step_name_remains_supported():
    client, runtime = make_client()
    with client:
        response = client.post(
            "/v1/tts/batch",
            json={
                "items": [
                    {
                        "text": "Legacy field",
                        "total_step": 7,
                    }
                ]
            },
        )

    assert response.status_code == 200
    assert runtime.full_request.total_step == 7
