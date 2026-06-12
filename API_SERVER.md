# Local API Server

The Python server loads the Supertonic ONNX sessions, voice styles, and
optional Whisper ASR pipeline once. It exposes full synthesis over HTTP and
incremental PCM over WebSocket.

The launcher rejects non-loopback addresses. This server is intended for local
development and benchmarking only.

## Install and start

```bash
cd py
uv sync --extra server
uv run serve_api.py --host 127.0.0.1 --port 8765
```

Startup completes only after the TTS runtime and, by default, the
`openai/whisper-base` ASR model are ready. Use `--lazy-asr` to defer Whisper
loading or `--no-asr` when requests will set `evaluate_asr` to `false`.

## Health and aggregate metrics

```bash
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/v1/metrics
```

Health reports loaded voices, sample rate, model load timings, and ASR state.
Aggregate metrics include request and failure counts, generated audio seconds,
generation throughput, mean WER, and the most recent request records.

## Full and batch synthesis

`POST /v1/tts/batch` accepts up to 16 items by default:

```json
{
  "include_audio": true,
  "items": [
    {
      "text": "Bonjour, votre rendez-vous est confirmé.",
      "language": "fr",
      "voice": "F1",
      "speed": 1.25,
      "sampling_steps": 8,
      "seed": 42,
      "evaluate_asr": true
    }
  ]
}
```

Each result contains optional base64 PCM16 WAV audio and:

- audio duration, generation time, end-to-end time, real-time factor, and
  throughput;
- model profile, voice, speed, and deterministic random seed;
- Whisper transcript, normalized reference and transcript, ASR latency, and
  word error rate.

The `sampling_steps` field accepts values from 1 to 12. Higher values generally
improve quality at the cost of latency. The legacy name
`total_step` remains accepted. The full endpoint defaults to eight denoising steps when `sampling_steps` is omitted.

## WebSocket streaming

Connect to `ws://127.0.0.1:8765/v1/tts/stream` and send one synthesis request
as JSON, including an optional `sampling_steps` value from 1 to 12. The
stream defaults to four denoising steps.

Messages arrive in this order:

1. A `start` JSON event describing `pcm_s16le`, sample rate, and channels.
2. A `segment` JSON event when each linguistic segment becomes available.
3. Binary little-endian signed 16-bit mono PCM frames.
4. A `complete` JSON event containing generation, time-to-first-audio, chunk,
   segment, throughput, ASR, and WER metrics.

An `error` JSON event closes the socket with code 1011 when synthesis or
evaluation fails.

This is segment-prefetch streaming. The released latent estimator is
non-causal, so the first audio frame follows completion of the first text
segment rather than the first input token. The following segment is generated
while the current audio is consumed. See [Streaming ONNX Runtime](STREAMING.md)
for the model-level limitation and causal-model roadmap.

## End-to-end benchmark

With the server running:

```bash
cd py
uv run benchmark_api.py \
  --base-url http://127.0.0.1:8765 \
  --mode both \
  --language fr \
  --voice F1 \
  --speed 1.25 \
  --batch-total-step 8 \
  --stream-total-step 4 \
  --seed 42 \
  --max-wer 0.50 \
  --output-dir results/api
```

The output directory contains:

- `batch.wav` and `batch_metrics.json`;
- `stream.wav` and `stream_metrics.json`;
- `health.json` and `server_metrics.json`;
- `benchmark_summary.json`, including the optional WER quality-gate result.

WER is computed from an actual Whisper transcription, not from synthesis
metadata. Scores can exceed 1.0 when the recognizer inserts more words than
the reference contains. Compare quality profiles with the same text, voice,
speed, and seed.
