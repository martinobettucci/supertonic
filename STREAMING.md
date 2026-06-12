# Streaming ONNX Runtime

The Python runtime exposes incremental PCM delivery through
`TextToSpeech.stream()`.

## Semantics

The current public Supertonic 3 ONNX model is not causal. Streaming therefore
operates at linguistic segment boundaries:

1. Incoming text is buffered until a sentence, clause, or bounded fallback
   segment is available.
2. The non-causal latent model completes that segment.
3. PCM blocks are emitted immediately.
4. A background producer synthesizes the next segment while the consumer plays
   the current audio.

This supports real-time playback when segment synthesis is faster than the
audio already buffered. It does not claim token-level or sample-level causal
generation.

## Python API

```python
from supertonic import load_text_to_speech, load_voice_style

tts = load_text_to_speech("assets/onnx", use_gpu=False)
style = load_voice_style(["assets/voice_styles/F1.json"])

for chunk in tts.stream(
    "The first sentence can play while the second one is generated.",
    "en",
    style,
    total_step=3,
    audio_chunk_ms=100,
):
    audio_device.write(chunk.audio)
```

The `text` argument can also be an iterable of incoming text fragments. Each
`AudioChunk` contains mono float32 PCM, timing metadata, segment indexes, and
start/end markers.

The default streaming profile uses three denoising steps. Increase the value
for quality or reduce it for lower latency. Whether a profile is real-time is
hardware and text dependent, so applications should monitor buffering rather
than assume a fixed step count is sufficient.

Run the local example:

```bash
cd py
uv run example_streaming.py --simulate-playback
```

## Windowed vocoder

`TextToSpeech.stream_vocoder()` decodes a completed latent tensor in bounded
windows. The vocoder is also non-causal, so every window includes context on
both sides and crops that context after decoding.

For the current Supertonic 3 graph:

- 20 latent positions of overlap reproduced full vocoder decoding exactly in
  local numerical tests.
- 16 positions produced approximately 82 dB SNR against full decoding.
- One latent position represents 3,072 waveform samples, about 69.7 ms at
  44.1 kHz.

Windowed vocoding bounds temporary memory and exposes PCM progressively after
the latent is ready. It does not remove latent-generation latency.

## Why the released graph cannot be made exactly causal by conversion

The released pipeline contains four graphs:

- duration predictor
- text encoder
- vector estimator
- vocoder

The vector estimator uses symmetric, edge-padded temporal ConvNeXt blocks and
length-aware text/speech cross-attention. Its output at the beginning of an
utterance depends on future latent positions and on the total latent length.
Windowing the existing graph changes both convolution context and positional
alignment. In a direct test, even 32 latent positions of overlap remained
materially different from full inference.

An ONNX graph rewrite can expose offsets and caches, but it cannot create
missing causal behavior while preserving the original model output. Changing
symmetric padding to left-only padding also changes the learned function and
requires weight adaptation.

## Path to a native causal model

A true token-to-PCM streaming variant requires a model-level fork:

1. Replace symmetric temporal convolutions with causal convolutions and expose
   per-layer state tensors.
2. Replace or adapt length-aware alignment so it accepts a global position
   offset without requiring the final utterance duration.
3. Train with chunked prefixes and randomized right-context budgets.
4. Distill from the released non-causal model to recover voice and prosody.
5. Add boundary and overlap losses to prevent clicks and repeated phonemes.
6. Export stateful ONNX graphs whose cache tensors are explicit inputs and
   outputs.
7. Validate time-to-first-audio, sustained real-time factor, boundary quality,
   and text coverage independently.

The repository currently ships inference graphs, not the original PyTorch
training model. Native causal conversion therefore needs compatible source
weights or a reimplementation followed by fine-tuning/distillation.
