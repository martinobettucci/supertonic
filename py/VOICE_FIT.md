# Voice Fitting (Experimental)

`voice_fit.py` fits a custom voice to **your own recordings** by searching Supertonic's
voice-style space so the frozen ONNX synthesizer best reproduces a target speaker. It is an
*analysis-by-synthesis* approximation of voice cloning that runs entirely locally with the
shipped ONNX assets — no model retraining.

> **This is not the official Voice Builder.** Supertone's
> [Voice Builder](https://supertonic.supertone.ai/voice-builder) builds a permanent,
> high-fidelity custom voice from a trained encoder. `voice_fit.py` is an unofficial, local
> approximation that blends the shipped presets to reach the *closest reachable* timbre. For
> production-grade cloning, use Voice Builder.

**Status:** verified end-to-end against the public Supertonic-3 assets. Fitting against audio
rendered from a known preset recovers that preset as the top blend component, and the output JSON
loads in stock Supertonic unchanged. Result sharpness scales with `--budget` / `--opt-steps` and
the speaker metric (the defaults are fast-iteration starting points).

## How It Works

Supertonic ships no audio→style encoder. A "voice" is just two stored tensors (`style_ttl`,
`style_dp`) fed to the frozen ONNX models. Instead of training an inverse encoder, `voice_fit.py`:

1. Represents a candidate voice as a **convex blend** (softmax-weighted average) of the shipped
   presets (M1–M5, F1–F5). This keeps the search low-dimensional and on the valid-voice manifold.
2. Renders a few short calibration sentences with the candidate style.
3. Embeds the rendered audio with a pretrained **speaker-verification** model (Resemblyzer by
   default) and scores **cosine similarity** against a target embedding averaged from your clips.
4. Drives the search with **CMA-ES** to maximize similarity.

The RNG is pinned during the search so identical weights always render identical audio (the flow
sampler otherwise draws fresh noise per call, making the objective noisy and the search slow).

**Honest ceiling:** a convex blend of the presets can only reach voices expressible as mixtures of
those presets. This gets you the closest *reachable* timbre, not an exact clone. See
[Limitations & Next Steps](#limitations--next-steps).

## Prerequisites

1. **ONNX assets + preset voices** in `../assets` (same as `example_onnx.py`):
   ```bash
   git clone https://huggingface.co/Supertone/supertonic-3 assets
   ```
   Expected: `assets/onnx/{duration_predictor,text_encoder,vector_estimator,vocoder}.onnx`,
   `assets/onnx/{tts.json,unicode_indexer.json}`, and `assets/voice_styles/{M1..M5,F1..F5}.json`.
2. **3–10+ clean recordings of the target speaker** (your own voice), mono, any sample rate.
   Supported: `.wav`, `.flac`, `.mp3`, `.m4a`, `.ogg`. Put them in a folder, e.g. `./my_voice_clips/`.
3. `voice_fit.py` must sit next to `helper.py` (i.e. in `py/`) so it can reuse the real inference
   pipeline. It already does.

## Installation

Beyond the base requirements, voice fitting needs a speaker encoder and the optimizer:

```bash
pip install -r requirements.txt   # base: onnxruntime, numpy, soundfile, librosa
pip install resemblyzer cma scipy # voice-fitting extras
```

(`scipy` is only used for the Powell fallback when `cma` is unavailable.)

## Usage

From the `py/` directory:

```bash
python voice_fit.py \
  --onnx-dir   ../assets/onnx \
  --voice-dir  ../assets/voice_styles \
  --refs-dir   ./my_voice_clips \
  --out-style  ./me.json \
  --out-demo   ./me_demo.wav \
  --budget     200
```

This will:
- Build a target embedding by averaging the speaker embeddings of every clip in `--refs-dir`
- Search the preset blend weights with CMA-ES for up to `--budget` evaluations, printing progress
  every 10 evals
- Print the final preset mix (e.g. `F2: 61.0%`, `F4: 24.0%`, ...)
- Write the fitted voice to `--out-style` — a style JSON in the exact format Supertonic loads
- Render a demo to `--out-demo`

Then synthesize anything with your fitted voice using the stock example:

```bash
python example_onnx.py --voice-style ./me.json --text "Hello, this is my voice."
```

### Tuning the search

- Lower `--budget` (e.g. `60`) for a quick look; raise it (e.g. `400`) to let CMA-ES converge further.
- `--opt-steps` sets denoising steps *during* the search (default `4` — low = fast but noisier
  audio). `--final-steps` (default `16`) affects only the demo render.
- Pass `--calib` with sentences whose phonetic content resembles your reference clips for a cleaner
  signal.

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--refs-dir` | str | **(required)** | Folder of target-speaker audio clips |
| `--onnx-dir` | str | `../assets/onnx` | Path to ONNX model directory |
| `--voice-dir` | str | `../assets/voice_styles` | Folder of preset voice-style JSONs to blend |
| `--out-style` | str | `./fitted_voice.json` | Where to write the fitted style JSON |
| `--out-demo` | str | `./fitted_demo.wav` | Where to write the demo render |
| `--lang` | str | `en` | Language code for calibration/demo text |
| `--budget` | int | 200 | Max objective evaluations (CMA-ES) |
| `--opt-steps` | int | 4 | Denoising steps during search (low = fast/noisy) |
| `--final-steps` | int | 16 | Denoising steps for the final demo render |
| `--speed` | float | 1.0 | Speech speed factor |
| `--seed` | int | 1234 | RNG seed (pinned for a deterministic objective) |
| `--demo-text` | str | (default sentence) | Text for the demo render |
| `--calib` | str+ | (3 sentences) | Short, phonetically varied calibration sentences |

## Output Format

The fitted `--out-style` JSON matches the preset format, so it is a drop-in for any Supertonic
runtime (`example_onnx.py`, the PyPI SDK, other language SDKs, etc.):

```json
{
  "style_ttl": { "dims": [1, d1, d2], "data": [ ... ] },
  "style_dp":  { "dims": [1, e1, e2], "data": [ ... ] }
}
```

## Limitations & Next Steps

- **Convex-hull ceiling.** You can only reach mixtures of the shipped presets. To go beyond, add a
  PCA/extrapolation mode (escape the hull) or a differentiable PyTorch port that refines the raw
  style tensors by gradient. See the `HANDOFF NOTES` block at the bottom of `voice_fit.py` for the
  ranked next steps.
- **The metric matters.** Resemblyzer is the light default; swapping to SpeechBrain ECAPA-TDNN
  (stub in `_load_speaker_encoder`) is usually the single biggest quality win.
- **For production-grade cloning**, use the official
  [Voice Builder](https://supertonic.supertone.ai/voice-builder).

## Consent & License

Fit only voices you have the right to use — your own, or one you have explicit consent to
reproduce. A style produced this way is a Derivative of the Model under OpenRAIL-M; the use-based
restrictions and attribution carry over. Fitting your own voice satisfies the consent restriction.
