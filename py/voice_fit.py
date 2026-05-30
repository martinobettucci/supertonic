#!/usr/bin/env python3
"""
voice_fit.py  —  Per-speaker style fitting for Supertonic (analysis-by-synthesis)
=================================================================================

WHAT THIS DOES
--------------
Supertonic ships NO audio->style encoder. A "voice" is just two stored tensors,
`style_ttl` and `style_dp`, fed to the frozen ONNX models. So instead of training
an inverse encoder (which can't generalize from 10 presets), we *search* the style
space directly so the frozen synthesizer best reproduces YOUR voice.

To stay on the valid-voice manifold and keep the search low-dimensional, we don't
optimize the raw style matrices. We optimize a small weight vector `w` over the
K=10 shipped presets and use a convex blend:

        style = sum_k softmax(w)_k * preset_k          (for both ttl and dp)

We render a few calibration sentences with the candidate style, embed the audio
with a pretrained speaker-verification model, and maximize cosine similarity to a
target embedding computed from your reference recordings. CMA-ES drives the search.
No model retraining; the existing ONNX assets are used as-is.

HONEST CEILING
--------------
A convex blend of 10 presets can only reach voices expressible as mixtures of those
10. This gets you the *closest reachable* timbre, not an exact clone. To go further
you need either (a) a real differentiable PyTorch port to refine the raw style
tensors by gradient, or (b) the real Voice Builder (a true encoder trained on a
large real-speech corpus). See HANDOFF notes at the bottom.

WHAT YOU MUST PROVIDE
---------------------
1. The Supertonic ONNX assets + preset JSONs (download from HuggingFace:
   `git clone https://huggingface.co/Supertone/supertonic-3 assets`). Expected:
       assets/onnx/{duration_predictor,text_encoder,vector_estimator,vocoder}.onnx
       assets/onnx/{tts.json,unicode_indexer.json}
       assets/voice_styles/{M1..M5,F1..F5}.json
2. 3-10+ clean recordings of the target speaker (your voice), mono WAV, any SR.
3. This file placed NEXT TO the repo's `helper.py` (i.e. in `supertonic/py/`) so it
   can reuse the real inference pipeline.

INSTALL
-------
    pip install onnxruntime soundfile numpy resemblyzer cma scipy librosa

RUN
---
    python voice_fit.py \
        --onnx-dir   ../assets/onnx \
        --voice-dir  ../assets/voice_styles \
        --refs-dir   ./my_voice_clips \
        --out-style  ./me.json \
        --out-demo   ./me_demo.wav \
        --budget     200

Then use the result with stock Supertonic:
    python example_onnx.py --voice-style ./me.json --text "Hello, this is my voice."

NOTE: Verified end-to-end against the public Supertonic-3 ONNX assets. The full loop
(preset blend -> ONNX synthesis -> speaker embedding -> CMA-ES -> style JSON + demo WAV)
runs, and fitting against clips rendered from a known preset recovers that preset as the
dominant blend component. Quality scales with --budget, --opt-steps, and the speaker
metric; the defaults are fast-iteration starting points, not converged settings.
"""

import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf

# Make the repo's helper.py importable when this file is colocated in supertonic/py/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from helper import load_text_to_speech, Style  # noqa: E402
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Could not import `helper` from the Supertonic repo. Place voice_fit.py in "
        "supertonic/py/ (next to helper.py), or add that dir to PYTHONPATH.\n"
        f"Original error: {e}"
    )


# --------------------------------------------------------------------------- #
# Speaker encoder (the optimization metric).                                  #
# Default: Resemblyzer (light, single pip install). Swap to ECAPA for a       #
# stronger metric — see the stub in `_load_speaker_encoder`.                  #
# --------------------------------------------------------------------------- #
class SpeakerEncoder:
    """Maps a waveform to an L2-normalized speaker embedding."""

    def __init__(self):
        from resemblyzer import VoiceEncoder
        self._enc = VoiceEncoder()
        self._preprocess = __import__("resemblyzer").preprocess_wav

    def embed_wav(self, wav: np.ndarray, sr: int) -> np.ndarray:
        # Resemblyzer handles resampling to 16k + normalization + light VAD.
        proc = self._preprocess(wav.astype(np.float32), source_sr=sr)
        emb = self._enc.embed_utterance(proc)  # already L2-normalized
        return emb.astype(np.float32)

    def embed_file(self, path: str) -> np.ndarray:
        proc = self._preprocess(path)  # loads + resamples internally
        return self._enc.embed_utterance(proc).astype(np.float32)


def _load_speaker_encoder() -> SpeakerEncoder:
    """
    To use SpeechBrain ECAPA-TDNN instead (more discriminative), replace the body:

        from speechbrain.inference.speaker import EncoderClassifier
        clf = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")
        # wrap clf.encode_batch(...), squeeze, then L2-normalize, resample to 16k first.

    Keep the public interface identical (.embed_wav / .embed_file returning a
    normalized 1-D np.float32 vector) so the rest of the script is unchanged.
    """
    return SpeakerEncoder()


# --------------------------------------------------------------------------- #
# Preset loading + blending (the low-dim search space).                       #
# --------------------------------------------------------------------------- #
def _read_style_json(path: str):
    with open(path) as f:
        j = json.load(f)
    ttl_dims = j["style_ttl"]["dims"]            # e.g. [1, d1, d2]
    dp_dims = j["style_dp"]["dims"]
    ttl = np.asarray(j["style_ttl"]["data"], np.float32).reshape(ttl_dims[1], ttl_dims[2])
    dp = np.asarray(j["style_dp"]["data"], np.float32).reshape(dp_dims[1], dp_dims[2])
    return ttl, dp, ttl_dims, dp_dims


def load_presets(voice_dir: str):
    names = sorted(
        f[:-5] for f in os.listdir(voice_dir)
        if f.endswith(".json")
    )
    if not names:
        raise SystemExit(f"No preset *.json found in {voice_dir}")
    ttl_list, dp_list = [], []
    ttl_dims = dp_dims = None
    for n in names:
        ttl, dp, td, dd = _read_style_json(os.path.join(voice_dir, f"{n}.json"))
        ttl_list.append(ttl)
        dp_list.append(dp)
        ttl_dims, dp_dims = td, dd
    ttl_presets = np.stack(ttl_list, 0)  # [K, d1, d2]
    dp_presets = np.stack(dp_list, 0)    # [K, e1, e2]
    print(f"Loaded {len(names)} presets: {', '.join(names)}")
    return names, ttl_presets, dp_presets, ttl_dims, dp_dims


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def blend(weights: np.ndarray, ttl_presets, dp_presets):
    """Convex blend of presets via softmax weights (stays near the valid manifold)."""
    a = _softmax(weights)                          # [K], sums to 1
    ttl = np.tensordot(a, ttl_presets, axes=(0, 0))  # [d1, d2]
    dp = np.tensordot(a, dp_presets, axes=(0, 0))    # [e1, e2]
    return ttl.astype(np.float32), dp.astype(np.float32), a


# --------------------------------------------------------------------------- #
# Objective: render candidate style -> embed -> 1 - cosine similarity.        #
# --------------------------------------------------------------------------- #
class Objective:
    def __init__(self, tts, encoder, target_emb, presets, calib, lang,
                 total_step, speed, seed):
        self.tts = tts
        self.encoder = encoder
        self.target = target_emb
        self.ttl_presets, self.dp_presets = presets
        self.calib = calib
        self.lang = lang
        self.total_step = total_step
        self.speed = speed
        self.seed = seed
        self.n_eval = 0
        self.best = (np.inf, None)

    def render(self, ttl, dp, text):
        # Fix RNG so the same weights -> the same audio (the flow sampler draws
        # Gaussian noise per call). Without this the objective is noisy and CMA
        # converges far slower.
        np.random.seed(self.seed)
        style = Style(ttl[None], dp[None])  # add batch dim -> [1, d, d]
        wav, dur = self.tts(text, self.lang, style, self.total_step, self.speed)
        T = int(self.tts.sample_rate * float(np.asarray(dur).reshape(-1)[0]))
        return wav[0, :T]

    def __call__(self, weights: np.ndarray) -> float:
        ttl, dp, _ = blend(np.asarray(weights, np.float32), self.ttl_presets, self.dp_presets)
        sims = []
        for text in self.calib:
            wav = self.render(ttl, dp, text)
            emb = self.encoder.embed_wav(wav, self.tts.sample_rate)
            sims.append(float(np.dot(emb, self.target)))  # both normalized -> cosine
        loss = 1.0 - float(np.mean(sims))
        self.n_eval += 1
        if loss < self.best[0]:
            self.best = (loss, np.array(weights, np.float32))
        if self.n_eval % 10 == 0:
            print(f"  eval {self.n_eval:4d} | loss {loss:.4f} | best {self.best[0]:.4f}")
        return loss


# --------------------------------------------------------------------------- #
# Optimizers: CMA-ES (preferred) with a SciPy Powell fallback.                #
# --------------------------------------------------------------------------- #
def optimize(objective, dim, budget, sigma0=0.6):
    x0 = np.zeros(dim, np.float32)  # softmax(0) = uniform blend: neutral start
    try:
        import cma
        es = cma.CMAEvolutionStrategy(
            x0, sigma0,
            {"bounds": [-4.0, 4.0], "maxfevals": budget, "seed": 1, "verbose": -9},
        )
        while not es.stop():
            xs = es.ask()
            es.tell(xs, [objective(x) for x in xs])
        return np.asarray(es.result.xbest, np.float32)
    except ImportError:
        print("`cma` not installed — falling back to SciPy Powell (weaker).")
        from scipy.optimize import minimize
        res = minimize(objective, x0, method="Powell",
                       options={"maxfev": budget, "xtol": 1e-3, "ftol": 1e-3})
        return np.asarray(res.x, np.float32)


# --------------------------------------------------------------------------- #
# Output: write the optimized style in the exact format Supertonic loads.     #
# --------------------------------------------------------------------------- #
def write_style_json(path, ttl, dp, ttl_dims, dp_dims):
    obj = {
        "style_ttl": {"dims": list(ttl_dims), "data": ttl.flatten().tolist()},
        "style_dp": {"dims": list(dp_dims), "data": dp.flatten().tolist()},
    }
    with open(path, "w") as f:
        json.dump(obj, f)
    print(f"Wrote optimized style -> {path}")


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Per-speaker Supertonic style fitting.")
    p.add_argument("--onnx-dir", default="../assets/onnx")
    p.add_argument("--voice-dir", default="../assets/voice_styles")
    p.add_argument("--refs-dir", required=True, help="Dir of target-speaker WAV clips")
    p.add_argument("--out-style", default="./fitted_voice.json")
    p.add_argument("--out-demo", default="./fitted_demo.wav")
    p.add_argument("--lang", default="en")
    p.add_argument("--budget", type=int, default=200, help="Max objective evaluations")
    p.add_argument("--opt-steps", type=int, default=4,
                   help="Denoising steps DURING search (low = fast/noisy)")
    p.add_argument("--final-steps", type=int, default=16,
                   help="Denoising steps for the final demo render")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--demo-text",
                   default="Hello — this voice was fitted to a reference recording.")
    p.add_argument("--calib", nargs="+", default=[
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore on a bright summer morning.",
        "Numbers like 3.5 million and dates like April 3rd should sound natural.",
    ], help="Short, phonetically varied sentences rendered each evaluation.")
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading frozen Supertonic ONNX pipeline...")
    tts = load_text_to_speech(args.onnx_dir, use_gpu=False)

    print("Loading speaker encoder (metric)...")
    encoder = _load_speaker_encoder()

    print(f"Building target embedding from clips in {args.refs_dir} ...")
    ref_files = [os.path.join(args.refs_dir, f) for f in sorted(os.listdir(args.refs_dir))
                 if f.lower().endswith((".wav", ".flac", ".mp3", ".m4a", ".ogg"))]
    if not ref_files:
        raise SystemExit(f"No audio clips found in {args.refs_dir}")
    target = np.mean([encoder.embed_file(f) for f in ref_files], axis=0)
    target = target / (np.linalg.norm(target) + 1e-9)
    print(f"  averaged {len(ref_files)} reference clips into target embedding")

    names, ttl_presets, dp_presets, ttl_dims, dp_dims = load_presets(args.voice_dir)

    objective = Objective(
        tts, encoder, target, (ttl_presets, dp_presets),
        args.calib, args.lang, args.opt_steps, args.speed, args.seed,
    )

    print(f"\nOptimizing {len(names)} blend weights (budget={args.budget}) ...")
    w = optimize(objective, dim=len(names), budget=args.budget)

    ttl, dp, mix = blend(w, ttl_presets, dp_presets)
    ranking = sorted(zip(names, mix), key=lambda t: -t[1])
    print("\nFinal preset mix:")
    for n, a in ranking:
        if a > 0.01:
            print(f"  {n:>4s}: {a:5.1%}")

    write_style_json(args.out_style, ttl, dp, ttl_dims, dp_dims)

    print(f"Rendering final demo ({args.final_steps} steps) ...")
    np.random.seed(args.seed)
    wav, dur = tts(args.demo_text, args.lang, Style(ttl[None], dp[None]),
                   args.final_steps, args.speed)
    T = int(tts.sample_rate * float(np.asarray(dur).reshape(-1)[0]))
    sf.write(args.out_demo, wav[0, :T], tts.sample_rate)
    print(f"Wrote demo -> {args.out_demo}\nDone. Best loss = {objective.best[0]:.4f}")


if __name__ == "__main__":
    main()


# =========================================================================== #
# HANDOFF NOTES (for Claude Code / next iteration)
# --------------------------------------------------------------------------- #
# Quick wins / experiments, roughly in order of value:
#   1. METRIC: swap Resemblyzer -> SpeechBrain ECAPA-TDNN (see _load_speaker_encoder).
#      ECAPA is the standard for speaker verification and is more discriminative;
#      it usually moves the optimum more than any other single change.
#   2. NOISE: instead of one fixed seed, average the loss over a SMALL fixed seed set
#      (e.g. 3 seeds) for a smoother, more robust landscape — at 3x the eval cost.
#   3. SUBSPACE: add a PCA mode. With K=10 presets, PCA(mean + <=9 comps) spans the
#      same affine subspace as the blend, but lets you exceed the convex hull
#      (extrapolate past any single preset). Parameterize by component coefficients
#      and add an L2 penalty on coefficient magnitude to avoid drifting off-manifold.
#   4. DECOUPLE: give style_ttl (timbre) and style_dp (pacing) SEPARATE weight vectors
#      (search dim 2K). Often pacing wants a different mix than timbre.
#   5. CONTENT MATCH: make the calibration text match phonetic content in your refs,
#      and compare embeddings of like-vs-like content for a cleaner signal.
#
# The real ceiling-raiser (bigger project):
#   6. DIFFERENTIABLE REFINEMENT: port the 4 ONNX models to PyTorch (onnx2torch, or
#      reimplement from the SupertonicTTS / LARoPE papers and load the weights), FREEZE
#      them, then optimize the raw style matrices (not just blend weights) by gradient
#      descent on a mel/latent reconstruction + speaker-similarity loss. This is the
#      only way to reach voices outside the preset span. It's also the foundation for
#      a true generalizing encoder (= reproducing Voice Builder), which additionally
#      needs a large corpus of real (text, audio, speaker) data.
#
# License reminder: a style produced this way is a Derivative of the Model under
# OpenRAIL-M — the use-based restrictions + attribution carry over. Fitting your OWN
# voice satisfies the consent restriction.
# =========================================================================== #
