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
# PCA subspace: escape the convex hull to reach ORIGINAL voices.              #
# A softmax blend is trapped inside the simplex of the K presets (every voice #
# is a *mixture*). PCA over the presets spans the same affine subspace but    #
# with UNBOUNDED coordinates, so coefficients outside the observed range      #
# extrapolate past every preset -> genuinely new timbres, not mixtures. We    #
# PCA the concatenated [style_ttl ; style_dp] jointly (they co-vary per       #
# speaker) and standardize each axis, so a coefficient reads as "how many     #
# standard deviations along this principal direction."                        #
# --------------------------------------------------------------------------- #
def build_pca(ttl_presets, dp_presets, n_comp):
    K = ttl_presets.shape[0]
    ttl_flat = ttl_presets.reshape(K, -1).astype(np.float64)
    dp_flat = dp_presets.reshape(K, -1).astype(np.float64)
    X = np.concatenate([ttl_flat, dp_flat], axis=1)            # [K, D]
    mean = X.mean(axis=0)
    _, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    rank = int(np.sum(S > 1e-8))
    n_comp = max(1, min(n_comp, rank))
    return {
        "mean": mean,
        "comps": Vt[:n_comp],                                  # [n_comp, D], orthonormal
        "scales": S[:n_comp] / np.sqrt(max(K - 1, 1)),         # per-axis std dev
        "split": ttl_flat.shape[1],
        "ttl_shape": tuple(ttl_presets.shape[1:]),
        "dp_shape": tuple(dp_presets.shape[1:]),
        "n_comp": n_comp,
    }


def pca_to_style(coef, pca):
    """Standardized PCA coefficients -> (style_ttl, style_dp)."""
    coef = np.asarray(coef, np.float64)
    x = pca["mean"] + (coef * pca["scales"]) @ pca["comps"]    # [D]
    s = pca["split"]
    ttl = x[:s].reshape(pca["ttl_shape"]).astype(np.float32)
    dp = x[s:].reshape(pca["dp_shape"]).astype(np.float32)
    return ttl, dp


def pca_describe(coef, pca, ttl_presets, dp_presets, names):
    """Report how 'original' a voice is: distance from the preset mean (in std
    units) and the single nearest preset by cosine (1.0 == identical timbre)."""
    ttl, dp = pca_to_style(coef, pca)
    v = np.concatenate([ttl.ravel(), dp.ravel()]).astype(np.float64)
    P = np.concatenate([ttl_presets.reshape(len(names), -1),
                        dp_presets.reshape(len(names), -1)], axis=1).astype(np.float64)
    cos = (P @ v) / (np.linalg.norm(P, axis=1) * np.linalg.norm(v) + 1e-9)
    j = int(np.argmax(cos))
    return float(np.linalg.norm(coef)), names[j], float(cos[j])


# --------------------------------------------------------------------------- #
# Objective: render candidate style -> embed -> 1 - cosine similarity.        #
# --------------------------------------------------------------------------- #
class Objective:
    """Maps a search vector -> style via `decode`, renders calibration audio,
    and scores 1 - cosine(speaker_embedding, target). `decode` is the only thing
    that differs between modes: a convex blend (BLEND) or a PCA reconstruction
    (PCA). `penalty` is an optional regularizer on the search vector."""

    def __init__(self, tts, encoder, target_emb, decode, calib, lang,
                 total_step, speed, seed, penalty=None):
        self.tts = tts
        self.encoder = encoder
        self.target = target_emb
        self.decode = decode                       # x -> (ttl[d1,d2], dp[e1,e2])
        self.penalty = penalty or (lambda x: 0.0)
        self.calib = calib
        self.lang = lang
        self.total_step = total_step
        self.speed = speed
        self.seed = seed
        self.n_eval = 0
        self.best = (np.inf, None)

    def render(self, ttl, dp, text):
        # Fix RNG so the same vector -> the same audio (the flow sampler draws
        # Gaussian noise per call). Without this the objective is noisy and CMA
        # converges far slower.
        np.random.seed(self.seed)
        style = Style(ttl[None], dp[None])  # add batch dim -> [1, d, d]
        wav, dur = self.tts(text, self.lang, style, self.total_step, self.speed)
        T = int(self.tts.sample_rate * float(np.asarray(dur).reshape(-1)[0]))
        return wav[0, :T]

    def __call__(self, x: np.ndarray) -> float:
        x = np.asarray(x, np.float32)
        ttl, dp = self.decode(x)
        sims = []
        for text in self.calib:
            wav = self.render(ttl, dp, text)
            emb = self.encoder.embed_wav(wav, self.tts.sample_rate)
            sims.append(float(np.dot(emb, self.target)))  # both normalized -> cosine
        loss = 1.0 - float(np.mean(sims)) + float(self.penalty(x))
        self.n_eval += 1
        if loss < self.best[0]:
            self.best = (loss, x.copy())
        if self.n_eval % 10 == 0:
            print(f"  eval {self.n_eval:4d} | loss {loss:.4f} | best {self.best[0]:.4f}")
        return loss


# --------------------------------------------------------------------------- #
# Optimizers: CMA-ES (preferred) with a SciPy Powell fallback.                #
# --------------------------------------------------------------------------- #
def optimize(objective, dim, budget, sigma0=0.6, bound=4.0):
    x0 = np.zeros(dim, np.float32)  # neutral start (blend: uniform; pca: preset mean)
    try:
        import cma
        es = cma.CMAEvolutionStrategy(
            x0, sigma0,
            {"bounds": [-bound, bound], "maxfevals": budget, "seed": 1, "verbose": -9},
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
    print(f"Wrote style -> {path}")


def _indexed(path, i):
    root, ext = os.path.splitext(path)
    return f"{root}_{i}{ext}"


def _render_demo(tts, ttl, dp, args, path):
    print(f"Rendering demo ({args.final_steps} steps) -> {path}")
    np.random.seed(args.seed)
    wav, dur = tts(args.demo_text, args.lang, Style(ttl[None], dp[None]),
                   args.final_steps, args.speed)
    T = int(tts.sample_rate * float(np.asarray(dur).reshape(-1)[0]))
    sf.write(path, wav[0, :T], tts.sample_rate)


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Per-speaker Supertonic style fitting.")
    p.add_argument("--onnx-dir", default="../assets/onnx")
    p.add_argument("--voice-dir", default="../assets/voice_styles")
    p.add_argument("--refs-dir", default=None,
                   help="Dir of target-speaker WAV clips (required unless --generate)")
    p.add_argument("--out-style", default="./fitted_voice.json")
    p.add_argument("--out-demo", default="./fitted_demo.wav")
    p.add_argument("--lang", default="en")
    p.add_argument("--mode", choices=["blend", "pca"], default="blend",
                   help="blend = convex mix of presets (safe); "
                        "pca = escape the convex hull for ORIGINAL voices")
    p.add_argument("--pca-comps", type=int, default=8,
                   help="PCA components for --mode pca / --generate (<= #presets-1)")
    p.add_argument("--pca-l2", type=float, default=0.02,
                   help="L2 penalty on PCA coefficients (keeps voices plausible)")
    p.add_argument("--generate", type=int, default=0, metavar="N",
                   help="Generate N original voices by sampling PCA space "
                        "(no reference needed); skips fitting")
    p.add_argument("--gen-sigma", type=float, default=1.4,
                   help="Std-dev of sampled PCA coefficients when generating")
    p.add_argument("--gen-seed", type=int, default=0, help="RNG seed for --generate")
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
    names, ttl_presets, dp_presets, ttl_dims, dp_dims = load_presets(args.voice_dir)

    # ---- Generation: create ORIGINAL voices by sampling PCA space (no refs) ----
    if args.generate > 0:
        pca = build_pca(ttl_presets, dp_presets, args.pca_comps)
        print(f"\nGenerating {args.generate} original voice(s) from {pca['n_comp']}-D "
              f"PCA space (sigma={args.gen_sigma}) ...")
        rng = np.random.default_rng(args.gen_seed)
        for i in range(args.generate):
            coef = rng.standard_normal(pca["n_comp"]) * args.gen_sigma
            ttl, dp = pca_to_style(coef, pca)
            norm, near, cos = pca_describe(coef, pca, ttl_presets, dp_presets, names)
            print(f"  voice {i}: |coef|={norm:4.2f} std | nearest preset {near} "
                  f"(cos {cos:.3f})")
            write_style_json(_indexed(args.out_style, i), ttl, dp, ttl_dims, dp_dims)
            _render_demo(tts, ttl, dp, args, _indexed(args.out_demo, i))
        print("Done generating original voices.")
        return

    # ---- Fitting (blend or pca) needs a reference ----
    print("Loading speaker encoder (metric)...")
    encoder = _load_speaker_encoder()
    if not args.refs_dir:
        raise SystemExit("--refs-dir is required for fitting (omit it only with --generate).")
    print(f"Building target embedding from clips in {args.refs_dir} ...")
    ref_files = [os.path.join(args.refs_dir, f) for f in sorted(os.listdir(args.refs_dir))
                 if f.lower().endswith((".wav", ".flac", ".mp3", ".m4a", ".ogg"))]
    if not ref_files:
        raise SystemExit(f"No audio clips found in {args.refs_dir}")
    target = np.mean([encoder.embed_file(f) for f in ref_files], axis=0)
    target = target / (np.linalg.norm(target) + 1e-9)
    print(f"  averaged {len(ref_files)} reference clips into target embedding")

    if args.mode == "pca":
        pca = build_pca(ttl_presets, dp_presets, args.pca_comps)
        decode = lambda x: pca_to_style(x, pca)                          # noqa: E731
        penalty = lambda x: args.pca_l2 * float(np.mean(np.square(x)))   # noqa: E731
        dim, sigma0, bound = pca["n_comp"], 1.0, 3.5
        print(f"\nFitting in PCA mode: {dim}-D coefficients (can leave the convex "
              f"hull), L2={args.pca_l2}, budget={args.budget} ...")
    else:
        decode = lambda x: blend(x, ttl_presets, dp_presets)[:2]         # noqa: E731
        penalty = None
        dim, sigma0, bound = len(names), 0.6, 4.0
        print(f"\nFitting in BLEND mode: {dim} convex weights, budget={args.budget} ...")

    objective = Objective(tts, encoder, target, decode, args.calib, args.lang,
                          args.opt_steps, args.speed, args.seed, penalty)
    x = optimize(objective, dim=dim, budget=args.budget, sigma0=sigma0, bound=bound)
    ttl, dp = decode(x)

    if args.mode == "pca":
        norm, near, cos = pca_describe(x, pca, ttl_presets, dp_presets, names)
        print(f"\nOriginal voice: {norm:.2f} std from the preset mean | nearest single "
              f"preset {near} (cos {cos:.3f}; <1 means it is NOT any mixture).")
        print("  coefficients:", [round(float(c), 2) for c in x])
    else:
        _, _, mix = blend(x, ttl_presets, dp_presets)
        print("\nFinal preset mix:")
        for n, a in sorted(zip(names, mix), key=lambda t: -t[1]):
            if a > 0.01:
                print(f"  {n:>4s}: {a:5.1%}")

    write_style_json(args.out_style, ttl, dp, ttl_dims, dp_dims)
    _render_demo(tts, ttl, dp, args, args.out_demo)
    print(f"Done. Best loss = {objective.best[0]:.4f}")


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
#   3. SUBSPACE: [DONE] PCA mode (--mode pca) and original-voice generation
#      (--generate N). PCA(mean + <=K-1 comps) spans the preset affine subspace but
#      with unbounded, standardized coefficients, so it can exceed the convex hull
#      (extrapolate past any single preset). An L2 penalty (--pca-l2) keeps samples
#      plausible. Next: decouple ttl/dp PCA, or learn the subspace from a real corpus.
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
