"""Persistent Whisper ASR and word-error-rate evaluation."""
from __future__ import annotations

import re
import threading
import time
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

import numpy as np


@dataclass(frozen=True)
class AsrEvaluation:
    model: str
    device: str
    transcript: str
    normalized_reference: str
    normalized_transcript: str
    wer: float
    latency_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_for_wer(text: str) -> str:
    """Normalize Unicode text into comparable, whitespace-separated words."""
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    characters = []
    for char in normalized:
        category = unicodedata.category(char)
        characters.append(char if category[0] in {"L", "N"} else " ")
    return re.sub(r"\s+", " ", "".join(characters)).strip()


def word_error_rate(reference: str, hypothesis: str) -> float:
    from jiwer import wer

    normalized_reference = normalize_for_wer(reference)
    normalized_hypothesis = normalize_for_wer(hypothesis)
    if not normalized_reference:
        raise ValueError("The WER reference is empty after normalization.")
    if not normalized_hypothesis:
        return 1.0
    return float(wer(normalized_reference, normalized_hypothesis))


class WhisperAsr:
    """Load one Transformers Whisper pipeline and reuse it across requests."""

    def __init__(
        self,
        model_id: str = "openai/whisper-base",
        device: str = "auto",
        *,
        pipeline_factory: Optional[Callable[..., Any]] = None,
    ):
        self.model_id = model_id
        self.requested_device = device
        self.pipeline_factory = pipeline_factory
        self.pipeline = None
        self.device_label = "unloaded"
        self.load_seconds: Optional[float] = None
        self._lock = threading.Lock()

    def load(self) -> None:
        if self.pipeline is not None:
            return
        started = time.perf_counter()
        pipeline_factory = self.pipeline_factory
        if pipeline_factory is None:
            from transformers import pipeline

            pipeline_factory = pipeline
        pipeline_device, self.device_label = _resolve_pipeline_device(
            self.requested_device
        )
        self.pipeline = pipeline_factory(
            "automatic-speech-recognition",
            model=self.model_id,
            device=pipeline_device,
        )
        self.load_seconds = time.perf_counter() - started

    def evaluate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        reference: str,
        language: str,
    ) -> AsrEvaluation:
        self.load()
        wav = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not len(wav):
            raise ValueError("Cannot transcribe empty audio.")
        generate_kwargs = {
            "task": "transcribe",
            "condition_on_prev_tokens": False,
            "do_sample": False,
            "num_beams": 1,
            "no_repeat_ngram_size": 4,
            "max_new_tokens": max(
                32,
                min(256, int((len(wav) / sample_rate) * 12 + 32)),
            ),
            "compression_ratio_threshold": 1.35,
        }
        language_hint = _normalize_whisper_language(language)
        if language_hint:
            generate_kwargs["language"] = language_hint

        started = time.perf_counter()
        with self._lock:
            result = self.pipeline(
                {"raw": wav, "sampling_rate": int(sample_rate)},
                generate_kwargs=generate_kwargs,
            )
        latency = time.perf_counter() - started
        transcript = (
            str(result.get("text") or "")
            if isinstance(result, dict)
            else str(result)
        ).strip()
        normalized_reference = normalize_for_wer(reference)
        normalized_transcript = normalize_for_wer(transcript)
        score = word_error_rate(reference, transcript)
        return AsrEvaluation(
            model=self.model_id,
            device=self.device_label,
            transcript=transcript,
            normalized_reference=normalized_reference,
            normalized_transcript=normalized_transcript,
            wer=score,
            latency_seconds=latency,
        )


def _resolve_pipeline_device(device: str) -> tuple[int, str]:
    normalized = str(device or "auto").strip().lower()
    if normalized == "cpu":
        return -1, "cpu"
    if normalized not in {"auto", "cuda", "gpu"}:
        raise ValueError("ASR device must be one of: auto, cpu, cuda, gpu.")
    try:
        import torch

        if torch.cuda.is_available():
            return 0, "cuda:0"
    except Exception:
        pass
    if normalized in {"cuda", "gpu"}:
        raise RuntimeError("CUDA was requested for ASR but is unavailable.")
    return -1, "cpu"


def _normalize_whisper_language(language: str) -> str:
    normalized = str(language or "").strip().lower().replace("_", "-")
    return normalized.split("-", 1)[0] if normalized else ""
