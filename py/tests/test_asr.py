from __future__ import annotations

import numpy as np

from supertonic.asr import WhisperAsr, normalize_for_wer, word_error_rate


def test_normalize_for_wer_handles_unicode_and_punctuation():
    assert normalize_for_wer("Été, déjà-là !") == "été déjà là"
    assert word_error_rate("Bonjour le monde", "bonjour monde") == 1 / 3


def test_whisper_asr_reuses_loaded_pipeline_and_computes_wer():
    calls = []

    class FakePipeline:
        def __call__(self, audio, generate_kwargs):
            calls.append((audio, generate_kwargs))
            return {"text": "Bonjour le monde"}

    factory_calls = []

    def pipeline_factory(task, model, device):
        factory_calls.append((task, model, device))
        return FakePipeline()

    asr = WhisperAsr(
        "fake/whisper",
        "cpu",
        pipeline_factory=pipeline_factory,
    )
    wav = np.zeros(1_600, dtype=np.float32)
    first = asr.evaluate(wav, 16_000, "Bonjour le monde", "fr")
    second = asr.evaluate(wav, 16_000, "Bonjour le monde", "fr")

    assert first.wer == 0.0
    assert second.transcript == "Bonjour le monde"
    assert len(factory_calls) == 1
    assert len(calls) == 2
    assert calls[0][0]["sampling_rate"] == 16_000
    assert calls[0][1]["language"] == "fr"
    assert asr.device_label == "cpu"
