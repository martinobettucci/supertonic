from __future__ import annotations

import numpy as np

from helper import Style, TextToSpeech, iter_stream_text_chunks


class FakeVocoder:
    def __init__(self, samples_per_latent: int):
        self.samples_per_latent = samples_per_latent

    def run(self, _outputs, feeds):
        latent = feeds["latent"]
        values = latent[0, 0]
        wav = np.repeat(values, self.samples_per_latent).astype(np.float32)
        return [wav[None, :]]


def make_tts() -> TextToSpeech:
    cfg = {
        "ae": {"sample_rate": 1_000, "base_chunk_size": 4},
        "ttl": {"chunk_compress_factor": 2, "latent_dim": 2},
    }
    return TextToSpeech(
        cfg,
        text_processor=None,
        dp_ort=None,
        text_enc_ort=None,
        vector_est_ort=None,
        vocoder_ort=FakeVocoder(samples_per_latent=8),
    )


def test_incremental_text_chunking_respects_boundaries_and_limits():
    fragments = [
        "This first sentence arrives ",
        "in two fragments. A second sentence is deliberately long enough ",
        "to require another segment without losing text.",
    ]

    chunks = list(
        iter_stream_text_chunks(fragments, max_chars=60, min_chars=20)
    )

    assert chunks[0] == "This first sentence arrives in two fragments."
    assert all(len(chunk) <= 80 for chunk in chunks)
    reconstructed = " ".join(" ".join(chunks).split())
    original = " ".join("".join(fragments).split())
    assert reconstructed == original


def test_stream_vocoder_matches_full_fake_decode():
    tts = make_tts()
    latent = np.arange(40, dtype=np.float32).reshape(1, 4, 10)
    full = tts.vocoder_ort.run(None, {"latent": latent})[0][0]

    streamed = np.concatenate(
        list(tts.stream_vocoder(latent, chunk_latents=3, overlap_latents=2))
    )

    np.testing.assert_array_equal(streamed, full)


def test_stream_emits_ordered_pcm_chunks_with_segment_markers():
    tts = make_tts()
    style = Style(
        np.zeros((1, 50, 256), dtype=np.float32),
        np.zeros((1, 8, 16), dtype=np.float32),
    )

    def fake_infer_latent(text_list, lang_list, voice, total_step, speed):
        del lang_list, voice, total_step, speed
        value = float(len(text_list[0]))
        latent = np.full((1, 4, 6), value, dtype=np.float32)
        return latent, np.array([0.040], dtype=np.float32)

    tts._infer_latent = fake_infer_latent
    chunks = list(
        tts.stream(
            "First sentence is long enough. Second sentence is long enough.",
            "en",
            style,
            max_segment_chars=40,
            min_segment_chars=10,
            audio_chunk_ms=10,
            silence_duration=0.01,
            vocoder_chunk_latents=3,
            vocoder_overlap_latents=2,
        )
    )

    assert chunks
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert {chunk.segment_index for chunk in chunks} == {0, 1}
    for segment_index in (0, 1):
        segment = [c for c in chunks if c.segment_index == segment_index]
        assert segment[0].is_segment_start
        assert segment[-1].is_segment_end
        assert sum(len(c.audio) for c in segment) == 50
        assert all(c.mode == "segment-prefetch" for c in segment)
