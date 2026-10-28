import numpy as np
import pytest

from kws.features import (
    FeatureConfig,
    LogMelExtractor,
    compute_norm_stats,
    hann_window,
    mel_filterbank,
    normalize,
    num_frames,
)

CFG = FeatureConfig()


@pytest.fixture(scope="module")
def extractor():
    return LogMelExtractor(CFG)


def test_one_second_clip_gives_98_by_40(extractor):
    audio = np.random.default_rng(0).uniform(-0.5, 0.5, 16000)
    assert num_frames(16000, CFG) == 98
    assert extractor(audio).shape == (98, 40)


def test_hann_window_is_periodic():
    w = hann_window(400)
    assert w[0] == 0.0
    assert w[200] == pytest.approx(1.0)
    # A periodic window is symmetric around N/2, so w[n] == w[N - n] for n >= 1.
    np.testing.assert_allclose(w[1:], w[1:][::-1], atol=1e-12)


def test_mel_filterbank_shape_and_range():
    fb = mel_filterbank(CFG)
    assert fb.shape == (40, 257)
    assert fb.max() <= 1.0 + 1e-12
    assert np.all(fb.sum(axis=1) > 0), "every mel filter must cover at least one FFT bin"
    freqs = np.arange(257) * CFG.sample_rate / CFG.n_fft
    outside = (freqs < CFG.f_min) | (freqs > CFG.f_max)
    assert np.all(fb[:, outside] == 0)


def test_power_spectrum_matches_direct_dft(extractor):
    frame = np.random.default_rng(1).uniform(-1, 1, CFG.frame_length)
    padded = np.zeros(CFG.n_fft)
    padded[: CFG.frame_length] = frame * hann_window(CFG.frame_length)
    k = np.arange(CFG.n_fft // 2 + 1)[:, None]
    n = np.arange(CFG.n_fft)[None, :]
    dft = (padded * np.exp(-2j * np.pi * k * n / CFG.n_fft)).sum(axis=1)
    expected = np.log(np.abs(dft) ** 2 @ extractor.fbank.T + CFG.log_offset)
    np.testing.assert_allclose(extractor.log_mel_frames(frame[None, :])[0], expected, atol=1e-9)


def test_batch_matches_single(extractor):
    audio = np.random.default_rng(2).uniform(-0.5, 0.5, (3, 16000))
    batch = extractor(audio)
    for i in range(3):
        np.testing.assert_array_equal(batch[i], extractor(audio[i]))


def test_norm_stats_normalize_to_zero_mean_unit_std():
    feats = np.random.default_rng(3).normal(5.0, 2.0, (50, 98, 40)).astype(np.float32)
    mean, std = compute_norm_stats(feats)
    normed = normalize(feats, mean, std).reshape(-1, 40)
    np.testing.assert_allclose(normed.mean(axis=0), 0.0, atol=1e-4)
    np.testing.assert_allclose(normed.std(axis=0), 1.0, atol=1e-4)
