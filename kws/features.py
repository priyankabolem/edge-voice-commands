"""NumPy reference log-mel features.

This is the single source of truth for the feature pipeline. The C++ implementation
must reproduce it exactly, so every step is written out explicitly:

    frame (no centering) -> periodic Hann -> zero-pad to n_fft -> |rfft|^2
    -> HTK mel filterbank -> log(mel + offset) -> per-bin normalization

Mel scale: HTK (mel = 2595 * log10(1 + f / 700)), triangular filters with peak 1.0,
no area normalization. Filter edges are evaluated at the exact FFT bin frequencies.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 16000
    frame_length: int = 400
    hop_length: int = 160
    n_fft: int = 512
    n_mels: int = 40
    f_min: float = 20.0
    f_max: float = 7600.0
    log_offset: float = 1e-6

    @classmethod
    def from_dict(cls, cfg: dict, sample_rate: int = 16000) -> "FeatureConfig":
        return cls(sample_rate=sample_rate, **cfg)


def num_frames(num_samples: int, cfg: FeatureConfig) -> int:
    if num_samples < cfg.frame_length:
        return 0
    return 1 + (num_samples - cfg.frame_length) // cfg.hop_length


def hann_window(length: int) -> np.ndarray:
    """Periodic Hann window: w[n] = 0.5 - 0.5 cos(2 pi n / N), n = 0..N-1."""
    n = np.arange(length, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / length)


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(cfg: FeatureConfig) -> np.ndarray:
    """Return the (n_mels, n_fft // 2 + 1) mel filterbank matrix."""
    n_bins = cfg.n_fft // 2 + 1
    fft_freqs = np.arange(n_bins, dtype=np.float64) * cfg.sample_rate / cfg.n_fft
    mel_points = np.linspace(hz_to_mel(cfg.f_min), hz_to_mel(cfg.f_max), cfg.n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    fbank = np.zeros((cfg.n_mels, n_bins), dtype=np.float64)
    for m in range(cfg.n_mels):
        left, center, right = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        rising = (fft_freqs - left) / (center - left)
        falling = (right - fft_freqs) / (right - center)
        fbank[m] = np.maximum(0.0, np.minimum(rising, falling))
    return fbank


class LogMelExtractor:
    """Computes un-normalized log-mel features; window and filterbank are built once."""

    def __init__(self, cfg: FeatureConfig = FeatureConfig()):
        self.cfg = cfg
        self.window = hann_window(cfg.frame_length)
        self.fbank = mel_filterbank(cfg)

    def frames(self, waveform: np.ndarray) -> np.ndarray:
        """Split a 1-D or (batch, samples) waveform into (..., n_frames, frame_length)."""
        waveform = np.asarray(waveform, dtype=np.float64)
        n = num_frames(waveform.shape[-1], self.cfg)
        idx = (
            np.arange(self.cfg.frame_length)[None, :] + self.cfg.hop_length * np.arange(n)[:, None]
        )
        return waveform[..., idx]

    def log_mel_frames(self, frames: np.ndarray) -> np.ndarray:
        """Map (..., frame_length) frames to (..., n_mels) log-mel values."""
        windowed = frames * self.window
        spectrum = np.fft.rfft(windowed, n=self.cfg.n_fft, axis=-1)
        power = spectrum.real**2 + spectrum.imag**2
        mel = power @ self.fbank.T
        return np.log(mel + self.cfg.log_offset)

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """Waveform in [-1, 1] -> float32 log-mel of shape (..., n_frames, n_mels)."""
        return self.log_mel_frames(self.frames(waveform)).astype(np.float32)


def compute_norm_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-mel-bin mean and std over all clips and frames of (n_clips, n_frames, n_mels)."""
    flat = features.reshape(-1, features.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    return mean.astype(np.float32), np.maximum(std, 1e-8).astype(np.float32)


def normalize(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((features - mean) / std).astype(np.float32)
