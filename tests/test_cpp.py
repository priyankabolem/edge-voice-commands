"""Tests that call the C++ binaries. Skipped if the C++ build or exported files are missing."""

import json
import subprocess

import numpy as np
import pytest
import soundfile as sf
import torch

from kws.data import ROOT, label_names, load_config
from kws.export import run_pte
from kws.features import FeatureConfig, LogMelExtractor, normalize

CFG = load_config()
BUILD_DIR = ROOT / "cpp" / "build"
ASSETS_DIR = ROOT / "assets"
FEATURES_DIR = ROOT / CFG["paths"]["features_dir"]
INT8_PTE = ROOT / CFG["export"]["int8_pte"]

pytestmark = pytest.mark.skipif(
    not (BUILD_DIR / "batch_infer").exists() or not (ASSETS_DIR / "config.txt").exists(),
    reason="C++ build or assets missing; run make cpp",
)
requires_model = pytest.mark.skipif(not INT8_PTE.exists(), reason="run make export first")


def run(*args, check=True):
    return subprocess.run([str(a) for a in args], capture_output=True, text=True, check=check)


@pytest.fixture
def tone_wav(tmp_path):
    def make(name, samplerate=16000, channels=1, subtype="PCM_16", seconds=1.0):
        t = np.arange(int(samplerate * seconds)) / samplerate
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)
        if channels > 1:
            audio = np.stack([audio] * channels, axis=1)
        path = tmp_path / name
        sf.write(path, audio, samplerate, subtype=subtype)
        return path

    return make


def test_cpp_features_match_python_reference(tone_wav, tmp_path):
    # A short clip also exercises zero-padding to 1 s.
    wav = tone_wav("tone.wav", seconds=0.6)
    out = tmp_path / "features.f32"
    run(BUILD_DIR / "dump_features", ASSETS_DIR, out, wav)
    cpp = np.fromfile(out, dtype="<f4").reshape(98, 40)

    audio, _ = sf.read(wav, dtype="int16")
    audio = np.pad(audio, (0, 16000 - len(audio))).astype(np.float32) / 32768.0
    extractor = LogMelExtractor(FeatureConfig.from_dict(CFG["features"]))
    stats = np.load(FEATURES_DIR / "norm_stats.npz")
    reference = normalize(extractor(audio), stats["mean"], stats["std"])
    assert np.abs(cpp - reference).max() < 1e-3


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"channels": 2}, "expected mono"),
        ({"samplerate": 44100}, "expected 16000 Hz"),
        ({"subtype": "PCM_24"}, "expected 16-bit"),
        ({"subtype": "FLOAT"}, "expected PCM"),
    ],
)
def test_wav_reader_rejects_unsupported_formats(tone_wav, tmp_path, kwargs, message):
    wav = tone_wav("bad.wav", **kwargs)
    result = run(BUILD_DIR / "dump_features", ASSETS_DIR, tmp_path / "out.f32", wav, check=False)
    assert result.returncode != 0
    assert message in result.stderr


@requires_model
def test_classify_wav_prints_known_label(tone_wav):
    result = run(BUILD_DIR / "classify_wav", INT8_PTE, ASSETS_DIR, tone_wav("tone.wav"))
    label, score = result.stdout.split()
    assert label in label_names(CFG)
    assert 0.0 < float(score) <= 1.0


@requires_model
def test_batch_infer_matches_python_pte(tmp_path):
    waves = np.load(FEATURES_DIR / "testing_waves.npy", mmap_mode="r")[:100]
    clips = tmp_path / "clips.i16"
    np.ascontiguousarray(waves, dtype="<i2").tofile(clips)
    out = tmp_path / "logits.f32"
    run(BUILD_DIR / "batch_infer", INT8_PTE, ASSETS_DIR, clips, out)
    cpp = np.fromfile(out, dtype="<f4").reshape(100, 12)

    stats = np.load(FEATURES_DIR / "norm_stats.npz")
    feats = normalize(
        np.load(FEATURES_DIR / "testing_logmel.npy", mmap_mode="r")[:100],
        stats["mean"],
        stats["std"],
    )
    python = run_pte(INT8_PTE, torch.from_numpy(feats).unsqueeze(1)).numpy()
    assert (cpp.argmax(1) == python.argmax(1)).mean() == 1.0


@requires_model
def test_hot_paths_do_not_allocate():
    result = run(BUILD_DIR / "alloc_check", INT8_PTE, ASSETS_DIR, 200)
    counts = json.loads(result.stdout)["allocations"]
    assert counts == {"feature_frame": 0, "feature_clip": 0, "inference": 0}
