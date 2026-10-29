import json
import random

import numpy as np
import pytest

from kws.data import (
    ROOT,
    SPLITS,
    Augmenter,
    NoiseRegion,
    label_names,
    load_config,
    pad_or_trim,
    speaker_id,
)
from kws.features import compute_norm_stats

CFG = load_config()
FEATURES_DIR = ROOT / CFG["paths"]["features_dir"]

requires_features = pytest.mark.skipif(
    not (FEATURES_DIR / "norm_stats.npz").exists(),
    reason="precomputed features not found; run scripts/precompute_features.py",
)


def test_labels_are_the_twelve_expected_strings():
    # Guards against YAML parsing bare yes/no/on/off as booleans.
    expected = [
        "yes",
        "no",
        "up",
        "down",
        "left",
        "right",
        "on",
        "off",
        "stop",
        "go",
        "_unknown_",
        "_silence_",
    ]
    assert label_names(CFG) == expected


def test_pad_or_trim():
    assert len(pad_or_trim(np.ones(100, np.int16), 16000)) == 16000
    assert len(pad_or_trim(np.ones(20000, np.int16), 16000)) == 16000
    padded = pad_or_trim(np.ones(10, np.int16), 16)
    assert padded[:10].sum() == 10 and padded[10:].sum() == 0


def test_time_shift_moves_signal_and_zero_fills():
    aug = Augmenter([], CFG)
    audio = np.arange(1, 11, dtype=np.float32)
    np.testing.assert_array_equal(aug.time_shift(audio, 3)[:3], 0)
    np.testing.assert_array_equal(aug.time_shift(audio, 3)[3:], audio[:-3])
    np.testing.assert_array_equal(aug.time_shift(audio, -3)[:-3], audio[3:])
    np.testing.assert_array_equal(aug.time_shift(audio, -3)[-3:], 0)


def test_augmenter_keeps_length_and_range():
    noise = NoiseRegion("n", np.full(48000, 30000, np.int16), 0, 48000)
    aug = Augmenter([noise], CFG)
    rng = random.Random(0)
    audio = np.full(16000, 0.99, np.float32)
    for _ in range(20):
        out = aug(audio, rng)
        assert out.shape == (16000,)
        assert np.abs(out).max() <= 1.0


@requires_features
def test_splits_have_no_overlapping_files_or_speakers():
    files, speakers = {}, {}
    for split in SPLITS:
        items = json.loads((FEATURES_DIR / f"{split}_index.json").read_text())
        files[split] = {it["path"] for it in items if it["path"] is not None}
        speakers[split] = {speaker_id(p) for p in files[split]}
    for a, b in [("training", "validation"), ("training", "testing"), ("validation", "testing")]:
        assert not files[a] & files[b], f"{a} and {b} share files"
        assert not speakers[a] & speakers[b], f"{a} and {b} share speakers"


@requires_features
def test_silence_crops_use_disjoint_noise_regions():
    ranges = {}
    for split in SPLITS:
        items = json.loads((FEATURES_DIR / f"{split}_index.json").read_text())
        for it in items:
            if it["path"] is None:
                lo, hi = ranges.setdefault((split, it["noise"]), [np.inf, -np.inf])
                ranges[(split, it["noise"])] = [
                    min(lo, it["offset"]),
                    max(hi, it["offset"] + CFG["data"]["clip_samples"]),
                ]
    for (split_a, noise_a), (lo_a, hi_a) in ranges.items():
        for (split_b, noise_b), (lo_b, hi_b) in ranges.items():
            if noise_a == noise_b and split_a != split_b:
                assert hi_a <= lo_b or hi_b <= lo_a, f"{noise_a}: {split_a}/{split_b} overlap"


@requires_features
def test_norm_stats_come_from_training_data_only():
    stats = np.load(FEATURES_DIR / "norm_stats.npz")
    train_mean, train_std = compute_norm_stats(np.load(FEATURES_DIR / "training_logmel.npy"))
    np.testing.assert_allclose(stats["mean"], train_mean, rtol=1e-6)
    np.testing.assert_allclose(stats["std"], train_std, rtol=1e-6)

    val_mean, _ = compute_norm_stats(np.load(FEATURES_DIR / "validation_logmel.npy"))
    assert not np.allclose(stats["mean"], val_mean, rtol=1e-6)


@requires_features
def test_saved_features_have_expected_shape():
    for split in SPLITS:
        feats = np.load(FEATURES_DIR / f"{split}_logmel.npy", mmap_mode="r")
        labels = np.load(FEATURES_DIR / f"{split}_labels.npy")
        assert feats.shape[1:] == (98, 40)
        assert feats.shape[0] == labels.shape[0]
        assert np.isfinite(feats[:500]).all()
        assert set(np.unique(labels)) == set(range(len(label_names(CFG)))), (
            f"{split}: missing a class"
        )


def test_keyword_shift_tail_negatives_and_keep_constraint():
    from kws.data import keyword_shift

    aug = {"time_shift_ms": 300, "keyword_keep_min": 0.8, "tail_negative_prob": 0.5,
           "tail_visible": [0.15, 0.6]}  # fmt: skip
    onset, offset, n = 6000, 10000, 16000
    rng = random.Random(0)
    tails = 0
    for _ in range(500):
        shift, is_tail = keyword_shift(onset, offset, rng, aug, 16000, n)
        start, end = onset + shift, offset + shift
        inside = min(end, n) - max(start, 0)
        if is_tail:
            tails += 1
            assert start < 0, "a tail negative must cut off the start of the word"
            assert 0.15 * 4000 - 1 <= inside <= 0.6 * 4000 + 1
        else:
            assert abs(shift) <= 4800 and inside >= 0.8 * 4000
    assert 200 < tails < 300
