"""Streaming tests: incremental features vs clip features, detector behaviour, matching."""

import json
import subprocess

import numpy as np
import pytest
import soundfile as sf

from kws.data import ROOT, load_config
from kws.evaluate_stream import match

CFG = load_config()
BUILD_DIR = ROOT / "cpp" / "build"
ASSETS_DIR = ROOT / "assets"
INT8_PTE = ROOT / CFG["export"]["int8_pte"]
HOP = CFG["features"]["hop_length"]

requires_cpp = pytest.mark.skipif(
    not (BUILD_DIR / "stream").exists() or not INT8_PTE.exists(),
    reason="C++ build or exported model missing; run make export cpp",
)


def run_stream(wav, *extra):
    out = subprocess.run(
        [str(BUILD_DIR / "stream"), str(INT8_PTE), str(ASSETS_DIR), str(wav), *map(str, extra)],
        check=True, capture_output=True, text=True,
    )  # fmt: skip
    return json.loads(out.stdout)


@requires_cpp
def test_streaming_window_equals_clip_features(tmp_path):
    audio = (np.random.default_rng(0).normal(0, 3000, 3 * 16000)).clip(-32768, 32767)
    audio = audio.astype(np.int16)
    wav = tmp_path / "noise.wav"
    sf.write(wav, audio, 16000, subtype="PCM_16")

    hop_index = 150
    run_stream(wav, "--dump-window", hop_index, tmp_path / "window.f32")
    window = np.fromfile(tmp_path / "window.f32", dtype="<f4").reshape(98, 40)

    # The newest frame at hop h starts at sample HOP * (h - 2) (3 hops of history),
    # so the 98-frame window starts 97 frames earlier.
    start = HOP * (hop_index - 2 - 97)
    segment = tmp_path / "segment.wav"
    sf.write(segment, audio[start : start + 16000], 16000, subtype="PCM_16")
    subprocess.run(
        [str(BUILD_DIR / "dump_features"), str(ASSETS_DIR), str(tmp_path / "clip.f32"),
         str(segment)],
        check=True,
    )  # fmt: skip
    clip = np.fromfile(tmp_path / "clip.f32", dtype="<f4").reshape(98, 40)
    np.testing.assert_array_equal(window, clip)


@requires_cpp
def test_silence_produces_no_detections_and_meets_budget(tmp_path):
    wav = tmp_path / "silence.wav"
    sf.write(wav, np.zeros(5 * 16000, np.int16), 16000, subtype="PCM_16")
    result = run_stream(wav, "--threshold", 0.5)
    assert result["detections"] == []
    assert result["budget_overruns"] == 0
    assert result["hot_loop_allocations"] == 0
    assert result["hops"] == 500


def test_match_counts_hits_misses_and_false_alarms():
    keywords = [{"label": "yes", "onset_s": 1.0}, {"label": "no", "onset_s": 5.0}]
    detections = [
        {"time_s": 1.3, "label": "yes"},  # hit
        {"time_s": 1.5, "label": "yes"},  # duplicate -> false alarm
        {"time_s": 5.2, "label": "go"},  # wrong label -> false alarm; "no" is missed
        {"time_s": 9.0, "label": "up"},  # nothing nearby -> false alarm
    ]
    m = match(detections, keywords, tolerance_s=0.75)
    assert (m["correct"], m["missed"], m["false_alarms"]) == (1, 1, 3)
    assert m["delays_s"] == pytest.approx([0.3])


@requires_cpp
def test_min_consecutive_suppresses_short_runs(tmp_path):
    # A run longer than the whole stream can never be reached, so nothing fires.
    stream = ROOT / "data" / "stream" / "validation_stream.wav"
    if not stream.exists():
        pytest.skip("run make streams first")
    assert run_stream(stream, "--threshold", 0.3, "--min-consecutive", 100000)["detections"] == []
    assert run_stream(stream, "--threshold", 0.3)["detections"]


def test_choose_threshold_prefers_net_correct_then_higher_threshold():
    from kws.evaluate_stream import choose_threshold

    sweep = [
        {"threshold": 0.4, "correct": 10, "false_alarms": 4},
        {"threshold": 0.5, "correct": 9, "false_alarms": 2},
        {"threshold": 0.6, "correct": 8, "false_alarms": 1},
    ]
    assert choose_threshold(sweep) == 0.6
