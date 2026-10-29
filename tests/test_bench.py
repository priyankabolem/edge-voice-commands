"""Tests for the C++ benchmark and per-operator profiler. Skipped without a C++ build."""

import json
import subprocess
import warnings

import pytest
from executorch.devtools import Inspector

from kws.data import ROOT, load_config

CFG = load_config()
BUILD_DIR = ROOT / "cpp" / "build"
INT8_PTE = ROOT / CFG["export"]["int8_pte"]

pytestmark = pytest.mark.skipif(
    not (BUILD_DIR / "bench").exists() or not INT8_PTE.exists(),
    reason="C++ build or exported model missing; run make export cpp",
)


@pytest.mark.parametrize("threads", [1, 2])
def test_bench_reports_ordered_stats_and_applies_threads(threads):
    out = subprocess.run(
        [str(BUILD_DIR / "bench"), str(INT8_PTE), str(ROOT / "assets"), "--threads", str(threads),
         "--warmup", "5", "--iters", "50"],
        check=True, capture_output=True, text=True,
    )  # fmt: skip
    result = json.loads(out.stdout)
    assert result["threads_active"] == threads
    for stage, s in result["latency_us"].items():
        assert 0 < s["min"] <= s["median"] <= s["p95"] <= s["p99"] <= s["max"], stage


def test_profile_ops_records_xnnpack_operators(tmp_path):
    trace = tmp_path / "int8.etdp"
    subprocess.run(
        [str(BUILD_DIR / "profile_ops"), str(INT8_PTE), str(trace), "--iters", "5"],
        check=True,
        capture_output=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inspector = Inspector(etdump_path=str(trace))
    execute = next(b for b in inspector.event_blocks if b.name == "Execute")
    names = [e.name for e in execute.events if e.is_delegated_op]
    assert sum("Convolution" in n for n in names) == 5
    assert any(e.name == "DELEGATE_CALL" for e in execute.events)


@pytest.mark.skipif(not (BUILD_DIR / "bench_fft").exists(), reason="run make cpp first")
def test_neon_fft_is_bit_identical_to_scalar():
    out = subprocess.run(
        [str(BUILD_DIR / "bench_fft"), str(ROOT / "assets"), "--warmup", "10", "--iters", "100",
         "--check-frames", "2000"],
        capture_output=True, text=True,
    )  # fmt: skip
    if "no NEON path" in out.stderr:
        pytest.skip("build has no NEON path")
    result = json.loads(out.stdout)
    assert result["frames_with_power_mismatch"] == 0
    assert result["frames_with_feature_mismatch"] == 0
    assert out.returncode == 0
