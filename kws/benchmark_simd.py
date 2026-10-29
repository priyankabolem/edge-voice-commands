"""Scalar vs NEON FFT (optional extra 3) and write results/simd.json.

1. cpp/build/bench_fft: bit-exactness of the two FFT paths on random frames, and the
   latency of the FFT, one feature frame and one 1 s clip with each path
   (median across process repeats).
2. Fallback check: INT8 logits on the full test set from the default build (NEON) and
   from cpp/build-scalar (-DKWS_ENABLE_NEON=OFF, no NEON code compiled) must be
   bitwise identical.
"""

import json
import statistics
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from kws.data import ROOT, load_config

ASSETS_DIR = ROOT / "assets"
BUILDS = {"neon": ROOT / "cpp" / "build", "scalar_fallback": ROOT / "cpp" / "build-scalar"}
REPEATS = 3


def bench(build_dir: Path) -> dict:
    out = subprocess.run(
        [str(build_dir / "bench_fft"), str(ASSETS_DIR)], check=True, capture_output=True, text=True
    )
    return json.loads(out.stdout)


def main():
    cfg = load_config()
    runs = [bench(BUILDS["neon"]) for _ in range(REPEATS)]
    stages = runs[0]["latency_us"].keys()
    latency = {
        stage: {s: round(statistics.median(r["latency_us"][stage][s] for r in runs), 3)
                for s in ("min", "median", "p95", "p99", "max")}
        for stage in stages
    }  # fmt: skip
    speedup = {
        name: round(latency[f"{name}_scalar"]["median"] / latency[f"{name}_neon"]["median"], 2)
        for name in ("fft_power", "feature_frame", "feature_clip")
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        clips = tmp / "test.i16"
        waves = np.load(ROOT / cfg["paths"]["features_dir"] / "testing_waves.npy")
        np.ascontiguousarray(waves, dtype="<i2").tofile(clips)
        logits = {}
        for name, build_dir in BUILDS.items():
            out = tmp / f"{name}.f32"
            subprocess.run(
                [str(build_dir / "batch_infer"), str(ROOT / cfg["export"]["int8_pte"]),
                 str(ASSETS_DIR), str(clips), str(out)],
                check=True, capture_output=True,
            )  # fmt: skip
            logits[name] = out.read_bytes()

    report = {
        "kernel": "radix-2 FFT butterflies (stages with >= 4 butterflies per block) and power "
        "spectrum, NEON float32x4 with explicit fused multiply-adds",
        "bit_exactness": {
            # The frames come from a fixed seed, so each run checks the same frames.
            "random_frames_checked": runs[0]["check_frames"],
            "runs": REPEATS,
            "frames_with_power_mismatch": sum(r["frames_with_power_mismatch"] for r in runs),
            "frames_with_feature_mismatch": sum(r["frames_with_feature_mismatch"] for r in runs),
            "int8_logits_neon_vs_scalar_fallback_build_identical": logits["neon"]
            == logits["scalar_fallback"],
            "test_clips": int(waves.shape[0]),
        },
        "fft_share_of_scalar_frame": round(
            latency["fft_power_scalar"]["median"] / latency["feature_frame_scalar"]["median"], 3
        ),
        "speedup_median": speedup,
        "latency_us": latency,
        "method": f"bench_fft: {runs[0]['iters']} timed iterations (clip: {runs[0]['clip_iters']}) "
        f"after warm-up, {REPEATS} process repeats; median across repeats",
    }
    out = ROOT / cfg["paths"]["results_dir"] / "simd.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("bit_exactness", "fft_share_of_scalar_frame",
                                             "speedup_median")}, indent=1))  # fmt: skip


if __name__ == "__main__":
    main()
