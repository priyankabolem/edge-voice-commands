"""Compare the C++ pipeline linked against the prebuilt ExecuTorch runtime (pip wheel)
with the same code linked against ExecuTorch built from source (scripts/build_executorch.sh).

For each build: end-to-end parity with the Python runtime on the full test set, hot-path
allocations, latency (benchmark settings from the config, with the two builds interleaved
repeat by repeat so drift affects both equally), the streaming run on the test stream at
the chosen threshold, and binary/library sizes. Writes results/runtime_comparison.json.
"""

import json
import statistics
import subprocess
import tempfile
from pathlib import Path

from kws.benchmark import ASSETS_DIR, STAGES, STATS, run_json
from kws.data import ROOT, load_config
from kws.evaluate_stream import run_stream
from kws.parity_cpp import end_to_end

BUILDS = {
    "pip_wheel": ROOT / "cpp" / "build",
    "source": ROOT / "cpp" / "build-source",
}


def wheel_library_bytes() -> dict[str, int]:
    """Shared ExecuTorch libraries the pip-wheel build loads at run time."""
    out = subprocess.run(
        [str(ROOT / ".venv" / "bin" / "python"), "-c",
         "import executorch; print(list(executorch.__path__)[0])"],
        check=True, capture_output=True, text=True,
    )  # fmt: skip
    lib_dir = Path(out.stdout.strip()) / "lib"
    names = ["libexecutorch.dylib", "libexecutorch_backend_xnnpack.dylib",
             "libexecutorch_threadpool.dylib"]  # fmt: skip
    return {name: (lib_dir / name).stat().st_size for name in names}


def latency(cfg: dict) -> dict:
    bcfg = cfg["benchmark"]
    samples = {build: {} for build in BUILDS}
    for model in ("fp32", "int8"):
        pte = ROOT / cfg["export"][f"{model}_pte"]
        for threads in bcfg["threads"]:
            key = f"{model}_threads{threads}"
            for build in BUILDS:
                samples[build][key] = []
            for _ in range(bcfg["repeats"]):
                for build, build_dir in BUILDS.items():
                    result = run_json(
                        build_dir / "bench", pte, ASSETS_DIR, "--threads", threads,
                        "--warmup", bcfg["warmup"], "--iters", bcfg["iters"],
                    )  # fmt: skip
                    if result["threads_active"] != threads:
                        raise RuntimeError(f"{build}: wrong thread count")
                    samples[build][key].append(result["latency_us"])
    return {
        build: {
            key: {
                stage: {s: round(statistics.median(r[stage][s] for r in reps), 3) for s in STATS}
                for stage in STAGES
            }
            for key, reps in per_key.items()
        }
        for build, per_key in samples.items()
    }


def main():
    cfg = load_config()
    threshold = json.loads((ROOT / "results" / "streaming.json").read_text())["chosen_threshold"]
    report = {"builds": {}, "method": {
        "latency": "benchmark config; builds interleaved per repeat; median across repeats",
        "streaming": f"test stream, INT8, threshold {threshold} (chosen in Stage 7)",
    }}  # fmt: skip
    with tempfile.TemporaryDirectory() as tmp_dir:
        for build, build_dir in BUILDS.items():
            stream = run_stream(cfg, "testing", threshold, build_dir)
            report["builds"][build] = {
                "build_info": run_json(build_dir / "build_info"),
                "end_to_end_parity": end_to_end(cfg, Path(tmp_dir), build_dir),
                "hot_path_allocations": run_json(
                    build_dir / "alloc_check", ROOT / cfg["export"]["int8_pte"], ASSETS_DIR, 1000
                )["allocations"],
                "streaming": {
                    k: stream[k]
                    for k in (
                        "hop_latency_us",
                        "budget_overruns",
                        "real_time_factor",
                        "warmup_inference_us",
                        "hot_loop_allocations",
                    )
                },  # fmt: skip
                "streaming_detections": stream["detections"],
                "stream_binary_bytes": (build_dir / "stream").stat().st_size,
            }
            print(build, "parity/streaming done")
    report["builds"]["pip_wheel"]["shared_library_bytes"] = wheel_library_bytes()
    wheel, source = report["builds"]["pip_wheel"], report["builds"]["source"]
    report["identical_streaming_detections"] = (
        wheel["streaming_detections"] == source["streaming_detections"]
    )

    lat = latency(cfg)
    for build in BUILDS:
        report["builds"][build]["latency_us"] = lat[build]
    report["inference_median_ratio_source_over_wheel"] = {
        key: round(lat["source"][key]["inference"]["median"]
                   / lat["pip_wheel"][key]["inference"]["median"], 3)
        for key in lat["source"]
    }  # fmt: skip
    out = ROOT / cfg["paths"]["results_dir"] / "runtime_comparison.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["inference_median_ratio_source_over_wheel"], indent=1))
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
