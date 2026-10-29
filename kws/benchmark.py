"""Run the C++ latency benchmark for FP32/INT8 x thread counts and write results/benchmark.json.

Each configuration is run `repeats` times in separate processes. The summary reports,
per statistic, the median across repeats; all raw runs are kept as well.
"""

import json
import statistics
import subprocess

from kws.data import ROOT, load_config

BUILD_DIR = ROOT / "cpp" / "build"
ASSETS_DIR = ROOT / "assets"
STAGES = ("feature_frame", "feature_clip", "inference", "clip_end_to_end")
STATS = ("min", "mean", "median", "p95", "p99", "max")


def run_json(*args) -> dict:
    out = subprocess.run([str(a) for a in args], check=True, capture_output=True, text=True)
    return json.loads(out.stdout)


def power_state() -> dict:
    def sh(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True).stdout

    batt = sh("pmset", "-g", "batt").splitlines()
    low_power = [ln.split()[-1] for ln in sh("pmset", "-g").splitlines() if "lowpowermode" in ln]
    return {
        "power_source": batt[0].split("'")[1] if batt and "'" in batt[0] else None,
        "low_power_mode": low_power[0] == "1" if low_power else None,
    }


def measure(cfg: dict, build_dir=BUILD_DIR) -> tuple[list[dict], dict]:
    """Run bench for FP32/INT8 x thread counts; return raw runs and per-config medians."""
    bcfg = cfg["benchmark"]
    runs, summary = [], {}
    for model in ("fp32", "int8"):
        pte = ROOT / cfg["export"][f"{model}_pte"]
        for threads in bcfg["threads"]:
            repeats = []
            for r in range(bcfg["repeats"]):
                result = run_json(
                    build_dir / "bench", pte, ASSETS_DIR,
                    "--threads", threads, "--warmup", bcfg["warmup"], "--iters", bcfg["iters"],
                )  # fmt: skip
                if result["threads_active"] != threads:
                    raise RuntimeError(
                        f"requested {threads} threads, got {result['threads_active']}"
                    )
                runs.append({"model": model, "threads": threads, "repeat": r, **result})
                repeats.append(result["latency_us"])
            key = f"{model}_threads{threads}"
            summary[key] = {
                stage: {
                    s: round(statistics.median(rep[stage][s] for rep in repeats), 3) for s in STATS
                }
                for stage in STAGES
            }
            print(key, {st: summary[key][st]["median"] for st in STAGES})

    return runs, summary


def main():
    cfg = load_config()
    bcfg = cfg["benchmark"]
    runs, summary = measure(cfg)
    hop_us = 1e6 * cfg["features"]["hop_length"] / cfg["data"]["sample_rate"]
    report = {
        "method": (
            f"{bcfg['warmup']} warm-up + {bcfg['iters']} timed iterations per stage, "
            f"steady_clock, {bcfg['repeats']} process repeats; summary = median across repeats"
        ),
        "hop_budget_us": hop_us,
        "summary_us": summary,
        "build": run_json(BUILD_DIR / "build_info"),
        "power": power_state(),
        "environment": json.loads((ROOT / "results" / "environment.json").read_text()),
        "runs": runs,
    }
    out = ROOT / cfg["paths"]["results_dir"] / "benchmark.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
