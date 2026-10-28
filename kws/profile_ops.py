"""Per-operator latency inside the XNNPACK delegate, from ETDump traces.

Runs cpp/build/profile_ops for FP32 and INT8, parses the traces with the ExecuTorch
Inspector, and writes results/op_profile.json: median time per XNNPACK operator,
its share of the delegate call, and MACs per layer for comparison.
"""

import json
import subprocess
import tempfile
import warnings
from pathlib import Path

from executorch.devtools import Inspector

from kws.data import ROOT, load_config

BUILD_DIR = ROOT / "cpp" / "build"


def profile(pte: Path, threads: int, iters: int, tmp: Path) -> dict:
    trace = tmp / f"{pte.stem}_t{threads}.etdp"
    subprocess.run(
        [str(BUILD_DIR / "profile_ops"), str(pte), str(trace), "--threads", str(threads),
         "--iters", str(iters)],
        check=True, capture_output=True,
    )  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # No ETRecord/debug buffer; timings are still complete.
        inspector = Inspector(etdump_path=str(trace))

    execute = next(b for b in inspector.event_blocks if b.name == "Execute")
    ops, delegate_us = [], None
    for event in execute.events:
        if event.perf_data is None:
            continue
        median_us = float(1000.0 * event.perf_data.p50)  # Inspector reports milliseconds.
        if event.is_delegated_op:
            ops.append({"op": event.name, "median_us": round(median_us, 2)})
        elif event.name == "DELEGATE_CALL":
            delegate_us = median_us
    ops_total = sum(o["median_us"] for o in ops)
    for o in ops:
        o["share_of_ops_total"] = round(o["median_us"] / ops_total, 4) if ops_total else 0.0
    return {
        "threads": threads,
        "iterations": iters,
        "delegate_call_median_us": round(delegate_us, 2),
        "sum_of_op_medians_us": round(ops_total, 2),
        "ops": ops,
        "top3": sorted(ops, key=lambda o: o["median_us"], reverse=True)[:3],
    }


def main():
    cfg = load_config()
    pcfg = cfg["benchmark"]["profile"]
    report = {
        "note": (
            "XNNPACK operator timings via ETDump (1 us resolution). Compare "
            "delegate_call_median_us with benchmark_inference_median_us to see the "
            "profiling overhead."
        ),
        "macs_per_layer": json.loads((ROOT / "results" / "model_summary.json").read_text())[
            "macs_per_layer"
        ],
    }
    benchmark = json.loads((ROOT / "results" / "benchmark.json").read_text())
    with tempfile.TemporaryDirectory() as tmp_dir:
        for model in ("fp32", "int8"):
            pte = ROOT / cfg["export"][f"{model}_pte"]
            report[model] = profile(pte, pcfg["threads"], pcfg["iters"], Path(tmp_dir))
            report[model]["benchmark_inference_median_us"] = benchmark["summary_us"][
                f"{model}_threads{pcfg['threads']}"
            ]["inference"]["median"]
            print(model, [(o["op"], o["median_us"]) for o in report[model]["top3"]])
    out = ROOT / cfg["paths"]["results_dir"] / "op_profile.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
