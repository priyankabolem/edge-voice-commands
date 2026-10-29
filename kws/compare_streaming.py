"""Before/after comparison of streaming detection (optional extra 2).

Evaluates every combination of
    model:     baseline (Stage 2) or streaming_model (larger shifts + word-tail negatives)
    detector:  min_consecutive in compare_min_consecutive (1 = Stage 7 behaviour)
with the Stage 7 method: the threshold is chosen on the validation stream, then the
same test stream is scored. Also records each model's clip accuracy. Writes
results/streaming_comparison.json.
"""

import copy
import json
import statistics

from kws.data import ROOT, load_config
from kws.evaluate_stream import choose_threshold, evaluate_split

MODELS = {"baseline": None, "streaming_model": "streaming_model"}


def clip_accuracy(cfg: dict) -> dict:
    results = ROOT / cfg["paths"]["results_dir"]
    pte = json.loads((results / "pte_eval.json").read_text())["test_accuracy"]
    return {
        "fp32_pytorch": json.loads((results / "fp32_pytorch.json").read_text())["accuracy"],
        "int8_pte": pte["int8_pte"],
    }


def main():
    base_cfg = load_config()
    report = {
        "method": "threshold chosen on the validation stream (max correct - false alarms); "
        "test stream scored at that threshold",
        "models": {},
        "runs": [],
    }
    for model, variant in MODELS.items():
        cfg = load_config(variant=variant)
        report["models"][model] = {
            "int8_pte": cfg["export"]["int8_pte"],
            "augmentation": cfg["augmentation"],
            "clip_accuracy": clip_accuracy(cfg),
        }
        for n in base_cfg["stream"]["compare_min_consecutive"]:
            run_cfg = copy.deepcopy(cfg)
            run_cfg["stream"]["min_consecutive"] = n
            val, _ = evaluate_split(run_cfg, "validation")
            threshold = choose_threshold(val["sweep"])
            test, test_runs = evaluate_split(run_cfg, "testing")
            stream_run, m = test_runs[threshold]
            row = next(r for r in test["sweep"] if r["threshold"] == threshold)
            report["runs"].append(
                {
                    "model": model,
                    "min_consecutive": n,
                    "threshold": threshold,
                    "validation": next(r for r in val["sweep"] if r["threshold"] == threshold),
                    "test": row,
                    "test_detection_delay_median_s": round(statistics.median(m["delays_s"]), 3)
                    if m["delays_s"]
                    else None,
                    "test_hop_latency_p99_us": stream_run["hop_latency_us"]["p99"],
                    "test_false_alarm_detections": m["false_alarm_detections"],
                    "test_missed_keywords": [
                        {"label": k["label"], "onset_s": k["onset_s"]} for k in m["missed_keywords"]
                    ],
                    "sweep": {"validation": val["sweep"], "testing": test["sweep"]},
                }
            )
            print(model, n, threshold, row["correct"], row["false_alarms"])
    out = ROOT / base_cfg["paths"]["results_dir"] / "streaming_comparison.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
