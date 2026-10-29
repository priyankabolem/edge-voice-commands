"""Evaluate streaming keyword detection and write results/streaming.json.

For each split's stream (validation and testing), runs cpp/build/stream at every
threshold in the sweep and matches detections to ground truth: a detection is correct
if it has the same label as a keyword whose speech onset is within +-tolerance_s.
Each keyword can be matched once (closest pairs first). Unmatched detections are false
alarms. The operating threshold is chosen on the validation stream only (most correct
detections minus false alarms; ties go to the higher threshold) and then reported on
the test stream.
"""

import argparse
import json
import statistics
import subprocess

from kws.data import ROOT, load_config

BUILD_DIR = ROOT / "cpp" / "build"
STREAM_DIR = ROOT / "data" / "stream"


def run_stream(cfg: dict, split: str, threshold: float, build_dir=BUILD_DIR) -> dict:
    s = cfg["stream"]
    out = subprocess.run(
        [str(build_dir / "stream"), str(ROOT / cfg["export"]["int8_pte"]), str(ROOT / "assets"),
         str(STREAM_DIR / f"{split}_stream.wav"), "--threads", str(s["threads"]),
         "--infer-every", str(s["infer_every_hops"]), "--smooth", str(s["smoothing_window"]),
         "--threshold", str(threshold), "--refractory-s", str(s["refractory_s"]),
         "--min-consecutive", str(s.get("min_consecutive", 1))],
        check=True, capture_output=True, text=True,
    )  # fmt: skip
    return json.loads(out.stdout)


def match(detections: list[dict], keywords: list[dict], tolerance_s: float) -> dict:
    pairs = sorted(
        (abs(d["time_s"] - k["onset_s"]), i, j)
        for i, d in enumerate(detections)
        for j, k in enumerate(keywords)
        if d["label"] == k["label"] and abs(d["time_s"] - k["onset_s"]) <= tolerance_s
    )
    used_d, used_k, delays = set(), set(), []
    for _, i, j in pairs:
        if i not in used_d and j not in used_k:
            used_d.add(i)
            used_k.add(j)
            delays.append(detections[i]["time_s"] - keywords[j]["onset_s"])
    false_alarms = [d for i, d in enumerate(detections) if i not in used_d]
    missed = [k for j, k in enumerate(keywords) if j not in used_k]
    return {
        "correct": len(used_k),
        "missed": len(keywords) - len(used_k),
        "false_alarms": len(false_alarms),
        "delays_s": delays,
        "false_alarm_detections": false_alarms,
        "missed_keywords": missed,
    }


def nearest_event(time_s: float, events: list[dict]) -> dict:
    """Closest ground-truth event by onset, with the detection's offset from it."""
    event = min(events, key=lambda e: abs(time_s - e["onset_s"]))
    return {"label": event["label"], "seconds_after_onset": round(time_s - event["onset_s"], 3)}


def choose_threshold(sweep: list[dict]) -> float:
    """Most correct detections minus false alarms; ties go to the higher threshold."""
    return max(sweep, key=lambda r: (r["correct"] - r["false_alarms"], r["threshold"]))["threshold"]


def evaluate_split(cfg: dict, split: str) -> dict:
    s = cfg["stream"]
    truth = json.loads((STREAM_DIR / f"{split}_stream.json").read_text())
    keywords = [e for e in truth["events"] if e["label"] in cfg["data"]["keywords"]]
    hours = truth["duration_s"] / 3600.0
    sweep, runs = [], {}
    for threshold in s["sweep_thresholds"]:
        run = run_stream(cfg, split, threshold)
        m = match(run["detections"], keywords, s["tolerance_s"])
        runs[threshold] = (run, m)
        sweep.append(
            {
                "threshold": threshold,
                "detection_rate": m["correct"] / len(keywords),
                "correct": m["correct"],
                "missed": m["missed"],
                "false_alarms": m["false_alarms"],
                "false_alarms_per_hour": round(m["false_alarms"] / hours, 2),
            }
        )
    return {
        "duration_s": truth["duration_s"],
        "num_keywords": len(keywords),
        "num_unknown": truth["num_unknown"],
        "sweep": sweep,
    }, runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", help="config variant, e.g. streaming_model")
    cfg = load_config(variant=parser.parse_args().variant)
    s = cfg["stream"]
    val, _ = evaluate_split(cfg, "validation")
    threshold = choose_threshold(val["sweep"])
    test, test_runs = evaluate_split(cfg, "testing")
    run, m = test_runs[threshold]
    truth_events = json.loads((STREAM_DIR / "testing_stream.json").read_text())["events"]
    test_row = next(r for r in test["sweep"] if r["threshold"] == threshold)

    report = {
        "detector": {
            "model": cfg["export"]["int8_pte"],
            "threads": s["threads"],
            "infer_every_hops": s["infer_every_hops"],
            "smoothing_window": s["smoothing_window"],
            "refractory_s": s["refractory_s"],
            "min_consecutive": s.get("min_consecutive", 1),
            "tolerance_s": s["tolerance_s"],
        },
        "threshold_selection": "max(correct - false_alarms) on the validation stream",
        "chosen_threshold": threshold,
        "test": {
            **{k: test_row[k] for k in ("detection_rate", "correct", "missed", "false_alarms")},
            "false_alarms_per_hour": test_row["false_alarms_per_hour"],
            "num_keywords": test["num_keywords"],
            "duration_s": test["duration_s"],
            "detection_delay_after_onset_s": {
                "median": round(statistics.median(m["delays_s"]), 3),
                "min": round(min(m["delays_s"]), 3),
                "max": round(max(m["delays_s"]), 3),
            },
            "false_alarm_detections": [
                {**d, "nearest_event": nearest_event(d["time_s"], truth_events)}
                for d in m["false_alarm_detections"]
            ],
            "missed_keywords": [
                {
                    "label": k["label"],
                    "path": k["path"],
                    "onset_s": k["onset_s"],
                    "speech_duration_s": round(k["offset_s"] - k["onset_s"], 3),
                }
                for k in m["missed_keywords"]
            ],
            "hop_latency_us": run["hop_latency_us"],
            "hop_budget_us": run["hop_budget_us"],
            "budget_overruns": run["budget_overruns"],
            "real_time_factor": run["real_time_factor"],
            "warmup_inference_us": run["warmup_inference_us"],
            "hot_loop_allocations": run["hot_loop_allocations"],
        },
        "sweep": {"validation": val["sweep"], "testing": test["sweep"]},
    }
    out = ROOT / cfg["paths"]["results_dir"] / "streaming.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("chosen_threshold", "sweep")}, indent=1))
    skip = ("false_alarm_detections", "missed_keywords")
    print(json.dumps({k: v for k, v in report["test"].items() if k not in skip},
                     indent=1))  # fmt: skip


if __name__ == "__main__":
    main()
