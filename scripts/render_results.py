"""Fill the README results tables from results/*.json.

Each table lives between `<!-- results:NAME -->` and `<!-- /results:NAME -->` markers in
README.md and is regenerated from the measurement files, so README numbers cannot drift
from the measurements. With --check, exits non-zero if README.md is out of date.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
README = ROOT / "README.md"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def us(x: float) -> str:
    return f"{x:.1f}"


def kb(n: int) -> str:
    return f"{n / 1024:.1f} KB ({n:,} B)"


def render_model() -> str:
    s = load("model_summary.json")
    t = load("train_log.json")
    return table(
        ["Parameters", "Budget", "MACs per inference", "Best epoch (val)", "Training time"],
        [
            [
                f"{s['num_params']:,}",
                f"{s['param_budget']:,}",
                f"{s['macs_total']:,}",
                f"{t['best_epoch']} ({pct(t['best_val_acc'])})",
                f"{t['train_seconds'] / 60:.1f} min ({t['device']})",
            ]
        ],
    )


def render_accuracy() -> str:
    fp32, int8 = load("fp32_pytorch.json"), load("int8_pytorch.json")
    pte, cpp = load("pte_eval.json"), load("parity_cpp.json")["end_to_end"]
    acc = pte["test_accuracy"]
    rows = [
        ["FP32", "PyTorch (eager)", pct(fp32["accuracy"])],
        ["INT8", "PyTorch (PT2E reference)", pct(int8["accuracy"])],
        ["FP32", "ExecuTorch `.pte`, Python runtime", pct(acc["fp32_pte"])],
        ["INT8", "ExecuTorch `.pte`, Python runtime", pct(acc["int8_pte"])],
        ["FP32", "C++ features + ExecuTorch", pct(cpp["fp32"]["cpp_accuracy"])],
        ["INT8", "C++ features + ExecuTorch", pct(cpp["int8"]["cpp_accuracy"])],
    ]
    return (
        table(["Precision", "Runtime", "Test accuracy"], rows)
        + f"\n\nTest set: {fp32['num_examples']:,} clips, 12 classes."
    )


def render_sizes() -> str:
    sizes = load("pte_eval.json")["file_size_bytes"]
    return table(
        ["Artifact", "Size"],
        [
            ["FP32 PyTorch checkpoint (`.pt`)", kb(sizes["fp32_checkpoint_pt"])],
            ["FP32 ExecuTorch program (`.pte`)", kb(sizes["fp32_pte"])],
            ["INT8 ExecuTorch program (`.pte`)", kb(sizes["int8_pte"])],
        ],
    )


def render_quantization() -> str:
    q = load("quant_experiments.json")
    rows = [
        [
            r["calibration_samples"],
            "per-channel" if r["per_channel_weights"] else "per-tensor",
            pct(r["test_accuracy"]),
            f"{r['drop_points']:.2f}",
            pct(r["top1_agreement_with_fp32"]),
            f"{r['logit_sqnr_db']:.2f}",
        ]
        for r in q["runs"]
    ]
    return table(
        [
            "Calibration clips",
            "Weights",
            "Test accuracy",
            "Drop (points)",
            "Top-1 agreement with FP32",
            "Logit SQNR (dB)",
        ],
        rows,
    )


def render_parity() -> str:
    py, cpp = load("parity_python.json"), load("parity_cpp.json")
    f32, i8 = py["fp32_pytorch_vs_fp32_pte"], py["int8_pytorch_vs_int8_pte"]
    e2e, feat = cpp["end_to_end"], cpp["features"]
    rows = [
        [
            "PyTorch FP32 vs `.pte` FP32",
            f"{f32['max_abs_logit_diff']:.2e}",
            pct(f32["top1_agreement"]),
        ],
        [
            "PyTorch INT8 vs `.pte` INT8",
            f"{i8['max_abs_logit_diff']:.3f} "
            f"({i8['max_abs_logit_diff_in_steps']:.0f} output steps)",
            pct(i8["top1_agreement"]),
        ],
        [
            "C++ FP32 vs Python `.pte` FP32",
            f"{e2e['fp32']['max_abs_logit_diff']:.2e}",
            pct(e2e["fp32"]["top1_agreement_cpp_vs_python_pte"]),
        ],
        [
            "C++ INT8 vs Python `.pte` INT8",
            f"{e2e['int8']['max_abs_logit_diff']:.3f}",
            pct(e2e["int8"]["top1_agreement_cpp_vs_python_pte"]),
        ],
    ]
    allocs = cpp["hot_path_allocations"]["int8"]
    return (
        table(["Comparison (full test set)", "Max abs logit diff", "Top-1 agreement"], rows)
        + f"\n\nC++ vs NumPy log-mel features on {feat['num_clips']} test WAV files: max abs "
        f"difference {feat['max_abs_diff']:.2e} (target < {feat['target_max_abs_diff']:.0e}).\n"
        f"Heap allocations over {allocs['iterations']} iterations of the per-frame, per-clip and "
        f"per-inference paths: {sum(allocs['allocations'].values())}."
    )


def render_latency() -> str:
    bench = load("benchmark.json")
    b = bench["summary_us"]
    rows = []
    configs = [
        ("fp32_threads1", "FP32, 1 thread"),
        ("fp32_threads4", "FP32, 4 threads"),
        ("int8_threads1", "INT8, 1 thread"),
        ("int8_threads4", "INT8, 4 threads"),
    ]
    for key, label in configs:
        inf = b[key]["inference"]
        rows.append(
            [
                label,
                us(inf["median"]),
                us(inf["p95"]),
                us(inf["p99"]),
                us(inf["max"]),
                us(b[key]["feature_frame"]["median"]),
                us(b[key]["feature_clip"]["median"]),
            ]
        )
    headers = [
        "Config",
        "Inference median",
        "p95",
        "p99",
        "max",
        "Feature frame median",
        "Feature clip median",
    ]
    return table(headers, rows) + f"\n\nAll values in microseconds. Method: {bench['method']}."


def render_profile() -> str:
    p = load("op_profile.json")
    rows = []
    for model in ("fp32", "int8"):
        for o in p[model]["top3"]:
            rows.append([model.upper(), o["op"], us(o["median_us"]), pct(o["share_of_ops_total"])])
    return table(["Model", "XNNPACK operator", "Median (us)", "Share of inference"], rows) + (
        f"\n\nSingle thread, {p['int8']['iterations']} iterations, recorded with ETDump."
    )


def render_streaming() -> str:
    s = load("streaming.json")
    t = s["test"]
    lat = t["hop_latency_us"]
    summary = table(
        ["Metric (test stream)", "Value"],
        [
            ["Stream length / keywords", f"{t['duration_s']:.1f} s / {t['num_keywords']}"],
            ["Threshold (chosen on validation stream)", s["chosen_threshold"]],
            ["Detection rate", f"{pct(t['detection_rate'])} ({t['correct']}/{t['num_keywords']})"],
            ["False alarms", f"{t['false_alarms']} ({t['false_alarms_per_hour']:.1f} per hour)"],
            [
                "Median detection delay after speech onset",
                f"{t['detection_delay_after_onset_s']['median']:.3f} s",
            ],
            ["Per-hop processing p99 / max", f"{us(lat['p99'])} / {us(lat['max'])} us"],
            ["Hop budget / overruns", f"{t['hop_budget_us']:.0f} us / {t['budget_overruns']}"],
            ["Real-time factor", f"{t['real_time_factor']:.4f}"],
            ["Heap allocations in streaming loop", t["hot_loop_allocations"]],
        ],
    )
    sweep = table(
        ["Threshold", "Validation: detected / false alarms", "Test: detected / false alarms"],
        [
            [
                v["threshold"],
                f"{v['correct']} / {v['false_alarms']}",
                f"{t_['correct']} / {t_['false_alarms']}",
            ]
            for v, t_ in zip(s["sweep"]["validation"], s["sweep"]["testing"])
        ],
    )
    return summary + "\n\nThreshold sweep:\n\n" + sweep


def render_environment() -> str:
    e = load("environment.json")
    b = load("parity_cpp.json")["build"]
    pk = e["packages"]
    cores = e["cpu_cores"]
    rows = [
        ["OS", f"macOS {e['os_version']} ({e['machine']})"],
        [
            "CPU",
            f"{e['cpu_model']}, {cores['performance']} performance + "
            f"{cores['efficiency']} efficiency cores",
        ],
        ["Python", e["python"]],
        ["PyTorch / torchao / ExecuTorch", f"{pk['torch']} / {pk['torchao']} / {pk['executorch']}"],
        ["C++ compiler / flags", f"{b['compiler']} / `{b['cxx_flags']}` ({b['build_type']})"],
        ["ExecuTorch C++ runtime", b["executorch_runtime"]],
        ["CMake", e["cmake"]],
    ]
    return table(["", ""], rows)


SECTIONS = {
    "model": render_model,
    "accuracy": render_accuracy,
    "sizes": render_sizes,
    "quantization": render_quantization,
    "parity": render_parity,
    "latency": render_latency,
    "profile": render_profile,
    "streaming": render_streaming,
    "environment": render_environment,
}


def render(text: str) -> str:
    for name, fn in SECTIONS.items():
        pattern = re.compile(rf"(<!-- results:{name} -->).*?(<!-- /results:{name} -->)", re.S)
        if not pattern.search(text):
            raise SystemExit(f"README.md is missing the results:{name} markers")
        text = pattern.sub(lambda m, fn=fn: f"{m.group(1)}\n{fn()}\n{m.group(2)}", text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if README.md is out of date")
    args = parser.parse_args()
    current = README.read_text()
    updated = render(current)
    if args.check:
        if updated != current:
            print("README.md results are out of date; run: make readme", file=sys.stderr)
            sys.exit(1)
        print("README.md results are up to date")
        return
    README.write_text(updated)
    print("Updated README.md results tables")


if __name__ == "__main__":
    main()
