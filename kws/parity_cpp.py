"""Compare the C++ pipeline against the Python reference and write results/parity_cpp.json.

1. Features: C++ dump_features (WAV reader + log-mel) vs kws/features.py on 200 test
   clips that exist as WAV files.
2. End to end: C++ batch_infer (features + ExecuTorch) on the full test set vs the
   Python ExecuTorch runtime on Python features, for the FP32 and INT8 programs.
3. Hot-path heap allocations (alloc_check) and the C++ build configuration.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from kws.data import ROOT, load_config
from kws.export import run_pte
from kws.features import normalize
from kws.train import load_eval_split

BUILD_DIR = ROOT / "cpp" / "build"
ASSETS_DIR = ROOT / "assets"
NUM_FEATURE_CLIPS = 200


def run(*args) -> str:
    return subprocess.run([str(a) for a in args], check=True, capture_output=True, text=True).stdout


def cpp_features(wav_paths: list[Path], out_path: Path, n_frames: int, n_mels: int) -> np.ndarray:
    run(BUILD_DIR / "dump_features", ASSETS_DIR, out_path, *wav_paths)
    return np.fromfile(out_path, dtype="<f4").reshape(len(wav_paths), n_frames, n_mels)


def cpp_logits(pte: Path, clips_path: Path, out_path: Path, num_classes: int) -> torch.Tensor:
    run(BUILD_DIR / "batch_infer", pte, ASSETS_DIR, clips_path, out_path)
    return torch.from_numpy(np.fromfile(out_path, dtype="<f4").reshape(-1, num_classes))


def feature_parity(cfg: dict, tmp: Path) -> dict:
    features_dir = ROOT / cfg["paths"]["features_dir"]
    index = json.loads((features_dir / "testing_index.json").read_text())
    file_rows = [i for i, item in enumerate(index) if item["path"] is not None]
    rows = np.random.default_rng(cfg["seed"]).choice(file_rows, NUM_FEATURE_CLIPS, replace=False)
    rows.sort()

    stats = np.load(features_dir / "norm_stats.npz")
    reference = normalize(
        np.load(features_dir / "testing_logmel.npy", mmap_mode="r")[rows],
        stats["mean"],
        stats["std"],
    )
    dataset_dir = ROOT / cfg["paths"]["dataset_dir"]
    wavs = [dataset_dir / index[r]["path"] for r in rows]
    cpp = cpp_features(wavs, tmp / "features.f32", *reference.shape[1:])
    diff = np.abs(cpp - reference)
    return {
        "num_clips": NUM_FEATURE_CLIPS,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "target_max_abs_diff": 1e-3,
        "passed": bool(diff.max() < 1e-3),
    }


def end_to_end(cfg: dict, tmp: Path) -> dict:
    features_dir = ROOT / cfg["paths"]["features_dir"]
    clips_path = tmp / "testing.i16"
    np.ascontiguousarray(np.load(features_dir / "testing_waves.npy"), dtype="<i2").tofile(
        clips_path
    )
    stats = np.load(features_dir / "norm_stats.npz")
    test_x, test_y = load_eval_split(cfg, "testing", stats["mean"], stats["std"])

    report = {"num_test_clips": len(test_y)}
    for name in ("fp32", "int8"):
        pte = ROOT / cfg["export"][f"{name}_pte"]
        python = run_pte(pte, test_x)
        cpp = cpp_logits(pte, clips_path, tmp / f"{name}_logits.f32", python.shape[1])
        report[name] = {
            "cpp_accuracy": float((cpp.argmax(1) == test_y).float().mean()),
            "python_pte_accuracy": float((python.argmax(1) == test_y).float().mean()),
            "top1_agreement_cpp_vs_python_pte": float(
                (cpp.argmax(1) == python.argmax(1)).float().mean()
            ),
            "max_abs_logit_diff": float((cpp - python).abs().max()),
            "mean_abs_logit_diff": float((cpp - python).abs().mean()),
            "num_clips_with_logit_diff_above_1e-4": int(
                ((cpp - python).abs().amax(1) > 1e-4).sum()
            ),
        }
    return report


def main():
    cfg = load_config()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        report = {
            "build": json.loads(run(BUILD_DIR / "build_info")),
            "features": feature_parity(cfg, tmp),
            "end_to_end": end_to_end(cfg, tmp),
            "hot_path_allocations": {
                name: json.loads(
                    run(
                        BUILD_DIR / "alloc_check",
                        ROOT / cfg["export"][f"{name}_pte"],
                        ASSETS_DIR,
                        1000,
                    )
                )
                for name in ("fp32", "int8")
            },
        }
    out = ROOT / cfg["paths"]["results_dir"] / "parity_cpp.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
