"""Build the train/validation/test splits and precompute reference log-mel features.

Writes to data/features/:
    {split}_index.json    example list (file path or silence-crop recipe) and label
    {split}_waves.npy     int16 waveforms, (n, 16000); used for on-the-fly augmentation
    {split}_logmel.npy    float32 un-normalized log-mel, (n, 98, 40)
    {split}_labels.npy    int64 label ids
    norm_stats.npz        per-mel-bin mean/std computed on the training split only
and results/data_summary.json.
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kws.data import (  # noqa: E402
    NOISE_DIR,
    ROOT,
    SPLITS,
    build_split_index,
    int16_to_float,
    label_names,
    load_config,
    load_example,
    read_wav_int16,
)
from kws.features import FeatureConfig, LogMelExtractor, compute_norm_stats  # noqa: E402

BATCH_SIZE = 256


def main():
    cfg = load_config()
    dataset_dir = ROOT / cfg["paths"]["dataset_dir"]
    out_dir = ROOT / cfg["paths"]["features_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_samples = cfg["data"]["clip_samples"]
    labels = label_names(cfg)
    label_to_id = {name: i for i, name in enumerate(labels)}
    extractor = LogMelExtractor(
        FeatureConfig.from_dict(cfg["features"], cfg["data"]["sample_rate"])
    )

    start_time = time.perf_counter()
    index = build_split_index(dataset_dir, cfg, seed=cfg["seed"])
    noise_audio = {p.name: read_wav_int16(p) for p in (dataset_dir / NOISE_DIR).glob("*.wav")}

    summary = {"labels": labels, "splits": {}}
    for split in SPLITS:
        items = index[split]
        waves = np.stack([load_example(it, dataset_dir, noise_audio, clip_samples) for it in items])
        feats = np.concatenate(
            [
                extractor(int16_to_float(waves[i : i + BATCH_SIZE]))
                for i in range(0, len(waves), BATCH_SIZE)
            ]
        )
        label_ids = np.array([label_to_id[it["label"]] for it in items], dtype=np.int64)

        (out_dir / f"{split}_index.json").write_text(json.dumps(items))
        np.save(out_dir / f"{split}_waves.npy", waves)
        np.save(out_dir / f"{split}_logmel.npy", feats)
        np.save(out_dir / f"{split}_labels.npy", label_ids)

        counts = Counter(it["label"] for it in items)
        summary["splits"][split] = {
            "num_examples": len(items),
            "feature_shape": list(feats.shape[1:]),
            "class_counts": {name: counts.get(name, 0) for name in labels},
        }
        print(f"{split}: {len(items)} examples, features {feats.shape}")

    train_feats = np.load(out_dir / "training_logmel.npy")
    mean, std = compute_norm_stats(train_feats)
    np.savez(out_dir / "norm_stats.npz", mean=mean, std=std)

    summary["norm_stats"] = {
        "computed_on": "training",
        "mean_per_bin": mean.tolist(),
        "std_per_bin": std.tolist(),
    }
    summary["feature_config"] = cfg["features"]
    summary["mel_scale"] = "HTK, triangular filters, peak 1.0, no area normalization"
    summary["precompute_seconds"] = round(time.perf_counter() - start_time, 1)
    results_path = ROOT / cfg["paths"]["results_dir"] / "data_summary.json"
    results_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {results_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
