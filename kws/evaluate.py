"""Evaluate the FP32 checkpoint on the test set.

Writes results/fp32_pytorch.json (accuracy, per-class accuracy, confusion matrix) and
results/model_summary.json (parameter count and multiply-accumulates per layer).
"""

import argparse
import json

import numpy as np
import torch

from kws.data import ROOT, load_config
from kws.model import DSCNN, count_macs, count_params
from kws.train import load_eval_split, predict


def classification_report(preds: np.ndarray, targets: np.ndarray, labels: list[str]) -> dict:
    n = len(labels)
    confusion = np.zeros((n, n), dtype=np.int64)
    np.add.at(confusion, (targets, preds), 1)
    per_class = {
        name: float(confusion[i, i] / confusion[i].sum()) if confusion[i].sum() else None
        for i, name in enumerate(labels)
    }
    return {
        "num_examples": int(len(targets)),
        "accuracy": float((preds == targets).mean()),
        "per_class_accuracy": per_class,
        "confusion_matrix": {
            "rows": "true label",
            "cols": "predicted label",
            "labels": labels,
            "matrix": confusion.tolist(),
        },
    }


def load_checkpoint(path) -> tuple[DSCNN, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = DSCNN.from_config(ckpt["config"], len(ckpt["labels"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", help="config variant, e.g. streaming_model")
    cfg = load_config(variant=parser.parse_args().variant)
    results_dir = ROOT / cfg["paths"]["results_dir"]
    model, ckpt = load_checkpoint(ROOT / cfg["train"]["checkpoint"])
    labels = ckpt["labels"]

    stats = np.load(ROOT / cfg["paths"]["features_dir"] / "norm_stats.npz")
    test_x, test_y = load_eval_split(cfg, "testing", stats["mean"], stats["std"])
    preds = predict(model, test_x, torch.device("cpu")).argmax(1).numpy()

    report = {
        "model": "DS-CNN FP32 (PyTorch eager, CPU)",
        "checkpoint_epoch": ckpt["epoch"],
        "checkpoint_val_acc": ckpt["val_acc"],
    }
    report.update(classification_report(preds, test_y.numpy(), labels))
    (results_dir / "fp32_pytorch.json").write_text(json.dumps(report, indent=2) + "\n")

    macs = count_macs(model)
    summary = {
        "num_params": count_params(model),
        "param_budget": cfg["model"]["max_params"],
        "macs_total": sum(macs.values()),
        "macs_per_layer": macs,
        "params_per_layer": {name: p.numel() for name, p in model.named_parameters()},
    }
    (results_dir / "model_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Test accuracy {report['accuracy']:.4f} on {report['num_examples']} clips")


if __name__ == "__main__":
    main()
