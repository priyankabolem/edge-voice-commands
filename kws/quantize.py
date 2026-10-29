"""INT8 post-training quantization with the PT2E flow and the XNNPACK quantizer.

    torch.export -> prepare_pt2e (fold BN, insert observers) -> calibrate -> convert_pt2e

Default run: quantize with the config settings, evaluate on the test set, write
results/int8_pytorch.json and save the quantized program for ExecuTorch export.
With --experiments: sweep calibration size and per-channel vs per-tensor weights and
write results/quant_experiments.json.
"""

import argparse
import json
import time

import numpy as np
import torch
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from torchao.quantization.pt2e import allow_exported_model_train_eval
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from kws.data import ROOT, load_config
from kws.evaluate import classification_report, load_checkpoint
from kws.train import load_eval_split, predict

EXAMPLE_INPUT = (torch.zeros(1, 1, 98, 40),)


def calibration_set(cfg: dict, num_samples: int, seed: int) -> torch.Tensor:
    """Random clean training clips, normalized, as (num_samples, 1, 98, 40)."""
    stats = np.load(ROOT / cfg["paths"]["features_dir"] / "norm_stats.npz")
    train_x, _ = load_eval_split(cfg, "training", stats["mean"], stats["std"])
    gen = torch.Generator().manual_seed(seed)
    return train_x[torch.randperm(len(train_x), generator=gen)[:num_samples]]


def quantize_model(
    model: torch.nn.Module,
    calib_x: torch.Tensor,
    per_channel: bool = True,
    quantizer: XNNPACKQuantizer | None = None,
) -> torch.fx.GraphModule:
    if quantizer is None:
        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=per_channel)
        )
    # Dynamic batch so calibration and evaluation can run in batches; the device
    # program is re-exported later with a fixed batch of 1.
    batch = torch.export.Dim("batch", min=1, max=1024)
    captured = torch.export.export(
        model.eval(), (torch.zeros(2, 1, 98, 40),), dynamic_shapes=({0: batch},)
    ).module()
    prepared = prepare_pt2e(captured, quantizer)
    with torch.no_grad():
        for i in range(0, len(calib_x), 100):
            prepared(calib_x[i : i + 100])
    converted = convert_pt2e(prepared)
    # Lets callers such as predict() use .eval() on the exported graph module.
    allow_exported_model_train_eval(converted)
    return converted


def count_ops(graph_module: torch.fx.GraphModule) -> dict[str, int]:
    counts = {}
    for node in graph_module.graph.nodes:
        if node.op == "call_function":
            name = str(node.target).split(".")[-2] if "." in str(node.target) else str(node.target)
            counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def test_logits(model, cfg: dict) -> tuple[torch.Tensor, np.ndarray]:
    stats = np.load(ROOT / cfg["paths"]["features_dir"] / "norm_stats.npz")
    test_x, test_y = load_eval_split(cfg, "testing", stats["mean"], stats["std"])
    return predict(model, test_x, torch.device("cpu"), batch_size=500), test_y.numpy()


def sqnr_db(reference: torch.Tensor, approx: torch.Tensor) -> float:
    """Signal-to-quantization-noise ratio: 10 log10(|ref|^2 / |ref - approx|^2)."""
    noise = (reference - approx).pow(2).sum()
    return float(10 * torch.log10(reference.pow(2).sum() / noise))


def fidelity(fp32_logits: torch.Tensor, int8_logits: torch.Tensor) -> dict:
    """How closely INT8 tracks FP32; finer-grained than an accuracy difference."""
    return {
        "top1_agreement_with_fp32": float(
            (fp32_logits.argmax(1) == int8_logits.argmax(1)).float().mean()
        ),
        "logit_sqnr_db": round(sqnr_db(fp32_logits, int8_logits), 2),
        "max_abs_logit_diff": float((fp32_logits - int8_logits).abs().max()),
    }


def folded_weight_sqnr(model: torch.nn.Module) -> dict[str, dict[str, float]]:
    """Weight SQNR of each conv after BN folding, for per-tensor vs per-channel INT8.

    Uses symmetric quantization to [-127, 127], as the XNNPACK quantizer does.
    """
    out = {}
    modules = dict(model.named_modules())
    for name, conv in modules.items():
        if not isinstance(conv, torch.nn.Conv2d):
            continue
        bn = modules[_bn_name(name)]
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        w = (conv.weight * scale.reshape(-1, 1, 1, 1)).detach()
        result = {}
        for mode in ("per_tensor", "per_channel"):
            if mode == "per_tensor":
                step = w.abs().max() / 127
            else:
                step = w.abs().amax(dim=(1, 2, 3), keepdim=True) / 127
            result[mode] = round(sqnr_db(w, torch.round(w / step).clamp(-127, 127) * step), 2)
        out[name] = result
    return out


def _bn_name(conv_name: str) -> str:
    """Name of the BatchNorm that follows a conv in DSCNN."""
    if conv_name == "conv":
        return "bn"
    return conv_name.replace("depthwise", "bn1").replace("pointwise", "bn2")


def run_default(cfg: dict):
    qcfg = cfg["quantization"]
    model, ckpt = load_checkpoint(ROOT / cfg["train"]["checkpoint"])
    calib_x = calibration_set(cfg, qcfg["calibration_samples"], cfg["seed"])
    start = time.perf_counter()
    quantized = quantize_model(model, calib_x, qcfg["per_channel"])
    quantize_seconds = time.perf_counter() - start

    report = {
        "model": "DS-CNN INT8 (PT2E + XNNPACKQuantizer, PyTorch reference kernels, CPU)",
        "calibration_samples": qcfg["calibration_samples"],
        "per_channel_weights": qcfg["per_channel"],
        "quantize_seconds": round(quantize_seconds, 2),
        "graph_op_counts": count_ops(quantized),
    }
    fp32_logits, test_y = test_logits(model, cfg)
    int8_logits, _ = test_logits(quantized, cfg)
    report.update(classification_report(int8_logits.argmax(1).numpy(), test_y, ckpt["labels"]))
    report.update(fidelity(fp32_logits, int8_logits))
    fp32 = json.loads((ROOT / cfg["paths"]["results_dir"] / "fp32_pytorch.json").read_text())
    report["fp32_accuracy"] = fp32["accuracy"]
    report["accuracy_drop_points"] = round(100 * (fp32["accuracy"] - report["accuracy"]), 3)
    out = ROOT / cfg["paths"]["results_dir"] / "int8_pytorch.json"
    out.write_text(json.dumps(report, indent=2) + "\n")

    program_path = ROOT / qcfg["quantized_program"]
    # Loading this file requires importing torchao.quantization.pt2e first, which
    # registers the quantized_decomposed ops it references.
    torch.export.save(torch.export.export(quantized, EXAMPLE_INPUT), program_path)
    print(
        f"INT8 test accuracy {report['accuracy']:.4f} "
        f"(FP32 {fp32['accuracy']:.4f}, drop {report['accuracy_drop_points']} points)"
    )


def run_experiments(cfg: dict):
    qcfg = cfg["quantization"]
    model, ckpt = load_checkpoint(ROOT / cfg["train"]["checkpoint"])
    fp32 = json.loads((ROOT / cfg["paths"]["results_dir"] / "fp32_pytorch.json").read_text())
    fp32_logits, test_y = test_logits(model, cfg)
    runs = []
    for num_samples in qcfg["experiment_calibration_sizes"]:
        calib_x = calibration_set(cfg, num_samples, cfg["seed"])
        for per_channel in (True, False):
            quantized = quantize_model(model, calib_x, per_channel)
            int8_logits, _ = test_logits(quantized, cfg)
            acc = float((int8_logits.argmax(1).numpy() == test_y).mean())
            runs.append(
                {
                    "calibration_samples": num_samples,
                    "per_channel_weights": per_channel,
                    "test_accuracy": acc,
                    "drop_points": round(100 * (fp32["accuracy"] - acc), 3),
                    **fidelity(fp32_logits, int8_logits),
                }
            )
            print(runs[-1])
    out = ROOT / cfg["paths"]["results_dir"] / "quant_experiments.json"
    summary = {
        "fp32_accuracy": fp32["accuracy"],
        "num_test_clips": int(len(test_y)),
        "runs": runs,
        "folded_weight_sqnr_db": folded_weight_sqnr(model),
    }
    out.write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", action="store_true")
    parser.add_argument("--variant", help="config variant, e.g. streaming_model")
    args = parser.parse_args()
    cfg = load_config(variant=args.variant)
    if args.experiments:
        run_experiments(cfg)
    else:
        run_default(cfg)


if __name__ == "__main__":
    main()
