"""Export FP32 and INT8 models to ExecuTorch .pte (XNNPACK), then verify them with the
ExecuTorch Python runtime.

Writes:
    models/kws_fp32.pte, models/kws_int8.pte
    results/delegation.txt     which ops were lowered to XNNPACK, per model
    results/pte_eval.json      .pte test accuracy, file sizes, delegation counts
    results/parity_python.json PyTorch vs .pte: max abs logit diff, top-1 agreement
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchao.quantization.pt2e  # noqa: F401  (registers quantized_decomposed ops for load)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.devtools.backend_debug import get_delegation_info
from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime
from tabulate import tabulate

from kws.data import ROOT, load_config
from kws.evaluate import load_checkpoint
from kws.train import load_eval_split

EXAMPLE_INPUT = (torch.zeros(1, 1, 98, 40),)


def lower_to_pte(program: torch.export.ExportedProgram, path: Path) -> dict:
    """Lower to XNNPACK, write the .pte, and return the delegation breakdown."""
    edge = to_edge_transform_and_lower(program, partitioner=[XnnpackPartitioner()])
    info = get_delegation_info(edge.exported_program().graph_module)
    path.write_bytes(edge.to_executorch().buffer)
    return {
        "num_delegated_subgraphs": info.num_delegated_subgraphs,
        "num_delegated_nodes": info.num_delegated_nodes,
        "num_non_delegated_nodes": info.num_non_delegated_nodes,
        "table": tabulate(
            info.get_operator_delegation_dataframe(), headers="keys", showindex=False
        ),
        "non_delegated_ops": sorted(
            op for op, b in info.delegation_by_operator.items() if b.non_delegated > 0
        ),
    }


def run_pte(path: Path, inputs: torch.Tensor) -> torch.Tensor:
    """Run a batch-1 .pte over every input, one clip at a time."""
    method = Runtime.get().load_program(path).load_method("forward")
    return torch.cat([method.execute([inputs[i : i + 1]])[0] for i in range(len(inputs))])


def run_module(module, inputs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat([module(inputs[i : i + 1]) for i in range(len(inputs))])


def parity(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    return {
        "max_abs_logit_diff": float((reference - candidate).abs().max()),
        "mean_abs_logit_diff": float((reference - candidate).abs().mean()),
        "top1_agreement": float((reference.argmax(1) == candidate.argmax(1)).float().mean()),
    }


def output_scale(program: torch.export.ExportedProgram) -> float:
    """Scale of the last activation quantize op, i.e. one INT8 step of the logits."""
    scales = [
        float(n.args[1])
        for n in program.graph.nodes
        if n.op == "call_function"
        and str(n.target).startswith("quantized_decomposed.quantize_per_tensor")
    ]
    return scales[-1]


def top2_margin(logits: torch.Tensor) -> torch.Tensor:
    top2 = logits.topk(2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", help="config variant, e.g. streaming_model")
    cfg = load_config(variant=parser.parse_args().variant)
    results_dir = ROOT / cfg["paths"]["results_dir"]
    fp32_pte = ROOT / cfg["export"]["fp32_pte"]
    int8_pte = ROOT / cfg["export"]["int8_pte"]
    checkpoint = ROOT / cfg["train"]["checkpoint"]
    int8_program_path = ROOT / cfg["quantization"]["quantized_program"]

    model, _ = load_checkpoint(checkpoint)
    int8_program = torch.export.load(int8_program_path)
    delegation = {
        "fp32": lower_to_pte(torch.export.export(model, EXAMPLE_INPUT), fp32_pte),
        "int8": lower_to_pte(int8_program, int8_pte),
    }

    lines = []
    for name, d in delegation.items():
        lines += [
            f"== {name.upper()} ({Path(cfg['export'][f'{name}_pte']).name}) ==",
            f"Delegated subgraphs: {d['num_delegated_subgraphs']}",
            f"Delegated nodes: {d['num_delegated_nodes']}",
            f"Non-delegated nodes: {d['num_non_delegated_nodes']}",
            f"Non-delegated op types: {', '.join(d['non_delegated_ops']) or 'none'}",
            "",
            d.pop("table"),
            "",
        ]
    (results_dir / "delegation.txt").write_text("\n".join(lines))

    stats = np.load(ROOT / cfg["paths"]["features_dir"] / "norm_stats.npz")
    test_x, test_y = load_eval_split(cfg, "testing", stats["mean"], stats["std"])

    logits = {
        "fp32_pytorch": run_module(model, test_x),
        "fp32_pte": run_pte(fp32_pte, test_x),
        "int8_pytorch": run_module(int8_program.module(), test_x),
        "int8_pte": run_pte(int8_pte, test_x),
    }
    accuracy = {k: float((v.argmax(1) == test_y).float().mean()) for k, v in logits.items()}

    parity_report = {
        "num_test_clips": len(test_y),
        "fp32_pytorch_vs_fp32_pte": parity(logits["fp32_pytorch"], logits["fp32_pte"]),
        "int8_pytorch_vs_int8_pte": parity(logits["int8_pytorch"], logits["int8_pte"]),
    }
    # Express the INT8 differences in output quantization steps, and check whether the
    # clips where top-1 differs were near-ties in the reference.
    step = output_scale(int8_program)
    int8_parity = parity_report["int8_pytorch_vs_int8_pte"]
    int8_parity["logit_quant_step"] = step
    int8_parity["max_abs_logit_diff_in_steps"] = int8_parity["max_abs_logit_diff"] / step
    int8_parity["mean_abs_logit_diff_in_steps"] = int8_parity["mean_abs_logit_diff"] / step
    margin = top2_margin(logits["int8_pytorch"])
    disagree = logits["int8_pytorch"].argmax(1) != logits["int8_pte"].argmax(1)
    int8_parity["num_top1_disagreements"] = int(disagree.sum())
    int8_parity["median_top2_margin_all_clips"] = float(margin.median())
    int8_parity["max_top2_margin_disagreeing_clips"] = (
        float(margin[disagree].max()) if disagree.any() else None
    )
    (results_dir / "parity_python.json").write_text(json.dumps(parity_report, indent=2) + "\n")

    eval_report = {
        "runtime": "ExecuTorch Python runtime, XNNPACK delegate, batch 1",
        "num_test_clips": len(test_y),
        "test_accuracy": accuracy,
        "file_size_bytes": {
            "fp32_checkpoint_pt": checkpoint.stat().st_size,
            "fp32_pte": fp32_pte.stat().st_size,
            "int8_pte": int8_pte.stat().st_size,
        },
        "delegation": delegation,
    }
    (results_dir / "pte_eval.json").write_text(json.dumps(eval_report, indent=2) + "\n")
    print(json.dumps({"accuracy": accuracy, **parity_report}, indent=2))
    print(json.dumps(eval_report["file_size_bytes"], indent=2))


if __name__ == "__main__":
    main()
