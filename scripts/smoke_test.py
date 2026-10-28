"""Stage 0 smoke test: export a tiny 2-layer model to .pte with XNNPACK and run it
with the ExecuTorch Python runtime. Compares the output against eager PyTorch and
writes results/smoke_test.json."""

import json
from pathlib import Path

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime

ROOT = Path(__file__).resolve().parent.parent
PTE_PATH = ROOT / "models" / "smoke_toy.pte"
OUT_PATH = ROOT / "results" / "smoke_test.json"


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(16, 32)
        self.fc2 = torch.nn.Linear(32, 4)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def main():
    torch.manual_seed(0)
    model = ToyModel().eval()
    example = (torch.randn(1, 16),)

    # 1. Capture the graph, 2. lower supported ops to XNNPACK, 3. serialize to .pte
    exported = torch.export.export(model, example)
    edge = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
    nodes = [n for n in edge.exported_program().graph_module.graph.nodes if n.op == "call_function"]
    num_delegate_calls = sum("executorch_call_delegate" in str(n.target) for n in nodes)
    non_delegated_ops = sorted(
        {
            str(n.target)
            for n in nodes
            if "executorch_call_delegate" not in str(n.target) and "getitem" not in str(n.target)
        }
    )
    et_program = edge.to_executorch()
    PTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PTE_PATH.write_bytes(et_program.buffer)

    # Run the .pte with the ExecuTorch Python runtime
    runtime = Runtime.get()
    program = runtime.load_program(PTE_PATH)
    method = program.load_method("forward")
    et_out = method.execute([example[0]])[0]

    with torch.no_grad():
        ref_out = model(example[0])

    max_abs_diff = float((et_out - ref_out).abs().max())
    result = {
        "pte_path": str(PTE_PATH.relative_to(ROOT)),
        "pte_size_bytes": PTE_PATH.stat().st_size,
        "num_xnnpack_delegate_calls": num_delegate_calls,
        "non_delegated_ops": non_delegated_ops,
        "output_shape": list(et_out.shape),
        "max_abs_diff_vs_pytorch": max_abs_diff,
        "passed": bool(torch.allclose(et_out, ref_out, atol=1e-5)),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("Smoke test FAILED: ExecuTorch output does not match PyTorch")


if __name__ == "__main__":
    main()
