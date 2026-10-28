import pytest
import torch

from kws.export import lower_to_pte, output_scale, parity, run_module, run_pte, top2_margin
from kws.model import DSCNN
from kws.quantize import quantize_model

EXAMPLE = (torch.zeros(1, 1, 98, 40),)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return DSCNN().eval()


def test_fp32_pte_is_fully_delegated_and_matches_pytorch(model, tmp_path):
    path = tmp_path / "fp32.pte"
    info = lower_to_pte(torch.export.export(model, EXAMPLE), path)
    assert info["num_delegated_subgraphs"] == 1
    assert set(info["non_delegated_ops"]) <= {"getitem"}

    x = torch.randn(4, 1, 98, 40)
    result = parity(run_module(model, x), run_pte(path, x))
    assert result["max_abs_logit_diff"] < 1e-4
    assert result["top1_agreement"] == 1.0


def test_int8_pte_is_fully_delegated_and_close_to_reference(model, tmp_path):
    quantized = quantize_model(model, torch.randn(32, 1, 98, 40))
    program = torch.export.export(quantized, EXAMPLE)
    path = tmp_path / "int8.pte"
    info = lower_to_pte(program, path)
    assert info["num_delegated_subgraphs"] == 1
    assert set(info["non_delegated_ops"]) <= {"getitem"}

    x = torch.randn(4, 1, 98, 40)
    diff = parity(run_module(program.module(), x), run_pte(path, x))["max_abs_logit_diff"]
    # Integer kernels may differ from the float reference by a few output steps.
    assert diff <= 10 * output_scale(program)


def test_top2_margin():
    logits = torch.tensor([[3.0, 1.0, 0.5], [0.2, 0.25, -1.0]])
    torch.testing.assert_close(top2_margin(logits), torch.tensor([2.0, 0.05]))
