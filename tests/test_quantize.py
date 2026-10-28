import pytest
import torch

from kws.model import DSCNN
from kws.quantize import count_ops, folded_weight_sqnr, quantize_model, sqnr_db


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(0)
    model = DSCNN().eval()
    calib = torch.randn(32, 1, 98, 40)
    return model, quantize_model(model, calib, per_channel=True)


def test_quantized_graph_folds_batchnorm_and_quantizes_weights(models):
    _, quantized = models
    ops = count_ops(quantized)
    assert not any("batch_norm" in op for op in ops)
    # 9 conv weights + 1 linear weight, each quantized per output channel.
    assert ops["dequantize_per_channel"] == 10


def test_quantized_output_tracks_fp32(models):
    model, quantized = models
    x = torch.randn(16, 1, 98, 40)
    with torch.no_grad():
        ref, out = model(x), quantized(x)
    assert out.shape == (16, 12)
    assert sqnr_db(ref, out) > 15


def test_quantized_program_survives_save_and_load(models, tmp_path):
    _, quantized = models
    path = tmp_path / "q.pt2"
    torch.export.save(torch.export.export(quantized, (torch.zeros(1, 1, 98, 40),)), path)
    loaded = torch.export.load(path).module()
    x = torch.randn(1, 1, 98, 40)
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), quantized(x))


def test_sqnr_db():
    ref = torch.ones(100)
    assert sqnr_db(ref, ref + 0.1) == pytest.approx(20.0, abs=1e-4)


def test_per_channel_weight_error_is_never_worse_than_per_tensor():
    torch.manual_seed(0)
    model = DSCNN().eval()
    for bn in [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]:
        bn.weight.data.uniform_(0.1, 3.0)
    for layer in folded_weight_sqnr(model).values():
        assert layer["per_channel"] >= layer["per_tensor"]
