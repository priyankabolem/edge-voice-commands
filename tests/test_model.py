import numpy as np
import torch
from torch import nn

from kws.data import label_names, load_config
from kws.evaluate import classification_report
from kws.model import DSCNN, count_macs, count_params

CFG = load_config()


def test_output_shape_and_param_budget():
    model = DSCNN.from_config(CFG, len(label_names(CFG))).eval()
    assert model(torch.zeros(2, 1, 98, 40)).shape == (2, 12)
    assert count_params(model) <= CFG["model"]["max_params"]


def test_first_conv_halves_time_and_frequency():
    model = DSCNN.from_config(CFG, 12).eval()
    out = model.conv(torch.zeros(1, 1, 98, 40))
    assert out.shape == (1, 64, 49, 20)


def test_mac_count_matches_hand_calculation():
    model = DSCNN(channels=64, num_blocks=1).eval()
    macs = count_macs(model)
    positions = 49 * 20
    assert macs["conv"] == positions * 64 * 10 * 4
    assert macs["blocks.0.depthwise"] == positions * 64 * 3 * 3
    assert macs["blocks.0.pointwise"] == positions * 64 * 64
    assert macs["fc"] == 64 * 12


def test_depthwise_separable_is_cheaper_than_standard_conv():
    standard = nn.Conv2d(64, 64, 3, padding=1, bias=False)
    block = DSCNN(num_blocks=1).blocks[0]
    separable = block.depthwise.weight.numel() + block.pointwise.weight.numel()
    assert separable < standard.weight.numel() / 7


def test_model_is_exportable():
    model = DSCNN.from_config(CFG, 12).eval()
    torch.export.export(model, (torch.zeros(1, 1, 98, 40),))


def test_classification_report():
    preds = np.array([0, 1, 1, 2])
    targets = np.array([0, 1, 2, 2])
    report = classification_report(preds, targets, ["a", "b", "c"])
    assert report["accuracy"] == 0.75
    assert report["per_class_accuracy"] == {"a": 1.0, "b": 1.0, "c": 0.5}
    assert report["confusion_matrix"]["matrix"][2] == [0, 1, 1]
