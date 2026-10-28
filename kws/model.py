"""DS-CNN keyword-spotting model (small variant of Zhang et al., "Hello Edge", 2017)."""

import torch
from torch import nn


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.depthwise(x)))
        return self.relu(self.bn2(self.pointwise(x)))


class DSCNN(nn.Module):
    """Input (batch, 1, 98, 40) normalized log-mel -> (batch, num_classes) logits."""

    def __init__(
        self,
        num_classes: int = 12,
        channels: int = 64,
        num_blocks: int = 4,
        first_kernel: tuple[int, int] = (10, 4),
        first_stride: tuple[int, int] = (2, 2),
        first_padding: tuple[int, int] = (4, 1),
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            1, channels, tuple(first_kernel), tuple(first_stride), tuple(first_padding), bias=False
        )
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.blocks = nn.Sequential(*[DepthwiseSeparableBlock(channels) for _ in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, num_classes)

    @classmethod
    def from_config(cls, cfg: dict, num_classes: int) -> "DSCNN":
        m = cfg["model"]
        return cls(
            num_classes,
            m["channels"],
            m["num_blocks"],
            m["first_kernel"],
            m["first_stride"],
            m["first_padding"],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn(self.conv(x)))
        x = self.blocks(x)
        return self.fc(torch.flatten(self.pool(x), 1))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_macs(model: nn.Module, input_shape=(1, 1, 98, 40)) -> dict[str, int]:
    """Multiply-accumulates per inference for every Conv2d and Linear layer."""
    macs = {}

    def hook(name):
        def fn(module, inputs, output):
            if isinstance(module, nn.Conv2d):
                kh, kw = module.kernel_size
                per_output = (module.in_channels // module.groups) * kh * kw
                macs[name] = output.numel() * per_output
            else:
                macs[name] = module.in_features * module.out_features

        return fn

    handles = [
        m.register_forward_hook(hook(name))
        for name, m in model.named_modules()
        if isinstance(m, (nn.Conv2d, nn.Linear))
    ]
    with torch.no_grad():
        model.eval()(torch.zeros(input_shape))
    for h in handles:
        h.remove()
    return macs
