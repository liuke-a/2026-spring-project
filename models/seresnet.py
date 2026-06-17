"""
从头训练的猫狗分类模型 SE-ResNet。

设计要点：
- SEBlock：Squeeze-and-Excitation 通道注意力，增强特征表达。
- SEBasicBlock：ResNet BasicBlock + SE，并支持随机深度 (DropPath)。
- SEResNet：标准 ResNet 骨干（7x7 stem + maxpool + 4 个 stage），默认深度
  [2,2,2,2]（18 层），可切换到 [3,4,6,3]（34 层）。
- 初始化：Kaiming 卷积初始化 + 残差分支末尾 BN gamma 置 0（zero-init
  residual），提升从随机初始化训练的稳定性。

与预训练流程互不影响，由 train_scratch.py / evaluate_scratch.py / predict_scratch.py 使用。
"""

import logging
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """随机深度：训练时按样本随机丢弃残差分支（保持期望不变）。"""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    # 每个样本一个 mask，形状 (N, 1, 1, 1) 以便广播
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()
    return x.div(keep_prob) * mask


class DropPath(nn.Module):
    """随机深度的模块封装。"""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation 通道注意力。

    全局平均池化 -> 1x1 降维 + ReLU -> 1x1 升维 + Sigmoid -> 逐通道缩放。
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3), keepdim=True)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


def conv3x3(in_c: int, out_c: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_c: int, out_c: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False)


class SEBasicBlock(nn.Module):
    """ResNet BasicBlock + SE + 可选 DropPath。"""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        reduction: int = 16,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction)
        self.downsample = downsample
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.drop_path(out) + identity
        return F.relu(out, inplace=True)


class SEResNet(nn.Module):
    """
    手写 SE-ResNet，用于从头训练猫狗二分类。

    Args:
        num_classes: 分类数（默认 2）
        layers: 四个 stage 的 block 数，默认 (2,2,2,2)=ResNet18，可用 (3,4,6,3)=ResNet34
        width: 起始通道数（默认 64）
        drop_rate: 分类头前的 Dropout 概率
        drop_path: 最大随机深度概率（沿网络深度线性递增）
        reduction: SE 模块降维比例
    """

    def __init__(
        self,
        num_classes: int = 2,
        layers: Sequence[int] = (2, 2, 2, 2),
        width: int = 64,
        drop_rate: float = 0.2,
        drop_path: float = 0.05,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        self.in_channels = width
        self.reduction = reduction

        # Stem: 7x7 s2 -> BN -> ReLU -> 3x3 maxpool s2
        self.conv1 = nn.Conv2d(3, width, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 沿深度线性递增的 DropPath 概率
        total_blocks = sum(layers)
        dpr = [drop_path * i / max(total_blocks - 1, 1) for i in range(total_blocks)]
        block_idx = 0

        stage_channels = [width, width * 2, width * 4, width * 8]
        stage_strides = [1, 2, 2, 2]
        stages = []
        for ch, n_blocks, stride in zip(stage_channels, layers, stage_strides):
            stages.append(self._make_stage(ch, n_blocks, stride, dpr[block_idx:block_idx + n_blocks]))
            block_idx += n_blocks
        self.layer1, self.layer2, self.layer3, self.layer4 = stages

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=drop_rate)
        self.fc = nn.Linear(stage_channels[-1] * SEBasicBlock.expansion, num_classes)

        self._init_weights()

        logger.info(
            f"SEResNet initialized: layers={tuple(layers)}, width={width}, "
            f"drop_rate={drop_rate}, drop_path={drop_path} -> {num_classes} classes"
        )

    def _make_stage(
        self,
        out_channels: int,
        n_blocks: int,
        stride: int,
        drop_path_rates: Sequence[float],
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels * SEBasicBlock.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels * SEBasicBlock.expansion, stride),
                nn.BatchNorm2d(out_channels * SEBasicBlock.expansion),
            )

        blocks = [
            SEBasicBlock(
                self.in_channels, out_channels, stride, downsample,
                self.reduction, drop_path_rates[0],
            )
        ]
        self.in_channels = out_channels * SEBasicBlock.expansion
        for i in range(1, n_blocks):
            blocks.append(
                SEBasicBlock(
                    self.in_channels, out_channels,
                    reduction=self.reduction, drop_path_rate=drop_path_rates[i],
                )
            )
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.constant_(m.bias, 0.0)

        # zero-init residual：将每个残差分支末尾 BN 的 gamma 置 0
        for m in self.modules():
            if isinstance(m, SEBasicBlock):
                nn.init.constant_(m.bn2.weight, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


def build_scratch_model(
    num_classes: int = 2,
    layers: Sequence[int] = (2, 2, 2, 2),
    width: int = 64,
    drop_rate: float = 0.2,
    drop_path: float = 0.05,
) -> SEResNet:
    """构建从头训练的 SE-ResNet 模型。"""
    return SEResNet(
        num_classes=num_classes,
        layers=layers,
        width=width,
        drop_rate=drop_rate,
        drop_path=drop_path,
    )


if __name__ == "__main__":
    # 快速自检：验证前向输出维度与参数量
    model = build_scratch_model(num_classes=2)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Input: {dummy.shape} -> Output: {out.shape}")
    print(f"Total params: {n_params / 1e6:.2f}M")
    assert out.shape == (2, 2), "Output shape mismatch!"
    print("SEResNet test passed.")
