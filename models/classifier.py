"""
模型定义模块。

使用 ImageNet 预训练 ResNet-18 作为 Backbone，替换原 FC 层为 2 分类头。
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


class CatDogClassifier(nn.Module):
    """
    猫狗二分类模型。

    基于 torchvision ResNet-18，保留卷积特征提取层，替换全连接层为 2 分类输出。

    Args:
        num_classes: 分类数（默认 2：猫/狗）
        pretrained: 是否加载 ImageNet 预训练权重
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True) -> None:
        super().__init__()

        # 加载预训练 Backbone
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # 获取原 FC 层输入维度（512 for ResNet18）
        in_features = self.backbone.fc.in_features

        # 替换原 FC 为 Identity（仅特征提取），后续接自定义 Head
        self.backbone.fc = nn.Identity()

        # 新分类头
        self.head = nn.Linear(in_features, num_classes)

        logger.info(
            f"Model initialized: ResNet18 backbone (pretrained={pretrained}), "
            f"head: {in_features} -> {num_classes}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


if __name__ == "__main__":
    # 快速测试：验证模型输出维度
    model = CatDogClassifier(num_classes=2, pretrained=True)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Input: {dummy.shape} -> Output: {out.shape}")
    assert out.shape == (2, 2), "Output shape mismatch!"
    print("Model test passed.")