"""
SimpleCNN：用于猫狗二分类的简单 VGG 风格卷积神经网络。

设计要点：
- 5 个卷积块，每块包含两次 3×3 卷积 + BN + ReLU，最后 2×2 最大池化
- 通道数逐步翻倍：3 → 32 → 64 → 128 → 256 → 512
- 分类头：AdaptiveAvgPool2d → Dropout → FC(512→256) → ReLU → Dropout → FC(256→2)
- 权重初始化：Kaiming Normal (Conv) + 常量 (BN) + Normal (Linear)

与预训练流程、SE-ResNet 流程互不影响，由 models/CNN/train.py / evaluate.py 使用。
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    """
    简单的 VGG 风格卷积神经网络，用于猫狗二分类。

    Args:
        num_classes: 分类数（默认 2）
        dropout: 分类头 Dropout 概率（默认 0.5）
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5) -> None:
        super().__init__()

        # Block 1: 3 -> 32, 224 -> 112
        self.block1 = self._make_block(3, 32)
        # Block 2: 32 -> 64, 112 -> 56
        self.block2 = self._make_block(32, 64)
        # Block 3: 64 -> 128, 56 -> 28
        self.block3 = self._make_block(64, 128)
        # Block 4: 128 -> 256, 28 -> 14
        self.block4 = self._make_block(128, 256)
        # Block 5: 256 -> 512, 14 -> 7
        self.block5 = self._make_block(256, 512)

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"SimpleCNN initialized: num_classes={num_classes}, "
            f"dropout={dropout}, params={total_params / 1e6:.2f}M"
        )

    @staticmethod
    def _make_block(in_channels: int, out_channels: int) -> nn.Sequential:
        """构建一个卷积块：Conv-BN-ReLU ×2 + MaxPool。"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def _init_weights(self) -> None:
        """Kaiming 初始化卷积层，常数初始化 BN，正态初始化线性层。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


def build_cnn_model(num_classes: int = 2, dropout: float = 0.5) -> SimpleCNN:
    """构建 SimpleCNN 模型的工厂函数。"""
    return SimpleCNN(num_classes=num_classes, dropout=dropout)


if __name__ == "__main__":
    # 快速自检：验证前向输出维度与参数量
    import sys
    from pathlib import Path

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(_PROJECT_ROOT))

    from utils.logger import setup_logger

    setup_logger("cnn_selfcheck")
    model = build_cnn_model(num_classes=2)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Input: {dummy.shape} -> Output: {out.shape}")
    print(f"Total params: {n_params / 1e6:.2f}M")
    assert out.shape == (2, 2), f"Output shape mismatch: expected (2,2), got {out.shape}"
    print("SimpleCNN test passed.")
