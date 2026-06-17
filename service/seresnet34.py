"""
SEResNet34 模型服务。

封装从头训练的 SE-ResNet34 架构，加载 weights/SEResNet34.pth 权重，
提供标准化推理接口。

Usage:
    from service import SEResNet34Service

    svc = SEResNet34Service()
    result = svc.predict("data/test/cat.1.jpg")
    print(result["label"], result["confidence"])
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union

import torchvision.transforms as T

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from models.seresnet import build_scratch_model

from service.base import BaseModelService

logger = logging.getLogger(__name__)


class SEResNet34Service(BaseModelService):
    """
    SE-ResNet34 猫狗分类服务。

    架构：SEResNet(layers=(3,4,6,3), width=64, num_classes=2)
    权重：weights/SEResNet34.pth
    """

    def __init__(
        self,
        weights_path: Union[str, Path] = "weights/SEResNet34_fp16.pth",
        device: Optional[str] = None,
    ) -> None:
        # 权重路径支持相对于项目根目录
        if not Path(weights_path).is_absolute():
            weights_path = PROJECT_ROOT / weights_path
        super().__init__(weights_path=weights_path, device=device)

    # ==================== 子类实现 ==================== #

    def _build_model(self):
        return build_scratch_model(
            num_classes=2,
            layers=(3, 4, 6, 3),   # SE-ResNet34
            width=64,
            drop_rate=0.2,
            drop_path=0.05,
        )

    def _get_transform(self) -> T.Compose:
        return T.Compose([
            T.Resize(int(IMAGE_SIZE * 1.14)),  # 256
            T.CenterCrop(IMAGE_SIZE),           # 224
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
