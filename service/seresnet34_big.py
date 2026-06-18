"""
SEResNet34-Big 模型服务。

封装从头训练的 SEResNet34 加宽变体（width=96），
加载 weights/SEResNet34_big_fp16.pth 权重，提供标准化推理接口。

架构：SEResNet(layers=(3,4,6,3), width=96, num_classes=2)
参数量：~48M

Usage:
    from service import SEResNet34BigService

    svc = SEResNet34BigService()
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


class SEResNet34BigService(BaseModelService):
    """
    SE-ResNet34-Big 猫狗分类服务。

    架构：SEResNet(layers=(3,4,6,3), width=96, num_classes=2)
    权重：weights/SEResNet34_big_fp16.pth（FP16 EMA 权重，~92MB）
    """

    def __init__(
        self,
        weights_path: Union[str, Path] = "weights/SEResNet34_big_fp16.pth",
        device: Optional[str] = None,
    ) -> None:
        if not Path(weights_path).is_absolute():
            weights_path = PROJECT_ROOT / weights_path
        super().__init__(weights_path=weights_path, device=device)

    # ==================== 子类实现 ==================== #

    def _build_model(self):
        return build_scratch_model(
            num_classes=2,
            layers=(3, 4, 6, 3),   # SE-ResNet34 深度
            width=96,               # 加宽变体（默认 64 → 96）
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
