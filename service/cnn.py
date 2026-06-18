"""
SimpleCNN 模型服务。

封装从头训练的 SimpleCNN 架构（VGG 风格），加载 checkpoints/cnn/best_model.pth，
提供与 SEResNet 系列统一的推理接口。

Usage:
    from service import CNNService

    svc = CNNService()
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
from models.CNN.CNN import SimpleCNN

from service.base import BaseModelService

logger = logging.getLogger(__name__)


class CNNService(BaseModelService):
    """
    SimpleCNN 猫狗分类服务。

    架构：SimpleCNN(num_classes=2) — 5 个 VGG 风格卷积块 + 分类头
    权重：checkpoints/cnn/best_model.pth
    """

    def __init__(
        self,
        weights_path: Union[str, Path] = "checkpoints/cnn/best_model.pth",
        device: Optional[str] = None,
    ) -> None:
        # 权重路径支持相对于项目根目录
        if not Path(weights_path).is_absolute():
            weights_path = PROJECT_ROOT / weights_path
        super().__init__(weights_path=weights_path, device=device)

    # ==================== 子类实现 ==================== #

    def _build_model(self):
        return SimpleCNN(num_classes=2, dropout=0.5)

    def _get_transform(self) -> T.Compose:
        return T.Compose([
            T.Resize(int(IMAGE_SIZE * 1.14)),  # 256
            T.CenterCrop(IMAGE_SIZE),           # 224
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


if __name__ == "__main__":
    svc = CNNService()

    # 对 data/images 目录下所有图片进行预测
    image_dir = PROJECT_ROOT / "data" / "images"
    image_paths = sorted(image_dir.glob("*"))
    image_paths = [p for p in image_paths if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")]

    if not image_paths:
        print(f"目录 {image_dir} 中没有找到图片文件。")
    else:
        print(f"共找到 {len(image_paths)} 张图片，开始预测...\n")
        results = svc.predict_batch([str(p) for p in image_paths])
        for path, result in zip(image_paths, results):
            print(f"{path.name}: {result['label']} (confidence: {result['confidence']:.4f})")
