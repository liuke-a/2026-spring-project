"""
模型服务抽象基类。

封装所有猫狗分类模型共用的推理逻辑：权重加载、图像预处理、
单张/批量推理、后处理。子类只需实现 _build_model() 和 _get_transform()。
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

logger = logging.getLogger(__name__)

# ==================== 标签常量 ====================
CAT_LABEL: int = 0
DOG_LABEL: int = 1
LABEL_MAP: Dict[int, str] = {0: "cat", 1: "dog"}


# ==================== 抽象基类 ====================
class BaseModelService(ABC):
    """
    猫狗分类模型服务抽象基类。

    子类必须实现：
      - _build_model() -> nn.Module      # 构建模型架构
      - _get_transform() -> T.Compose    # 返回预处理管线

    可选覆盖：
      - _load_state_dict(checkpoint)     # 自定义权重提取逻辑

    Usage:
        class MyService(BaseModelService):
            def _build_model(self):
                return MyModel(num_classes=2)

            def _get_transform(self):
                return T.Compose([...])
    """

    def __init__(
        self,
        weights_path: Union[str, Path],
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            weights_path: .pth 权重文件路径。
            device: 'cuda'、'cpu' 或 None（自动检测）。
        """
        self.weights_path = Path(weights_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.transform: Optional[T.Compose] = None
        self._load()

    # ==================== 子类必须实现的抽象方法 ====================

    @abstractmethod
    def _build_model(self) -> nn.Module:
        """构建并返回未加载权重的模型架构。"""

    @abstractmethod
    def _get_transform(self) -> T.Compose:
        """返回与训练时验证集一致的图像预处理管线。"""

    # ==================== 权重提取（子类可选覆盖） ====================

    def _load_state_dict(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """
        从 checkpoint dict 中提取 state_dict。

        优先级：
          1. ema_state_dict（EMA 权重通常精度更高、泛化能力更强）
          2. model_state_dict（原始模型权重）
          3. 裸 state_dict（checkpoint 本身即为 state_dict）

        同时剥离 torch.compile 产生的 _orig_mod. 前缀。
        """
        if "ema_state_dict" in checkpoint:
            state_dict = checkpoint["ema_state_dict"]
            logger.info("使用 EMA 权重。")
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            logger.info("使用原始模型权重。")
        else:
            state_dict = checkpoint
            logger.info("Checkpoint 为裸 state_dict。")

        # 剥离 _orig_mod. 前缀（torch.compile 残留）
        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k.removeprefix("_orig_mod.")] = v
        if any(k.startswith("_orig_mod.") for k in state_dict):
            n = sum(1 for k in state_dict if k.startswith("_orig_mod."))
            logger.info(f"已剥离 {n} 个 _orig_mod. 前缀键。")

        return cleaned

    # ==================== 加载流程 ====================

    def _load(self) -> None:
        """构建模型 → 加载权重 → 移至设备 → 设为 eval 模式。"""
        if not self.weights_path.exists():
            raise FileNotFoundError(f"权重文件不存在: {self.weights_path}")

        self.model = self._build_model()
        self.transform = self._get_transform()

        checkpoint = torch.load(self.weights_path, map_location="cpu")
        state_dict = self._load_state_dict(checkpoint)

        # 检测半精度权重
        first_param = next(iter(state_dict.values()))
        self.is_fp16 = first_param.dtype == torch.float16
        if self.is_fp16:
            self.model = self.model.half()
            logger.info("检测到 fp16 权重，模型切换为半精度模式。")

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"缺失键 ({len(missing)}): {missing}")
        if unexpected:
            logger.warning(f"多余键 ({len(unexpected)}): {unexpected}")

        self.model.to(self.device)
        self.model.eval()
        logger.info(f"模型已加载至 {self.device}: {self.weights_path}")

    # ==================== 图像预处理 ====================

    def preprocess(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
    ) -> torch.Tensor:
        """
        将图像转换为批量推理张量。

        Args:
            image: 文件路径、PIL Image 或 numpy 数组 (HxWxC, uint8)。

        Returns:
            shape (1, 3, H, W) 的张量。
        """
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            raise TypeError(f"不支持的图像类型: {type(image)}")
        return self.transform(img).unsqueeze(0)

    # ==================== 推理接口 ====================

    @torch.inference_mode()
    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
    ) -> Dict[str, Any]:
        """
        单张图像推理。

        Returns:
            {
                "label": "cat" | "dog",
                "class_id": 0 | 1,
                "confidence": float (0..1),
                "probabilities": {"cat": float, "dog": float},
            }
        """
        tensor = self.preprocess(image).to(self.device)
        if self.is_fp16:
            tensor = tensor.half()
        outputs = self.model(tensor)
        return self._postprocess(outputs)

    @torch.inference_mode()
    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        max_batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        批量图像推理，自动分片防止显存溢出。

        Args:
            images: 图像列表，每项类型同 predict()。
            max_batch_size: 单次送入模型的最大图像数。
                None 则依次尝试：实例属性 self.max_batch_size → 128。

        Returns:
            结果 dict 列表，与输入顺序一致。
        """
        if max_batch_size is None:
            max_batch_size = getattr(self, "max_batch_size", 128)

        if not images:
            return []

        results: List[Dict[str, Any]] = []
        for start in range(0, len(images), max_batch_size):
            chunk = images[start : start + max_batch_size]
            tensors = torch.cat([self.preprocess(img) for img in chunk], dim=0)
            tensors = tensors.to(self.device)
            if self.is_fp16:
                tensors = tensors.half()
            outputs = self.model(tensors)
            probs = torch.softmax(outputs.float(), dim=1)
            confs, preds = probs.max(dim=1)

            for i in range(len(chunk)):
                cls_id = int(preds[i].item())
                label = LABEL_MAP[cls_id]
                results.append({
                    "label": label,
                    "class_id": cls_id,
                    "confidence": float(confs[i].item()),
                    "probabilities": {
                        "cat": float(probs[i, 0].item()),
                        "dog": float(probs[i, 1].item()),
                    },
                })
            # 释放中间张量，减少显存占用
            del outputs, probs
        return results

    # ==================== 后处理 ====================

    def _postprocess(self, outputs: torch.Tensor) -> Dict[str, Any]:
        """将原始 logits (1, num_classes) 转换为结果 dict。"""
        probs = torch.softmax(outputs.float(), dim=1)
        conf, pred = probs.max(dim=1)
        cls_id = int(pred.item())
        return {
            "label": LABEL_MAP[cls_id],
            "class_id": cls_id,
            "confidence": float(conf.item()),
            "probabilities": {
                "cat": float(probs[0, 0].item()),
                "dog": float(probs[0, 1].item()),
            },
        }
