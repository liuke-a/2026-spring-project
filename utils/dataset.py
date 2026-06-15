"""
数据集加载与预处理模块。

基于 EDA 结论设计：
- 25,000 张图片全部有效，无损坏，格式统一为 JPEG
- 类别均衡（cat:dog = 1:1），无需重采样
- 原始尺寸均值 404x360，统一 resize 至 224x224 属于缩小操作，信息损失可控
- 标签通过文件名前缀（cat.* / dog.*）自动解析
"""

import logging
import random
from pathlib import Path
from typing import Tuple, Optional, Callable, List

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from config import (
    TRAIN_DIR, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    BATCH_SIZE, NUM_WORKERS, VAL_RATIO, RANDOM_SEED, MAX_SAMPLES
)

logger = logging.getLogger(__name__)


class CatDogDataset(Dataset):
    """
    猫狗分类数据集。

    从目录加载图片，根据文件名前缀自动分配标签：
        - 'cat.*' -> 0
        - 'dog.*' -> 1

    Args:
        root_dir: 图片目录路径
        transform: torchvision 变换操作（由外部传入）
        samples: 可选，直接传入样本列表（用于划分后的子集）
    """

    def __init__(
        self,
        root_dir: Path,
        transform: Optional[Callable] = None,
        samples: Optional[List[Tuple[Path, int]]] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform

        if samples is None:
            self.samples: List[Tuple[Path, int]] = []
            self._scan_directory(max_samples)
        else:
            self.samples = samples[:max_samples] if max_samples is not None else samples
            logger.info(f"Dataset loaded: {len(self.samples)} samples (from split list)")

    def _scan_directory(self, max_samples: Optional[int] = None) -> None:
        """扫描目录，解析标签，跳过非图片文件。"""
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.root_dir}")

        valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
        cat_count = dog_count = 0

        # 全量加载时直接遍历，无额外开销；限制样本数时先打乱，保证类别均衡
        if max_samples is None:
            file_iter = self.root_dir.iterdir()
        else:
            file_iter = list(self.root_dir.iterdir())
            random.shuffle(file_iter)

        for file_path in file_iter:
            if max_samples is not None and len(self.samples) >= max_samples:
                break

            if file_path.is_dir() or file_path.suffix.lower() not in valid_ext:
                continue

            name_lower = file_path.name.lower()
            if name_lower.startswith("cat"):
                label = 0
                cat_count += 1
            elif name_lower.startswith("dog"):
                label = 1
                dog_count += 1
            else:
                logger.warning(f"Unknown prefix, skipped: {file_path.name}")
                continue

            # EDA 已确认无损坏，但保留异常处理以符合工业规范
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except Exception as e:
                logger.error(f"Corrupted image skipped: {file_path.name} | {e}")
                continue

            self.samples.append((file_path, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid images found in {self.root_dir}")

        logger.info(f"Class distribution: cat={cat_count}, dog={dog_count}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        file_path, label = self.samples[idx]

        try:
            image = Image.open(file_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")
            # 兜底：返回空白图，避免单个样本导致整个 batch 崩溃
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=0)

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms() -> Tuple[Callable, Callable]:
    """
    构建训练集与验证集的图像变换。

    训练集：RandomResizedCrop + RandomHorizontalFlip + ColorJitter
    验证集：Resize + CenterCrop（保持比例，不引入随机性）

    均使用 ImageNet 标准化参数，以适配后续预训练模型。
    """
    train_transform = T.Compose([
        T.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_transform = T.Compose([
        T.Resize(int(IMAGE_SIZE * 1.14)),  # 256 for 224
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, val_transform


def get_dataloaders(max_samples: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
    """
    构建训练集与验证集的 DataLoader。

    从 TRAIN_DIR 加载全部数据，按 VAL_RATIO 随机划分为训练子集与验证子集。
    两个子集分别创建独立的 Dataset 实例，绑定不同的 transform。

    Args:
        max_samples: 限制加载样本总数，None 表示加载全部。用于快速验证。

    Returns:
        (train_loader, val_loader)
    """
    # 先扫描完整数据集
    full_dataset = CatDogDataset(root_dir=TRAIN_DIR, transform=None, max_samples=max_samples)
    total = len(full_dataset)

    # 计算划分长度
    val_size = int(total * VAL_RATIO)
    train_size = total - val_size

    logger.info(f"Splitting: train={train_size}, val={val_size}, total={total}")

    # 固定随机种子，保证划分结果可复现
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    indices = torch.randperm(total, generator=generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # 提取样本列表，创建独立的 Dataset 实例（避免 transform 互相覆盖）
    train_samples = [full_dataset.samples[i] for i in train_indices]
    val_samples = [full_dataset.samples[i] for i in val_indices]

    train_transform, val_transform = get_transforms()

    train_dataset = CatDogDataset(
        root_dir=TRAIN_DIR, transform=train_transform, samples=train_samples
    )
    val_dataset = CatDogDataset(
        root_dir=TRAIN_DIR, transform=val_transform, samples=val_samples
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # 独立测试：验证数据 pipeline 能否正常吐出 batch
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s"
    )

    train_loader, val_loader = get_dataloaders()
    images, labels = next(iter(train_loader))
    print(f"Train batch: {images.shape}, labels={labels}")
    print(f"Label 0 (cat): {(labels == 0).sum().item()}, Label 1 (dog): {(labels == 1).sum().item()}")