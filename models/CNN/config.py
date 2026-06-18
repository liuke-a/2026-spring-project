"""
SimpleCNN 训练专用配置。

完全自包含，不依赖项目根目录的 config.py 或 config_scratch.py。
"""

from pathlib import Path
from typing import Tuple

import torch

# ==================== 基础路径 ====================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
LOG_DIR = PROJECT_ROOT / "logs"

for _d in (LOG_DIR,):
    _d.mkdir(parents=True, exist_ok=True)

# ==================== 图像预处理 ====================
IMAGE_SIZE: int = 224
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# ==================== 数据集 ====================
VAL_RATIO: float = 0.1
RANDOM_SEED: int = 42
MAX_SAMPLES: int | None = None

# ==================== 设备 ====================
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== 模型结构 ====================
CNN_DROPOUT: float = 0.5

# ==================== 数据加载 ====================
BATCH_SIZE: int = 64              # SimpleCNN 参数量较小，可用较大 batch
NUM_WORKERS: int = 4

# ==================== 训练超参数 ====================
NUM_EPOCHS: int = 50
WARMUP_EPOCHS: int = 0            # SimpleCNN 从头训练，无需 backbone warmup
EARLY_STOP_PATIENCE: int = 10

# 优化器：AdamW（适合简单 CNN 从头训练）
CNN_LR: float = 1e-3
WEIGHT_DECAY: float = 1e-4

# ==================== 输出路径（独立子目录） ====================
CNN_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "cnn"
CNN_TENSORBOARD_DIR = LOG_DIR / "tensorboard_cnn"

for _d in (CNN_CHECKPOINT_DIR, CNN_TENSORBOARD_DIR):
    _d.mkdir(parents=True, exist_ok=True)
