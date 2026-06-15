"""
全局配置文件。

集中管理路径、超参数与训练配置，避免硬编码。
"""

from pathlib import Path
from typing import Tuple

# 项目根目录（config.py 所在目录）
PROJECT_ROOT = Path(__file__).parent.resolve()

# 数据路径
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

# 输出路径
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"

# 自动创建不存在的目录
for d in [CHECKPOINT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 图像预处理参数（ImageNet 标准，适配预训练模型）
IMAGE_SIZE: int = 224
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# 训练超参数（针对 RTX 4050 6G 优化）
BATCH_SIZE: int = 32          # 6G 显存安全值，后续可试探 24
NUM_WORKERS: int = 4            # Ubuntu 笔记本建议 4，CPU 吃紧可改 2
VAL_RATIO: float = 0.1         # 从 train 目录划分 10% 做验证
RANDOM_SEED: int = 42

# 日志格式
LOG_FORMAT: str = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
LOG_FILE = LOG_DIR / "train.log"


# ==================== 训练配置 ====================

# 训练超参数
NUM_EPOCHS: int = 30
WARMUP_EPOCHS: int = 3          # 先冻结 backbone 训练 3 轮
EARLY_STOP_PATIENCE: int = 5   # 验证集 5 轮不提升则停止

# 优化器
LR_BACKBONE: float = 1e-4     # 预训练部分学习率
LR_HEAD: float = 2e-3         # 新加分类头学习率
WEIGHT_DECAY: float = 1e-4

# 设备
DEVICE: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# TensorBoard
TENSORBOARD_DIR = LOG_DIR / "tensorboard"
TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)