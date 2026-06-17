"""
从头训练（SE-ResNet）专用配置 — RTX 3080Ti 12GB 优化版。

复用 config.py 的路径、图像尺寸与归一化常量，仅定义从头训练所需的
超参数与独立的 checkpoint / TensorBoard 子目录，避免与预训练迁移学习
流程互相覆盖。

关键优化（3080Ti 12GB vs 原 6GB 配置）：
- BATCH_SIZE 64→256，充分利用大显存与 GPU 并行度
- BASE_LR 按线性缩放 0.05→0.2
- 启用 TF32 / torch.compile / persistent_workers
- 增强数据增强强度（RandAugment + Mixup）
"""

from config import (  # noqa: F401  复用基础常量
    PROJECT_ROOT,
    DATA_DIR,
    TRAIN_DIR,
    TEST_DIR,
    LOG_DIR,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    VAL_RATIO,
    RANDOM_SEED,
    MAX_SAMPLES,
    DEVICE,
)

# ==================== 模型结构 ====================
# (2,2,2,2)=SE-ResNet18，(3,4,6,3)=SE-ResNet34
# MODEL_LAYERS: tuple = (2, 2, 2, 2)  # 18 层
MODEL_LAYERS: tuple = (3, 4, 6, 3)  # 34 层
MODEL_WIDTH: int = 64
DROP_RATE: float = 0.2          # 分类头前 Dropout
DROP_PATH: float = 0.05         # 随机深度最大概率（沿深度线性递增）

# ==================== 数据加载 ====================
# 覆盖 config.py 的默认值（仅 scratch 训练生效，不影响迁移学习）
NUM_WORKERS: int = 8             # 3080Ti 12GB：8 worker 保证数据供给，无瓶颈

# ==================== 训练超参数（针对 RTX 3080Ti 12GB 优化） ====================
BATCH_SIZE: int = 256            # 12GB 显存充裕，大 batch 充分利用 GPU
NUM_EPOCHS: int = 200             # 补偿大 batch 导致的每 epoch 步数减少
WARMUP_EPOCHS: int = 5           # 大 batch 下 warmup 阶段可缩短
EARLY_STOP_PATIENCE: int = 50    # 基于 EMA 验证精度的早停耐心

# 优化器：SGD + momentum + nesterov（从头训练 CNN 配合 cosine 终精度通常优于 Adam）
# 学习率按线性缩放：BASE_LR = 0.05 × (256 / 64) = 0.2
BASE_LR: float = 0.2             # 对应 BATCH_SIZE=256 的基准学习率（线性缩放）
MIN_LR: float = 1e-5             # cosine 退火最小学习率
MOMENTUM: float = 0.9
NESTEROV: bool = True
WEIGHT_DECAY: float = 5e-4

# ==================== 正则与增强 ====================
LABEL_SMOOTHING: float = 0.1
RANDOM_ERASING_PROB: float = 0.2   # 3080Ti 训练快，适度提高擦除概率
RAND_AUGMENT_N: int = 2             # RandAugment 操作数（3080Ti 算力充足，增强更强）
RAND_AUGMENT_M: int = 9             # RandAugment 强度（更强正则，提升泛化）

# Mixup / CutMix（训练循环内按 batch 随机切换其一，或不增强）
MIXUP_ALPHA: float = 0.2
CUTMIX_ALPHA: float = 1.0
MIXUP_PROB: float = 0.5             # 3080Ti 训练快，开启 mixup 增强泛化
MIXUP_SWITCH_PROB: float = 0.5      # 触发后选 cutmix 的概率（否则 mixup）

# ==================== EMA ====================
EMA_DECAY: float = 0.999        # 适配当前 step 数，避免 EMA 过慢长期滞后于 raw 模型

# ==================== 输出路径（独立子目录） ====================
SCRATCH_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "scratch"
SCRATCH_TENSORBOARD_DIR = LOG_DIR / "tensorboard_scratch"

for _d in (SCRATCH_CHECKPOINT_DIR, SCRATCH_TENSORBOARD_DIR):
    _d.mkdir(parents=True, exist_ok=True)
