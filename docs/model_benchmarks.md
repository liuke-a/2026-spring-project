# 模型配置与性能基准

本文档记录不同模型的训练配置和测试集性能指标，方便对比和复现。

---

## 1. SE-ResNet18 (From Scratch)

| 项目 | 值 |
|------|-----|
| **训练方式** | 从头训练（Scratch） |
| **训练脚本** | `train_scratch.py` |
| **模型架构** | SE-ResNet18（layers=(2,2,2,2), width=64） |
| **参数量** | ~11.7M |
| **输入尺寸** | 224×224 |
| **归一化** | ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) |

### 训练配置

| 超参数 | 值 |
|--------|-----|
| 优化器 | SGD + Momentum(0.9) + Nesterov |
| 学习率 | BaseLR=0.2, MinLR=1e-5（余弦退火 + 3 epoch warmup） |
| Weight Decay | 5e-4 |
| Batch Size | 256 |
| Epochs | 90（EarlyStop patience=15，实际停止于 Epoch 59） |
| 混合精度 | AMP (FP16) + GradScaler |
| TF32 | 开启 |
| torch.compile | mode=default |
| EMA | decay=0.99，对所有浮点 state 统一做 EMA |
| Label Smoothing | 0.1 |
| Dropout | 0.2（分类头前） |
| DropPath | 0.05（沿深度线性递增） |

### 数据增强

| 增强项 | 配置 |
|--------|------|
| RandomResizedCrop | scale=(0.6, 1.0) |
| RandomHorizontalFlip | p=0.5 |
| RandAugment | n=2, m=9 |
| ColorJitter | brightness=0.2, contrast=0.2, saturation=0.2 |
| RandomErasing | p=0.2 |
| Mixup | alpha=0.2, 触发概率 0.5 |
| CutMix | alpha=1.0, 触发后选择概率 0.5 |

### 数据划分

| 项目 | 数量 |
|------|------|
| 总图片数 | 37,500 |
| 训练集 | 22,500（60%） |
| 验证集 | 2,500（6.7%） |
| 测试集 | 12,500（33.3%） |
| 类别 | cat=0, dog=1（均衡） |

### 验证集性能

| 指标 | 值 |
|------|-----|
| **Accuracy** | **0.9748** |
| **Precision** | **0.9781** |
| **Recall** | **0.9710** |
| **F1-Score** | **0.9745** |

### 硬件环境

| 项目 | 值 |
|------|-----|
| GPU | RTX 3080Ti 12GB |
| CPU Workers | 8 |
| GPU 显存占用（训练） | ~4.4GB |

---

## 2. SE-ResNet34 (From Scratch)

与 SE-ResNet18 共享同一套训练管线（`train_scratch.py`、`config_scratch.py`），以下仅列出差异项，其余配置完全一致。

| 项目 | SE-ResNet18 | SE-ResNet34 |
|------|-------------|-------------|
| **模型架构** | layers=(2,2,2,2), width=64 | **layers=(3,4,6,3)**, width=64 |
| **参数量** | ~11.7M | **~21.3M** |
| **EMA Decay** | 0.99 | **0.995**（适配更多 step 数，避免 EMA 滞后） |
| **实际停止 Epoch** | 59（EarlyStop） | **79**（EarlyStop，patience 同为 15） |
| **GPU** | RTX 3080Ti 12GB（~4.4GB） | RTX 3080Ti 12GB **（~6.1GB）** |

### 验证集性能

| 指标 | SE-ResNet18 | SE-ResNet34 |
|------|-------------|-------------|
| **Accuracy** | 0.9748 | **0.9856** |
| **Precision** | 0.9781 | **0.9863** |
| **Recall** | 0.9710 | **0.9847** |
| **F1-Score** | 0.9745 | **0.9855** |

> 34 层模型相比 18 层在所有指标上均有 ~1% 的提升，额外参数量约 82%，推理速度略有下降但仍在可接受范围。


