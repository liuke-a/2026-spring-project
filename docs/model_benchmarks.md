# 模型配置与性能基准

本文档记录不同模型的训练配置和性能指标。

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

**Accuracy 与训练时一致**

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

**Accuracy 与训练时不一致，训练时 Accuracy 为 98.84%**

| 指标 | SE-ResNet18 | SE-ResNet34 |
|------|-------------|-------------|
| **Accuracy** | 0.9748 | **0.9856** |
| **Precision** | 0.9781 | **0.9863** |
| **Recall** | 0.9710 | **0.9847** |
| **F1-Score** | 0.9745 | **0.9855** |

> 34 层模型相比 18 层在所有指标上均有 ~1% 的提升，额外参数量约 82%，推理速度略有下降但仍在可接受范围。

---

## 3. SE-ResNet34-Big（加宽变体）

在 SE-ResNet34 基础上将基础通道数从 64 扩展至 96（1.5×），所有 stage 的通道等比放大。训练管线与 SE-ResNet34 完全一致（`train_scratch.py` + `config_scratch.py`），仅 `MODEL_WIDTH` 改为 96。验证准确率突破 99%。

### 差异对比（vs SE-ResNet34）

| 项目 | SE-ResNet34 | SE-ResNet34-Big |
|------|-------------|-----------------|
| **模型架构** | layers=(3,4,6,3), width=64 | layers=(3,4,6,3), **width=96** |
| **网络宽度** | 64 → 128 → 256 → 512 | **96 → 192 → 384 → 768** |
| **参数量** | ~21.3M | **~48.2M**（2.26×） |
| **FP16 权重体积** | 43.0 MB | **92.1 MB** |
| **最佳 Epoch** | 79 | **180** |
| **EMA Decay** | 0.995 | **0.999**（适配更多 step） |
| **EMA num_updates** | 14,275 | **31,675** |
| **GPU 显存（训练）** | ~6.1 GB | **~8.5 GB** |

### 验证集性能

**Accuracy 与训练时不一致，训练时 Accuracy 为 99.00%**

| 指标 | SE-ResNet34 | SE-ResNet34-Big |
|------|-------------|-----------------|
| **Accuracy** | 0.9856 | **0.9980** |
| **Precision** | 0.9863 | **0.9984** |
| **Recall** | 0.9847 | **0.9976** |
| **F1-Score** | 0.9855 | **0.9980** |

> 加宽变体将验证准确率从 98.56% 提升至 99.00%，在 2,500 张验证集上仅错分 25 张。代价是参数量翻倍（~48M vs ~21M），推理延迟增加约 50%，适合对精度要求极高的场景。

### 推理权重

| 文件 | 格式 | 大小 |
|------|------|------|
| `checkpoints/scratch/best_model.pth` | FP32 完整 checkpoint（含 optim/scheduler/scaler） | 552 MB |
| `weights/SEResNet34_big_fp16.pth` | FP16 纯推理权重（EMA） | 92.1 MB |

```python
from service import get_model_service
svc = get_model_service("seresnet34-big")
result = svc.predict("cat.jpg")  # → {"label": "cat", "confidence": 0.XXXX}
```

---

## 4. SimpleCNN（From Scratch）

SimpleCNN 是一个轻量级 VGG 风格卷积网络，独立于 SE-ResNet 系列，由 `models/CNN/` 子模块管理，拥有独立的训练、评估和预测脚本。

| 项目 | 值 |
|------|-----|
| **训练方式** | 从头训练（Scratch） |
| **训练脚本** | `models/CNN/train.py` |
| **模型架构** | SimpleCNN — 5 个 Conv Block（VGG 风格），每块 2×Conv-BN-ReLU + MaxPool |
| **通道数** | 3 → 32 → 64 → 128 → 256 → 512 |
| **分类头** | AdaptiveAvgPool2d → Dropout(0.5) → FC(512→256) → ReLU → Dropout(0.3) → FC(256→2) |
| **参数量** | **4.85M**（最轻量） |
| **输入尺寸** | 224×224 |
| **归一化** | ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) |

### 训练配置

| 超参数 | 值 |
|--------|-----|
| 优化器 | AdamW |
| 学习率 | BaseLR=1e-3, MinLR=1e-6（余弦退火） |
| Weight Decay | 1e-4 |
| Batch Size | 64 |
| Epochs | 50（EarlyStop patience=10，最佳 Epoch=40） |
| 混合精度 | AMP (FP16) + GradScaler |
| Dropout | 0.5（第一层），0.3（第二层） |

> SimpleCNN 从头训练，不依赖预训练权重，无需 warmup。使用 AdamW 优化器（非 SGD），适合小模型快速收敛。

### 数据增强

| 增强项 | 配置 |
|--------|------|
| Resize + CenterCrop | 推理时 resize 至 256（1.14×），再中心裁剪至 224×224 |
| Normalization | ImageNet 统计值 |

> SimpleCNN 的数据管线与 SE-ResNet 系列复用同一套 `get_dataloaders()`，训练时默认包含 RandomResizedCrop、RandomHorizontalFlip 等基础增强。

### 验证集性能

| 指标 | 值 |
|------|-----|
| **Accuracy** | **0.9732** |
| **Precision** | **0.9688** |
| **Recall** | **0.9774** |
| **F1-Score** | **0.9731** |

### 硬件环境

| 项目 | 值 |
|------|-----|
| GPU | RTX 4050 6GB |
| CPU Workers | 4 |
| GPU 显存占用（训练） | < 2GB（轻量级） |

---
## 5. Kaggle 测试集性能

所有模型在 [Dogs vs. Cats Redux: Kernels Edition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition) 官方测试集（12,500 张）上评估，同时对比 FP32 原始权重与 FP16 压缩权重的精度差异。

### 评分指标

使用 **多类别对数损失（Log Loss）**，越小越好：

$$
\text{LogLoss} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i) \right]
$$

其中 $n=12{,}500$ 为测试集图片数，$\hat{y}_i$ 为模型预测图片是 "dog" 的概率，$y_i \in \{0, 1\}$ 为真实标签（1=dog, 0=cat）。

提交格式为 CSV，每行包含图片 id 和预测为 dog 的概率：

```csv
id,label
1,0.5
2,0.5
```

### 结果对比

| 模型 | FP32 LogLoss | FP16 LogLoss | Δ | 说明 |
|------|:-----------:|:-----------:|:---:|------|
| SimpleCNN | 0.19002 | 0.18998 | −0.00004 | FP16 略优于 FP32（精度噪声） |
| SE-ResNet18 | 0.14455 | 0.14457 | +0.00002 | 精度损失可忽略 |
| SE-ResNet34 | 0.13076 | 0.13076 | **0** | 完全一致 |
| SE-ResNet34-Big | 0.10194 | 0.10195 | +0.00001 | 精度损失可忽略 |

> **结论**：FP16 压缩对精度的损失在 $10^{-5}$ 量级，完全可以忽略；12,500 张测试集上预测结果几乎逐样本一致。使用 FP16 推理权重可将磁盘占用和 GPU 显存减半，强烈推荐部署时使用。

### LogLoss 对比

```
SimpleCNN               ██████████████████████████████████ 0.19002
SE-ResNet18              ████████████████████████████ 0.14455
SE-ResNet34                ██████████████████████▌ 0.13076
SE-ResNet34-Big              ████████████████ 0.10194
                          (越低越好)
```

> SimpleCNN 作为 4.85M 参数的轻量模型，LogLoss 为 0.19，相比 SE-ResNet 系列有一定差距，但考虑到其仅 1/2.4~1/10 的参数量，性价比极高，适合资源受限的部署场景。SE-ResNet 系列的残差连接与通道注意力机制在猫狗分类任务上带来了显著的增益。

## 6. 模型性能对比

|模型| CNN | SE-ResNet18 | SE-ResNet34 | SE-ResNet34-Big |
|:-:|:-:|:-:|:-:|:-:|
|fp16 参数量| **4.85M** | ~11.7M | ~21.3M | ~48.2M |
|fp16 权重体积| **9.3 MB** | 21.6 MB | 41.0 MB | 92.1 MB |
|Accuracy| 0.9732 | 0.9748 | 0.9856 | **0.9980** |
|Precision| 0.9688 | 0.9781 | 0.9863 | **0.9984** |
|Recall| 0.9774 | 0.9710 | 0.9847 | **0.9976** |
|F1-Score| 0.9731 | 0.9745 | 0.9855 | **0.9980** |
|GPU 显存占用（训练）| **< 2 GB** | ~4.4 GB | ~6.1 GB | ~8.5 GB |
|kaggle LogLoss| 0.19002 | 0.14455 | 0.13076 | **0.10194** |
|kaggle LogLoss（fp16）| 0.18998 | 0.14457 | 0.13076 | **0.10195** |