# 2026-spring-project

# 0. 实验课程概述与通用要求
### **课程目标**
  本课程旨在通过"产学研用"深度结合的原则，培养大三学生解决复杂人工智能工程问题的能力。实训内容聚焦于工业界前沿架构（如问答系统、视觉语言模型、端侧推理加速等），要求学生不仅能够复现基础模型，更要具备针对工业特定约束进行系统优化的思维。

### 代码规范与学术诚信要求
- 工业级代码规范：  核心模块必须包含完善的异常处理机制（try-except 结构）及详尽的日志记录。
- 版本控制：  实验过程需全程使用 Git 管理，提交记录需符合代码规范。
- 文档化要求：  每个项目需提供符合行业标准的 README.md，包含算法流程、参数设置及复现步骤。
- 学术诚信：  严禁任何形式的代码抄袭。实验报告中引用的研究成果、论文算法及开源权重必须明确标注来源。

### 选题：猫狗图片分类
**项目目标**

  随着数字城市、智慧城市的快速建设，各类的图像识别的应用应有尽有，各系统都积累了大量的用户图片。例如，用户在社交平台上分享的动物图片、宠物店的监控中获得的动物图片、城市街头摄像头拍下的动物图片等。积累了相当多的动物图片后，城市对于动物的管理问题逐渐浮出水面。因此，急需对于动物的智能分类算法。

**技术要求**
1. 根据猫狗图片的原始数据，需要将原始数据处理为同一维度的数据；
2. 猫狗数据图像较大，像素点较多，会面对特征较多的问题，需要处理特征；
3. 设计对应的卷积神经网络模型，进行数据训练；
4. 模型训练完成后，对数据进行预测。

----
# 🐾 猫狗图片分类

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)
![CUDA](https://img.shields.io/badge/CUDA-13-76b900.svg)

## 📖 1. 项目背景与目标
随着数字城市与智慧城市的快速建设，各类系统（社交平台、宠物店监控、街头摄像头）积累了海量动物图像。传统的粗放式管理已无法满足需求。  
本项目旨在设计并实现一套**轻量、高效、可部署**的猫狗图像分类系统。不仅要求模型在公开数据集上达到高准确率，更强调在**工业特定约束**下的系统优化能力。

## 🧠 2. 模型架构：SE-ResNet

本项目采用自定义 **SE-ResNet**（Squeeze-and-Excitation ResNet）从头训练，不依赖任何预训练权重。

### 2.1 核心模块
| 模块 | 说明 |
|------|------|
| `SEBlock` | Squeeze-and-Excitation 通道注意力：全局平均池化 → 1x1 降维 + ReLU → 1x1 升维 + Sigmoid → 逐通道缩放 |
| `SEBasicBlock` | ResNet BasicBlock + SE 通道注意力 + 可选 DropPath（随机深度） |
| `SEResNet` | 标准 4 阶段 ResNet 骨干（7x7 stem + maxpool + 4 stage），深度可配置 |

### 2.2 模型深度配置
- **SE-ResNet18**：`layers=(2,2,2,2)`，共 18 层
- **SE-ResNet34**：`layers=(3,4,6,3)`，共 34 层

通过 `config_scratch.py` 中的 `MODEL_LAYERS` 参数切换。

### 2.3 训练策略
- **优化器**：SGD + momentum(0.9) + Nesterov
- **学习率调度**：线性 warmup(3 epochs) + 余弦退火
- **正则化**：Label Smoothing(0.1) + Mixup/CutMix + RandAugment + RandomErasing + DropPath + Dropout(0.2)
- **加速**：混合精度(AMP) + channels_last + TF32 + torch.compile
- **权重平滑**：EMA（指数滑动平均，decay=0.995）
- **早停**：基于 EMA 验证精度，耐心值 15 epochs

## 🛠️ 3. 运行环境与依赖

- **操作系统**: Ubuntu 24.04 LTS
- **编程语言**: Python 3.10+
- **深度学习框架**: PyTorch 2.x, torchvision 0.15+
- **硬件加速**: NVIDIA CUDA 13（推荐 RTX 3080Ti 12GB）
- **关键依赖库**: `tensorboard`, `tqdm`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`

**克隆仓库**:
```bash
# 普通克隆（会下载完整的模型权重 fp16 文件，> 100 MB）
git clone https://github.com/liuke-a/2026-spring-project.git

# 跳过 LFS 模型权重下载（仅获取指针文件，~130 bytes/个）
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/liuke-a/2026-spring-project.git
```

**git pull 时跳过模型权重**:
```bash
GIT_LFS_SKIP_SMUDGE=1 git pull
```

之后需要权重时
```bash
git lfs pull
```

或只拉特定文件：
```bash
git lfs pull --include="weights/SEResNet18_fp16.pth"
```


> 💡 模型权重（`weights/*_fp16.pth`）已托管在 Git LFS 上。使用 `GIT_LFS_SKIP_SMUDGE=1` 可以跳过权重下载，大幅加快克隆速度。如需后续拉取权重，执行 `git lfs pull`。

**一键安装依赖**:
```bash
pip install -r requirements.txt
```

## 4. 项目结构
```text
2026-spring-project/
├── data/                       # 数据集目录（gitignored）
│   ├── train/                  # 训练集（cat 与 dog 图片混合存放）
│   └── test/                   # 测试集
├── models/                     # 模型定义
│   └── seresnet.py             # SE-ResNet 模型（SEBlock + SEBasicBlock + SEResNet）
├── utils/                      # 工具函数
│   ├── dataset.py              # 数据集加载与预处理
│   ├── logger.py               # 日志配置
│   └── eda.py                  # 数据探索分析
├── docs/                       # 项目文档
│   ├── PROJECT_PLAN.md         # 项目计划书
│   ├── EDA_REPORT.md           # EDA 报告
│   └── model_benchmarks.md     # 模型对比记录
├── checkpoints/                # 模型权重（gitignored）
│   └── scratch/                # 从头训练的断点
├── logs/                       # 训练日志（gitignored）
├── config_scratch.py           # 全局配置（路径、超参数、模型结构）
├── train_scratch.py            # 训练脚本
├── evaluate_scratch.py         # 评估脚本（验证集指标 + 混淆矩阵）
├── predict_scratch.py          # 推理预测脚本（生成 submission.csv）
└── requirements.txt            # 依赖列表
```

## 5. 数据准备

### 5.1 数据来源
本实训提供的数据集名为 **"猫狗识别的数据及配套文件"**，已通过课程网盘获取。数据集已预先划分为训练集与测试集。

### 5.2 目录结构
```text
data/
├── train/          # 训练集（含标签，cat 与 dog 图片混合存放）
│   ├── cat.001.jpg
│   ├── dog.001.jpg
│   └── ...
└── test/           # 测试集（用于预测）
    ├── 001.jpg
    └── ...
```

### 5.3 数据概况
训练集共 **25,000 张 JPEG 图片**，类别均衡（猫 12,500 / 狗 12,500），无损坏文件。
原始尺寸范围 42×32 ~ 1050×768，经 EDA 确认绝大多数图片尺寸大于 224×224，
统一 resize 至 224×224 不会造成显著信息损失。

## 6. 使用说明

### 6.1 训练
```bash
# 从头训练 SE-ResNet
python train_scratch.py

# 从断点恢复训练
python train_scratch.py --resume
```

训练过程中的关键配置（`config_scratch.py`）：
- `MODEL_LAYERS`：模型深度，`(2,2,2,2)` 为 ResNet18，`(3,4,6,3)` 为 ResNet34
- `BATCH_SIZE`：默认 256（RTX 3080Ti 12GB）
- `NUM_EPOCHS`：默认 90
- `BASE_LR`：默认 0.2（线性缩放）

训练日志实时输出到终端和 `logs/train.log`，TensorBoard 日志写入 `logs/tensorboard_scratch/`。

### 6.2 评估
```bash
python evaluate_scratch.py
```
在验证集上计算 Accuracy / Precision / Recall / F1-Score，并绘制混淆矩阵保存至 `logs/confusion_matrix_scratch.png`。

### 6.3 预测
```bash
python predict_scratch.py
```
对 `data/test/` 目录下的图片批量推理，输出预测标签与置信度到 `submission_scratch.csv`。

## 7. 配置说明

所有配置集中在 `config_scratch.py`，主要参数如下：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_LAYERS` | `(2,2,2,2)` | 模型深度（18层/34层） |
| `MODEL_WIDTH` | `64` | 起始通道数 |
| `BATCH_SIZE` | `256` | 训练批大小 |
| `NUM_EPOCHS` | `90` | 最大训练轮数 |
| `BASE_LR` | `0.2` | 基础学习率（线性缩放） |
| `WARMUP_EPOCHS` | `3` | 热身轮数 |
| `EARLY_STOP_PATIENCE` | `15` | 早停耐心值 |
| `DROP_RATE` | `0.2` | 分类头前 Dropout |
| `DROP_PATH` | `0.05` | 随机深度最大概率 |
| `LABEL_SMOOTHING` | `0.1` | 标签平滑系数 |
| `MIXUP_PROB` | `0.5` | Mixup/CutMix 触发概率 |
| `EMA_DECAY` | `0.995` | EMA 衰减系数 |
