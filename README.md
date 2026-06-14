# 2026-spring-project

# 🐾 猫狗图片分类

> **项目定位**：基于“产学研用”深度结合原则的工业级人工智能实训项目。
> **核心目标**：构建高鲁棒性、轻量化的动物图像分类模型，并探索端侧推理加速方案，以解决智慧城市场景下的动物管理需求。

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.1-76b900.svg)
<!-- ![License](https://img.shields.io/badge/License-MIT-green.svg) -->

---

## 📖 1. 项目背景与目标
随着数字城市与智慧城市的快速建设，各类系统（社交平台、宠物店监控、街头摄像头）积累了海量动物图像。传统的粗放式管理已无法满足需求。
本项目旨在设计并实现一套**轻量、高效、可部署**的猫狗图像分类系统。不仅要求模型在公开数据集上达到高准确率，更强调在**工业特定约束**（如 RTX 4050 6GB 显存限制）下的系统优化能力（如混合精度训练、推理加速）。

---

## 🛠️ 2. 运行环境与依赖
本项目推荐在以下标准环境中运行，以确保最佳兼容性与性能：

- **操作系统**: Ubuntu 22.04 LTS (或 Windows 11 WSL2)
- **编程语言**: Python 3.8+
- **深度学习框架**: PyTorch 2.x, torchvision 0.15+, torchaudio 2.x
- **硬件加速**: NVIDIA CUDA 12.1, TensorRT 8.6+ (用于推理加速)
- **关键依赖库**: `transformers`, `peft`, `langchain`, `ultralytics`, `mmsdk` (根据实际使用精简)

**一键安装依赖**:
```bash
pip install -r requirements.txt
```
