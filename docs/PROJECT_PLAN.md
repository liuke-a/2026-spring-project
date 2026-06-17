# 🐾 项目一：猫狗图片分类 - 工业级实训工作大纲 (Project Plan)

> **项目背景**：基于“产学研用”原则，构建工业级动物智能分类算法，解决智慧城市场景下的图像识别需求。
> **硬件环境**：RTX 4050 (6GB VRAM), CUDA 12.1
> **数据规模**：13,000 张原始图像 (Train/Val/Test = 8:1:1)
> **技术栈**：Python 3.8+, PyTorch 2.x, TensorRT 8.6+, Git


---

## 📋 详细任务拆解 (Checklist)

### 📌 阶段一：工程初始化与环境搭建
- [ ] **1.1 目录结构搭建**：创建 `data/`, `src/`, `models/`, `utils/`, `logs/`, `docs/` 标准目录。
- [ ] **1.2 版本控制初始化**：执行 `git init`，配置 `.gitignore`（排除 `data/`, `*.pth`, `__pycache__`, `logs/`），创建 `main` 和 `dev` 分支。
- [ ] **1.3 环境配置**：配置 Python 3.8+, PyTorch 2.x, CUDA 12.1，生成并锁定 `requirements.txt`。
- [ ] **1.4 日志框架预埋**：在 `utils/logger.py` 中配置 `logging` 模块，设定包含时间、级别、模块名的标准日志格式。
- **📦 阶段交付物**：干净的 Git 初始 Commit、`requirements.txt`、基础项目骨架。

### 📌 阶段二：数据工程与特征处理
- [ ] **2.1 数据探索与清洗 (EDA)**：编写脚本遍历 13,000 张图片，使用 `try-except` 捕获并剔除损坏/无法读取的异常图片，统计猫狗比例。
- [ ] **2.2 数据集划分**：按 8:1:1 比例划分 Train (10,400张)、Val (1,300张)、Test (1,300张)。
- [ ] **2.3 统一维度与预处理**：使用 `torchvision.transforms`，将图像 Resize/Crop 至 **224×224**，并进行 Normalize。
- [ ] **2.4 数据增强 (防过拟合)**：在 Train 集上加入 RandomHorizontalFlip, RandomRotation(15), ColorJitter。
- [ ] **2.5 DataLoader 封装**：编写 `src/dataset.py`，设置 `num_workers=4`，利用 CPU 多核加速预处理，减轻 GPU 负担。
- **📦 阶段交付物**：`dataset.py`（含完善异常处理）、数据分布统计图、清洗后的数据集。

### 📌 阶段三：模型设计与构建
- [ ] **3.1 基线模型选择**：选用 `torchvision.models.resnet18` 或 `mobilenet_v3_large`（参数量小，适配 6GB 显存）。
- [ ] **3.2 模型定制**：替换最后的 Fully Connected (FC) 层，将输出维度修改为 `in_features -> 2` (猫/狗)。
- [ ] **3.3 模型封装**：在 `src/model.py` 中封装模型，添加 Type Hints（类型提示）和 Docstring（文档字符串）。
- **📦 阶段交付物**：`model.py`、模型结构打印输出（`print(model)`）截图。

### 📌 阶段四：模型训练与系统优化
- [ ] **4.1 训练配置**：定义 `CrossEntropyLoss`，优化器选用 `AdamW` (lr=1e-3)，配置 `CosineAnnealingLR` 学习率衰减。
- [ ] **4.2 🌟 核心优化：开启混合精度 (AMP)**：在训练循环中引入 `torch.cuda.amp.autocast()` 和 `GradScaler`，降低显存占用并提升训练速度。
- [ ] **4.3 训练循环编写**：实现标准的 Epoch 循环，包含前向传播、损失计算、反向传播、梯度清零。
- [ ] **4.4 监控与保存**：集成 TensorBoard 记录 Loss/Accuracy；实现 **Early Stopping**（耐心值设为 5）和 **Model Checkpoint**（仅保存 Val 集表现最佳的权重）。
- **📦 阶段交付物**：`train.py`、TensorBoard 训练曲线截图、最佳权重文件 `best_model.pth`。

### 📌 阶段五：模型评估与推理加速
- [ ] **5.1 多维度评估**：在独立的 Test 集上运行 `evaluate.py`，计算 Accuracy、Precision、Recall、F1-Score。
- [ ] **5.2 混淆矩阵绘制**：使用 `seaborn` 或 `matplotlib` 绘制混淆矩阵，分析模型误判情况。
- [ ] **5.3 推理接口封装**：编写 `predict.py`，提供 `predict(image_path)` 函数，输出分类结果及置信度。
- [ ] **5.4 🌟 加分项：推理加速 (TensorRT)**：将 `.pth` 模型导出为 `.onnx` 格式，使用 TensorRT 8.6+ 构建 Engine，对比 PyTorch 原生推理与 TensorRT 推理的 FPS 和延迟。
- **📦 阶段交付物**：`evaluate.py`、`predict.py`、混淆矩阵图、推理速度对比数据表。

### 📌 阶段六：工程化交付与文档化
- [ ] **6.1 代码审查 (Code Review)**：全局检查 `try-except` 是否覆盖文件读写，检查日志是否完备，确保无硬编码路径。
- [ ] **6.2 编写行业标准 README.md**：包含项目简介、环境安装、算法流程图、核心参数设置与复现步骤。
- [ ] **6.3 学术诚信声明**：在 README 和实训报告中明确标注：数据集来源、使用的预训练模型来源、参考的开源代码。
- [ ] **6.4 最终 Git 提交**：将所有代码 push 到远程仓库，清理无用分支，打上 `v1.0-final` 标签。
- **📦 阶段交付物**：最终版 `README.md`、符合规范的实训报告、整洁的 Git 仓库链接。

---

## 📝 开发日志与踩坑记录 (Dev Log)
*(在此处记录开发过程中遇到的关键问题及解决方案，例如：)*
- `[2026-06-16]` 遇到 RTX 4050 OOM 问题。解决方案：将 Batch Size 从 32 降至 16，并成功开启 AMP 混合精度训练，显存占用从 5.8GB 降至 3.2GB，训练速度提升 30%。
- `[2026-06-18]` ...

```

### 💡 如何使用这份文件：
1. **复制粘贴**：将上述代码块中的所有内容复制。
2. **创建文件**：在你的项目根目录下新建 `docs` 文件夹，然后在里面新建 `PROJECT_PLAN.md` 文件并粘贴。
3. **动态更新**：在 VS Code 或 Typora 中打开它，每完成一项任务，就把 `- [ ]` 手动改成 `- [x]`。
4. **报告素材**：在撰写最终实训报告时，直接把“详细任务拆解”和“开发日志”部分复制过去，稍加润色，就是一篇逻辑严密、体现工程素养的“项目实施过程”章节。

如果你需要我为你生成下一阶段的具体代码（比如带异常处理和日志的 `dataset.py` 或开启 AMP 的 `train.py`），随时告诉我！